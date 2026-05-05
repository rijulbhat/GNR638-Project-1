#!/usr/bin/env python3
"""Deterministic stitcher for shuffled, overlapping 128x128 map patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


PATCH_SIZE = 128
PATCH_RE = re.compile(r"patch_(\d+)\.png$")


@dataclass(frozen=True)
class OrientedPatch:
    state_id: int
    patch_id: int
    patch_pos: int
    rotation: int
    image: np.ndarray


@dataclass
class CandidateResult:
    rows: int
    cols: int
    overlap: int
    stride: int
    total_cost: float
    mean_seam_score: float
    max_seam_score: float
    placements: Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch shuffled overlapping map patches.")
    parser.add_argument("--patch-dir", default="patches", help="Directory containing patch_*.png files.")
    parser.add_argument("--out-image", default="outputs/reconstructed.png", help="Output stitched image path.")
    parser.add_argument("--out-meta", default="outputs/placements.json", help="Output placement metadata JSON path.")
    parser.add_argument("--out-seams", default="outputs/seam_scores.csv", help="Output seam scores CSV path.")
    parser.add_argument("--beam-width", type=int, default=500, help="Beam width for row-major layout search.")
    parser.add_argument("--top-k", type=int, default=40, help="Top edge candidates kept per oriented tile.")
    parser.add_argument("--overlap-min", type=int, default=8, help="Minimum candidate overlap in pixels.")
    parser.add_argument("--overlap-max", type=int, default=96, help="Maximum candidate overlap in pixels.")
    parser.add_argument("--overlap-candidates", type=int, default=5, help="Number of overlap values to solve.")
    parser.add_argument("--max-grid-candidates", type=int, default=8, help="Maximum grid shapes to solve.")
    parser.add_argument("--max-aspect-ratio", type=float, default=6.0, help="Skip very elongated grid candidates.")
    parser.add_argument("--descriptor-bins", type=int, default=32, help="Long-axis bins for edge descriptors.")
    parser.add_argument(
        "--perfect-threshold",
        type=float,
        default=1e-6,
        help="Stop searching once both mean and max seam scores are at or below this value.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reserved for reproducible tie-breaking.")
    return parser.parse_args()


def patch_id_from_path(path: Path) -> int:
    match = PATCH_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid patch filename: {path.name}")
    return int(match.group(1))


def load_patches(patch_dir: Path) -> Tuple[List[int], List[np.ndarray]]:
    paths = sorted(patch_dir.glob("patch_*.png"), key=patch_id_from_path)
    if not paths:
        raise FileNotFoundError(f"No patch_*.png files found in {patch_dir}")

    global PATCH_SIZE
    first_image = Image.open(paths[0]).convert("RGB")
    PATCH_SIZE = first_image.size[0]
    if first_image.size[0] != first_image.size[1]:
        raise ValueError(f"{paths[0]} is {first_image.size}; expected a square patch")

    patch_ids: List[int] = []
    patches: List[np.ndarray] = []
    for path in paths:
        patch_id = patch_id_from_path(path)
        image = Image.open(path).convert("RGB")
        if image.size != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"{path} is {image.size}; expected {(PATCH_SIZE, PATCH_SIZE)}")
        patch_ids.append(patch_id)
        patches.append(np.asarray(image, dtype=np.uint8))

    if patch_ids[0] != 0:
        raise ValueError("patch_0.png is required and must be present.")
    return patch_ids, patches


def make_oriented_patches(patch_ids: Sequence[int], patches: Sequence[np.ndarray]) -> List[OrientedPatch]:
    oriented: List[OrientedPatch] = []
    for patch_pos, (patch_id, image) in enumerate(zip(patch_ids, patches)):
        rotations = [0] if patch_id == 0 else [0, 90, 180, 270]
        for rotation in rotations:
            k = (rotation // 90) % 4
            rotated = np.rot90(image, k=k).copy()
            oriented.append(
                OrientedPatch(
                    state_id=len(oriented),
                    patch_id=patch_id,
                    patch_pos=patch_pos,
                    rotation=rotation,
                    image=rotated,
                )
            )
    return oriented


def factor_grid_candidates(num_patches: int, max_aspect_ratio: float, limit: int) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    root = int(math.sqrt(num_patches))
    for rows in range(1, root + 1):
        if num_patches % rows != 0:
            continue
        cols = num_patches // rows
        for shape in ((rows, cols), (cols, rows)):
            aspect = max(shape) / min(shape)
            if aspect <= max_aspect_ratio and shape not in candidates:
                candidates.append(shape)

    candidates.sort(key=lambda rc: (abs(math.log(rc[1] / rc[0])), -min(rc), rc[0], rc[1]))
    if limit > 0:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError(f"No grid candidates found for {num_patches} patches.")
    return candidates


def edge_strip(image: np.ndarray, side: str, overlap: int) -> np.ndarray:
    if side == "left":
        return image[:, :overlap, :]
    if side == "right":
        return image[:, PATCH_SIZE - overlap :, :]
    if side == "top":
        return image[:overlap, :, :]
    if side == "bottom":
        return image[PATCH_SIZE - overlap :, :, :]
    raise ValueError(f"Unknown side: {side}")


def strip_score(strip_a: np.ndarray, strip_b: np.ndarray) -> float:
    a = strip_a.astype(np.float32)
    b = strip_b.astype(np.float32)
    color = np.abs(a - b).mean()

    # Simple texture term. It makes flat regions less overconfident and helps align labels/roads.
    grad = 0.0
    if a.shape[0] > 1:
        grad += np.abs(np.diff(a, axis=0) - np.diff(b, axis=0)).mean()
    if a.shape[1] > 1:
        grad += np.abs(np.diff(a, axis=1) - np.diff(b, axis=1)).mean()
    return float(color + 0.25 * grad)


def color_strip_score(strip_a: np.ndarray, strip_b: np.ndarray) -> float:
    return float(np.abs(strip_a.astype(np.float32) - strip_b.astype(np.float32)).mean())


def downsample_descriptor(strip: np.ndarray, bins: int) -> np.ndarray:
    """Small edge descriptor used only for candidate pruning."""
    arr = strip.astype(np.float32)
    h, w, _ = arr.shape
    y_bins = bins if h >= w else max(2, min(8, h))
    x_bins = bins if w > h else max(2, min(8, w))

    y_edges = np.linspace(0, h, y_bins + 1, dtype=np.int32)
    x_edges = np.linspace(0, w, x_bins + 1, dtype=np.int32)
    cells: List[np.ndarray] = []
    for yi in range(y_bins):
        for xi in range(x_bins):
            block = arr[y_edges[yi] : y_edges[yi + 1], x_edges[xi] : x_edges[xi + 1]]
            cells.append(block.mean(axis=(0, 1)))
    desc = np.concatenate(cells).astype(np.float32)
    desc -= desc.mean()
    norm = np.linalg.norm(desc)
    if norm > 1e-6:
        desc /= norm
    return desc


def descriptor_distance_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Squared L2 using matrix multiplication, clipped for tiny negative numerical noise.
    s_norm = np.sum(source * source, axis=1, keepdims=True)
    t_norm = np.sum(target * target, axis=1, keepdims=True).T
    dist = s_norm + t_norm - 2.0 * (source @ target.T)
    np.maximum(dist, 0.0, out=dist)
    return dist


def infer_overlaps(
    oriented: Sequence[OrientedPatch],
    overlap_min: int,
    overlap_max: int,
    keep: int,
) -> List[int]:
    """Rank overlaps by how well patch_0's right/bottom edges find matching neighbors."""
    anchor = oriented[0]
    scored: List[Tuple[float, int, float, float]] = []
    overlap_max = min(overlap_max, PATCH_SIZE - 1)

    for overlap in tqdm(range(overlap_min, overlap_max + 1), desc="Inferring overlap"):
        right = edge_strip(anchor.image, "right", overlap)
        bottom = edge_strip(anchor.image, "bottom", overlap)
        best_right = float("inf")
        best_down = float("inf")
        for target in oriented[1:]:
            if target.patch_id == anchor.patch_id:
                continue
            best_right = min(best_right, color_strip_score(right, edge_strip(target.image, "left", overlap)))
            best_down = min(best_down, color_strip_score(bottom, edge_strip(target.image, "top", overlap)))
        scored.append((best_right + best_down, overlap, best_right, best_down))

    scored.sort(key=lambda item: (item[0], item[1]))
    overlaps: List[int] = []
    best_total = scored[0][0]
    deltas: List[float] = []
    for idx, (total, overlap, right_score, down_score) in enumerate(scored):
        if idx > 0:
            delta = total - scored[idx - 1][0]
            deltas.append(delta)
            if len(overlaps) >= 2:
                median_delta = float(np.median(deltas)) if deltas else 0.0
                if total > best_total * 1.25 and delta > max(best_total * 0.5, median_delta * 3.0):
                    break
        overlaps.append(overlap)
        if len(overlaps) >= keep:
            break
    print("Selected overlap candidates:")
    for total, overlap, right_score, down_score in scored[: len(overlaps)]:
        print(f"  overlap={overlap:3d} score={total:.4f} right={right_score:.4f} down={down_score:.4f}")
    return overlaps


def build_edge_candidates(
    oriented: Sequence[OrientedPatch],
    overlap: int,
    top_k: int,
    descriptor_bins: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    left_desc = np.stack([downsample_descriptor(edge_strip(p.image, "left", overlap), descriptor_bins) for p in oriented])
    right_desc = np.stack([downsample_descriptor(edge_strip(p.image, "right", overlap), descriptor_bins) for p in oriented])
    top_desc = np.stack([downsample_descriptor(edge_strip(p.image, "top", overlap), descriptor_bins) for p in oriented])
    bottom_desc = np.stack([downsample_descriptor(edge_strip(p.image, "bottom", overlap), descriptor_bins) for p in oriented])

    patch_ids = np.array([p.patch_id for p in oriented])
    horizontal_dist = descriptor_distance_matrix(right_desc, left_desc)
    vertical_dist = descriptor_distance_matrix(bottom_desc, top_desc)
    same_patch = patch_ids[:, None] == patch_ids[None, :]
    horizontal_dist[same_patch] = np.inf
    vertical_dist[same_patch] = np.inf

    k = min(top_k, len(oriented) - 1)
    horizontal: List[List[int]] = []
    vertical: List[List[int]] = []
    for idx in range(len(oriented)):
        h_idx = np.argpartition(horizontal_dist[idx], k)[:k]
        h_idx = h_idx[np.argsort(horizontal_dist[idx, h_idx])]
        v_idx = np.argpartition(vertical_dist[idx], k)[:k]
        v_idx = v_idx[np.argsort(vertical_dist[idx, v_idx])]
        horizontal.append([int(v) for v in h_idx if np.isfinite(horizontal_dist[idx, v])])
        vertical.append([int(v) for v in v_idx if np.isfinite(vertical_dist[idx, v])])
    return horizontal, vertical


def solve_layout(
    oriented: Sequence[OrientedPatch],
    rows: int,
    cols: int,
    overlap: int,
    beam_width: int,
    horizontal_candidates: Sequence[Sequence[int]],
    vertical_candidates: Sequence[Sequence[int]],
) -> CandidateResult | None:
    patch_pos_by_state = [p.patch_pos for p in oriented]
    exact_cache: Dict[Tuple[str, int, int], float] = {}

    def h_score(left_state: int, right_state: int) -> float:
        key = ("h", left_state, right_state)
        if key not in exact_cache:
            exact_cache[key] = strip_score(
                edge_strip(oriented[left_state].image, "right", overlap),
                edge_strip(oriented[right_state].image, "left", overlap),
            )
        return exact_cache[key]

    def v_score(top_state: int, bottom_state: int) -> float:
        key = ("v", top_state, bottom_state)
        if key not in exact_cache:
            exact_cache[key] = strip_score(
                edge_strip(oriented[top_state].image, "bottom", overlap),
                edge_strip(oriented[bottom_state].image, "top", overlap),
            )
        return exact_cache[key]

    start_mask = 1 << oriented[0].patch_pos
    beam: List[Tuple[float, float, Tuple[int, ...], int]] = [(0.0, 0.0, (0,), start_mask)]
    total_cells = rows * cols

    for pos in tqdm(range(1, total_cells), desc=f"Solving {rows}x{cols} o={overlap}", leave=False):
        row, col = divmod(pos, cols)
        next_beam: List[Tuple[float, float, Tuple[int, ...], int]] = []

        for cost, max_seam, placements, used_mask in beam:
            left_state = placements[-1] if col > 0 else None
            up_state = placements[(row - 1) * cols + col] if row > 0 else None

            if left_state is not None and up_state is not None:
                left_set = set(horizontal_candidates[left_state])
                up_set = set(vertical_candidates[up_state])
                candidate_states = list(left_set & up_set)
                if len(candidate_states) < max(5, min(len(left_set), len(up_set)) // 8):
                    candidate_states = list(left_set | up_set)
            elif left_state is not None:
                candidate_states = list(horizontal_candidates[left_state])
            elif up_state is not None:
                candidate_states = list(vertical_candidates[up_state])
            else:
                candidate_states = list(range(len(oriented)))

            for state in candidate_states:
                patch_bit = 1 << patch_pos_by_state[state]
                if used_mask & patch_bit:
                    continue
                local_cost = 0.0
                local_max = max_seam
                if left_state is not None:
                    score = h_score(left_state, state)
                    local_cost += score
                    local_max = max(local_max, score)
                if up_state is not None:
                    score = v_score(up_state, state)
                    local_cost += score
                    local_max = max(local_max, score)
                next_beam.append((cost + local_cost, local_max, placements + (state,), used_mask | patch_bit))

        if not next_beam:
            return None
        next_beam.sort(key=lambda item: (item[0], item[1]))
        beam = next_beam[:beam_width]

    seam_count = rows * (cols - 1) + (rows - 1) * cols
    best_cost, best_max, best_placements, _ = min(beam, key=lambda item: (item[0] / max(1, seam_count), item[1]))
    return CandidateResult(
        rows=rows,
        cols=cols,
        overlap=overlap,
        stride=PATCH_SIZE - overlap,
        total_cost=best_cost,
        mean_seam_score=best_cost / max(1, seam_count),
        max_seam_score=best_max,
        placements=best_placements,
    )


def score_suspicious_seams(result: CandidateResult, oriented: Sequence[OrientedPatch]) -> int:
    scores = seam_scores(result, oriented)
    if not scores:
        return 0
    values = np.array([score for _, _, score in scores], dtype=np.float32)
    threshold = float(np.median(values) + 3.0 * np.std(values))
    return int(np.sum(values > threshold))


def seam_scores(result: CandidateResult, oriented: Sequence[OrientedPatch]) -> List[Tuple[str, Tuple[int, int], float]]:
    scores: List[Tuple[str, Tuple[int, int], float]] = []
    for row in range(result.rows):
        for col in range(result.cols):
            idx = row * result.cols + col
            state = result.placements[idx]
            if col + 1 < result.cols:
                right = result.placements[idx + 1]
                scores.append(
                    (
                        "horizontal",
                        (idx, idx + 1),
                        strip_score(
                            edge_strip(oriented[state].image, "right", result.overlap),
                            edge_strip(oriented[right].image, "left", result.overlap),
                        ),
                    )
                )
            if row + 1 < result.rows:
                down = result.placements[idx + result.cols]
                scores.append(
                    (
                        "vertical",
                        (idx, idx + result.cols),
                        strip_score(
                            edge_strip(oriented[state].image, "bottom", result.overlap),
                            edge_strip(oriented[down].image, "top", result.overlap),
                        ),
                    )
                )
    return scores


def render_result(result: CandidateResult, oriented: Sequence[OrientedPatch]) -> np.ndarray:
    height = PATCH_SIZE + (result.rows - 1) * result.stride
    width = PATCH_SIZE + (result.cols - 1) * result.stride
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)

    for idx, state in enumerate(result.placements):
        row, col = divmod(idx, result.cols)
        y = row * result.stride
        x = col * result.stride
        accum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += oriented[state].image.astype(np.float32)
        weights[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1.0

    weights[weights == 0] = 1.0
    return np.clip(accum / weights, 0, 255).astype(np.uint8)


def write_metadata(path: Path, result: CandidateResult, oriented: Sequence[OrientedPatch], num_patches: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    placements = []
    for idx, state in enumerate(result.placements):
        row, col = divmod(idx, result.cols)
        patch = oriented[state]
        placements.append(
            {
                "patch_id": patch.patch_id,
                "row": row,
                "col": col,
                "rotation": patch.rotation,
                "x": col * result.stride,
                "y": row * result.stride,
            }
        )
    payload = {
        "patch_size": PATCH_SIZE,
        "num_patches": num_patches,
        "rows": result.rows,
        "cols": result.cols,
        "overlap": result.overlap,
        "stride": result.stride,
        "mean_seam_score": result.mean_seam_score,
        "max_seam_score": result.max_seam_score,
        "placements": placements,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_seams(path: Path, result: CandidateResult, oriented: Sequence[OrientedPatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "cell_a", "cell_b", "patch_a", "patch_b", "score"])
        for direction, (cell_a, cell_b), score in seam_scores(result, oriented):
            patch_a = oriented[result.placements[cell_a]].patch_id
            patch_b = oriented[result.placements[cell_b]].patch_id
            writer.writerow([direction, cell_a, cell_b, patch_a, patch_b, f"{score:.6f}"])


def validate_result(result: CandidateResult, oriented: Sequence[OrientedPatch], num_patches: int) -> None:
    patch_ids = [oriented[state].patch_id for state in result.placements]
    if len(patch_ids) != num_patches:
        raise RuntimeError("Incomplete placement.")
    if len(set(patch_ids)) != num_patches:
        raise RuntimeError("A patch was used more than once.")
    first = oriented[result.placements[0]]
    if first.patch_id != 0 or first.rotation != 0:
        raise RuntimeError("patch_0 is not fixed at the top-left with rotation 0.")


def main() -> None:
    args = parse_args()
    patch_dir = Path(args.patch_dir)
    patch_ids, patches = load_patches(patch_dir)
    oriented = make_oriented_patches(patch_ids, patches)
    print(f"Loaded {len(patches)} patches and {len(oriented)} oriented states.")

    grid_candidates = factor_grid_candidates(len(patches), args.max_aspect_ratio, args.max_grid_candidates)
    print("Grid candidates:", ", ".join(f"{r}x{c}" for r, c in grid_candidates))

    overlaps = infer_overlaps(oriented, args.overlap_min, args.overlap_max, args.overlap_candidates)
    results: List[CandidateResult] = []
    best_so_far: CandidateResult | None = None

    for overlap in overlaps:
        print(f"Building edge candidates for overlap={overlap}...")
        horizontal, vertical = build_edge_candidates(oriented, overlap, args.top_k, args.descriptor_bins)
        for rows, cols in grid_candidates:
            result = solve_layout(
                oriented=oriented,
                rows=rows,
                cols=cols,
                overlap=overlap,
                beam_width=args.beam_width,
                horizontal_candidates=horizontal,
                vertical_candidates=vertical,
            )
            if result is None:
                print(f"  {rows}x{cols} overlap={overlap}: no complete layout")
                continue
            suspicious = score_suspicious_seams(result, oriented)
            print(
                f"  {rows}x{cols} overlap={overlap}: "
                f"mean={result.mean_seam_score:.4f} max={result.max_seam_score:.4f} suspicious={suspicious}"
            )
            results.append(result)
            if best_so_far is None or (result.mean_seam_score, result.max_seam_score) < (
                best_so_far.mean_seam_score,
                best_so_far.max_seam_score,
            ):
                best_so_far = result
            if result.mean_seam_score <= args.perfect_threshold and result.max_seam_score <= args.perfect_threshold:
                print("Perfect-seam layout found; stopping candidate search early.")
                results = [result]
                break
        if len(results) == 1 and results[0].mean_seam_score <= args.perfect_threshold and results[0].max_seam_score <= args.perfect_threshold:
            break

    if not results:
        raise RuntimeError("No complete layout found.")

    results.sort(key=lambda r: (r.mean_seam_score, score_suspicious_seams(r, oriented), r.max_seam_score))
    best = results[0]
    validate_result(best, oriented, len(patches))
    print(
        f"Selected layout: {best.rows}x{best.cols}, overlap={best.overlap}, "
        f"mean={best.mean_seam_score:.4f}, max={best.max_seam_score:.4f}"
    )

    out_image = Path(args.out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_result(best, oriented)).save(out_image)
    write_metadata(Path(args.out_meta), best, oriented, len(patches))
    write_seams(Path(args.out_seams), best, oriented)
    print(f"Wrote {out_image}")
    print(f"Wrote {args.out_meta}")
    print(f"Wrote {args.out_seams}")


if __name__ == "__main__":
    main()
