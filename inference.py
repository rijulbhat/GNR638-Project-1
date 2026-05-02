#!/usr/bin/env python3
"""End-to-end inference: stitch map patches → OCR → answer MCQs → write submission.csv.

Usage
-----
    python inference.py --test_dir <path_to_test_dir>

Expected directory structure
-----------------------------
    <test_dir>/
    ├── patches/        # patch_0.png … patch_N.png
    ├── test.csv        # MCQ questions
    └── sample_submission.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from map_qa import answer_question, extract_ocr, load_questions, save_debug, write_submission


PYTHON = sys.executable
_DEFAULT_VLM = str(Path(__file__).parent / "Qwen3-VL-8B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct map patches and answer map MCQs.")
    parser.add_argument(
        "--test_dir",
        required=True,
        help="Test directory containing patches/, test.csv, and sample_submission.csv.",
    )
    parser.add_argument("--out-dir", default="part2_outputs", help="Directory for outputs and debug files.")
    parser.add_argument("--submission", default="submission.csv", help="Output submission CSV path.")
    parser.add_argument("--skip-stitch", action="store_true", help="Skip stitching and use --image directly.")
    parser.add_argument("--image", default=None, help="Existing reconstructed image (requires --skip-stitch).")
    parser.add_argument("--known-overlap", type=int, default=32, help="Patch overlap in pixels; <=0 to infer.")
    parser.add_argument("--force-answer", action="store_true", help="Always pick top option; never abstain.")
    parser.add_argument(
        "--vlm-model",
        default=_DEFAULT_VLM,
        metavar="MODEL",
        help=f"Local path or HuggingFace repo for Qwen3-VL (default: {_DEFAULT_VLM}).",
    )
    return parser.parse_args()


def count_patches(patch_dir: Path) -> int:
    return len(list(patch_dir.glob("patch_*.png")))


def hyperparams_for_count(num_patches: int) -> Dict[str, int]:
    if num_patches <= 80:
        return {"beam_width": 100, "top_k": 30, "max_grid_candidates": 1}
    if num_patches <= 144:
        return {"beam_width": 300, "top_k": 60, "max_grid_candidates": 1}
    if num_patches <= 256:
        return {"beam_width": 1500, "top_k": 200, "max_grid_candidates": 1}
    return {"beam_width": 2500, "top_k": 250, "max_grid_candidates": 1}


def run_stitch(patch_dir: Path, out_dir: Path, known_overlap: int) -> Path:
    num_patches = count_patches(patch_dir)
    if num_patches == 0:
        raise FileNotFoundError(f"No patch_*.png files found in {patch_dir}")

    params = hyperparams_for_count(num_patches)
    out_image = out_dir / "reconstructed.png"
    cmd = [
        PYTHON, "stitch.py",
        "--patch-dir", str(patch_dir),
        "--out-image", str(out_image),
        "--out-meta", str(out_dir / "placements.json"),
        "--out-seams", str(out_dir / "seam_scores.csv"),
        "--beam-width", str(params["beam_width"]),
        "--top-k", str(params["top_k"]),
        "--max-grid-candidates", str(params["max_grid_candidates"]),
    ]
    if known_overlap > 0:
        cmd += ["--overlap-min", str(known_overlap), "--overlap-max", str(known_overlap), "--overlap-candidates", "1"]

    print(f"Running stitcher (patches={num_patches}, beam={params['beam_width']}, top-k={params['top_k']})...")
    subprocess.run(cmd, check=True)
    return out_image


def serialize_phrase(phrase) -> Dict[str, object]:
    return {
        "text": phrase.text,
        "norm_text": phrase.norm_text,
        "conf": phrase.conf,
        "left": phrase.left,
        "top": phrase.top,
        "width": phrase.width,
        "height": phrase.height,
        "center_x": phrase.center[0],
        "center_y": phrase.center[1],
    }


def answer_questions(
    image_path: Path,
    test_csv: Path,
    out_dir: Path,
    submission_path: Path,
    force_answer: bool,
    vlm: Optional[object] = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    questions = load_questions(test_csv)

    print(f"Running OCR on {image_path}...")
    phrases = extract_ocr(image_path)
    print(f"Indexed {len(phrases)} OCR phrases.")

    predictions: List[int] = []
    debug_rows = []
    for row in questions:
        options = [row[f"option_{idx}"] for idx in range(1, 5)]
        result = answer_question(
            row["question"], options, phrases,
            image.width, image.height,
            vlm=vlm, image_path=image_path,
        )
        raw = result["predicted_option"]
        prediction = int(raw) if int(raw) in {1, 2, 3, 4, 5} else 5
        if force_answer and prediction == 5:
            prediction = int(result["option_scores"][0]["option_index"])
        predictions.append(prediction)

        vlm_tag = f", vlm={result.get('vlm_option')}" if vlm is not None else ""
        print(
            f"{row['id']}: predicted {prediction} "
            f"(confidence={result['confidence']:.3f}, margin={result['margin']:.3f}{vlm_tag})"
        )
        debug_rows.append({
            "id": row["id"],
            "question": row["question"],
            "options": options,
            "prediction": prediction,
            "vlm_option": result.get("vlm_option"),
            "confidence": result["confidence"],
            "margin": result["margin"],
            "abstain_reason": result["abstain_reason"],
            "option_scores": result["option_scores"],
            "evidence": result["evidence"],
        })

    write_submission(questions, predictions, submission_path)
    save_debug(out_dir / "qa_debug.json", {
        "image": str(image_path),
        "test_csv": str(test_csv),
        "num_phrases": len(phrases),
        "vlm_model": str(vlm) if vlm is not None else None,
        "ocr_phrases_preview": [
            serialize_phrase(p)
            for p in sorted(phrases, key=lambda p: (p.conf, len(p.text)), reverse=True)[:500]
        ],
        "answers": debug_rows,
    })
    print(f"Wrote {submission_path}")
    print(f"Wrote {out_dir / 'qa_debug.json'}")


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir).resolve()
    patch_dir = test_dir / "patches"
    test_csv = test_dir / "test.csv"

    if not patch_dir.exists():
        raise FileNotFoundError(f"patches/ not found in {test_dir}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found in {test_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_stitch:
        if not args.image:
            raise ValueError("--image is required with --skip-stitch")
        image_path = Path(args.image)
    else:
        image_path = run_stitch(patch_dir, out_dir, args.known_overlap)

    vlm: Optional[object] = None
    if args.vlm_model:
        try:
            from ocr_vlm.vlm_answerer import VLMAnswerer
            vlm = VLMAnswerer(args.vlm_model)
        except Exception as exc:  # noqa: BLE001
            print(f"[VLM] Failed to load '{args.vlm_model}': {exc}")
            print("[VLM] Falling back to heuristic-only answering.")

    answer_questions(
        image_path=image_path,
        test_csv=test_csv,
        out_dir=out_dir,
        submission_path=Path(args.submission),
        force_answer=args.force_answer,
        vlm=vlm,
    )


if __name__ == "__main__":
    main()
