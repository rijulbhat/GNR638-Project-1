#!/usr/bin/env python3
"""OCR indexing and heuristic MCQ answering for stitched map images."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ocr_vlm.easy_ocr import extract_ocr_easyocr


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "of", "in", "on", "at", "a", "an", "is", "are",
    "shown", "visible", "which", "what", "amongst", "options", "major",
}
TYPE_WORDS = {"lake", "dam", "metro", "river", "talao", "station", "road"}


@dataclass
class OCRWord:
    text: str
    norm_text: str
    conf: float
    left: int
    top: int
    width: int
    height: int
    source_scale: float
    line_key: Tuple[int, int, int]

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)


@dataclass
class OCRPhrase:
    text: str
    norm_text: str
    words: Tuple[OCRWord, ...]
    conf: float
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)


@dataclass
class Match:
    phrase: OCRPhrase
    score: float


def phrase_types(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return sorted(set(tokens) & TYPE_WORDS)


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def normalized_tokens(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_reasonable_ocr_text(text: str) -> bool:
    if not text:
        return False
    if "\t" in text or "\n" in text or "\r" in text:
        return False
    if len(text) > 80:
        return False
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return False
    digits = sum(ch.isdigit() for ch in text)
    if digits / max(1, len(text)) > 0.25:
        return False
    return True


def token_overlap_score(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    inter = len(set(query_tokens) & set(candidate_tokens))
    return inter / max(1, len(query_tokens))


def content_overlap_score(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    query = [tok for tok in query_tokens if tok not in TYPE_WORDS]
    candidate = [tok for tok in candidate_tokens if tok not in TYPE_WORDS]
    if not query:
        return 0.0
    return len(set(query) & set(candidate)) / len(set(query))


def fragment_bonus(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    bonus = 0.0
    for qtok in query_tokens:
        if len(qtok) < 5 or qtok in TYPE_WORDS:
            continue
        for ctok in candidate_tokens:
            if len(ctok) < 3:
                continue
            if ctok in qtok or qtok in ctok:
                ratio = min(len(ctok), len(qtok)) / len(qtok)
                if ratio >= 0.5:
                    bonus = max(bonus, 0.35 * ratio)
    return bonus


def type_bonus(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    query_types = set(query_tokens) & TYPE_WORDS
    cand_types = set(candidate_tokens) & TYPE_WORDS
    if not query_types:
        return 0.0
    return 1.0 if query_types & cand_types else -0.15


def text_match_score(query: str, candidate: str) -> float:
    q_norm = normalize_text(query)
    c_norm = normalize_text(candidate)
    q_tokens = normalized_tokens(query)
    c_tokens = normalized_tokens(candidate)
    content = content_overlap_score(q_tokens, c_tokens)
    score = (
        0.55 * sequence_ratio(q_norm, c_norm)
        + 0.35 * token_overlap_score(q_tokens, c_tokens)
        + 0.10 * type_bonus(q_tokens, c_tokens)
    )
    score += fragment_bonus(q_tokens, c_tokens)
    if [tok for tok in q_tokens if tok not in TYPE_WORDS] and content == 0.0:
        score -= 0.18
    if len(q_norm) >= 8 and len(c_norm) <= 4:
        score -= 0.12
    return score


def dedupe_words(words: Sequence[OCRWord]) -> List[OCRWord]:
    best_by_key: Dict[Tuple[str, int, int], OCRWord] = {}
    for word in words:
        key = (word.norm_text, word.left // 8, word.top // 8)
        prev = best_by_key.get(key)
        if prev is None or word.conf > prev.conf:
            best_by_key[key] = word
    return list(best_by_key.values())


def build_phrases(words: Sequence[OCRWord], max_ngram: int = 6) -> List[OCRPhrase]:
    grouped: Dict[Tuple[int, int, int], List[OCRWord]] = defaultdict(list)
    for word in words:
        grouped[word.line_key].append(word)

    phrases: List[OCRPhrase] = []
    seen = set()
    for line_words in grouped.values():
        line_words = sorted(line_words, key=lambda w: w.left)
        filtered = [w for w in line_words if w.conf >= 20 and w.norm_text]
        for n in range(1, min(max_ngram, len(filtered)) + 1):
            for idx in range(len(filtered) - n + 1):
                span = filtered[idx : idx + n]
                text = " ".join(w.text for w in span)
                norm_text = normalize_text(text)
                if len(norm_text) < 3:
                    continue
                if not is_reasonable_ocr_text(text):
                    continue
                key = (norm_text, span[0].left // 8, span[0].top // 8, span[-1].left // 8)
                if key in seen:
                    continue
                seen.add(key)
                left = min(w.left for w in span)
                top = min(w.top for w in span)
                right = max(w.left + w.width for w in span)
                bottom = max(w.top + w.height for w in span)
                phrases.append(
                    OCRPhrase(
                        text=text,
                        norm_text=norm_text,
                        words=tuple(span),
                        conf=sum(w.conf for w in span) / len(span),
                        left=left,
                        top=top,
                        width=right - left,
                        height=bottom - top,
                    )
                )
    return phrases


def extract_ocr(image_path: Path) -> List[OCRPhrase]:
    return extract_ocr_easyocr(image_path)


def top_matches(query: str, phrases: Sequence[OCRPhrase], limit: int = 10) -> List[Match]:
    scored = [Match(phrase=p, score=text_match_score(query, p.text)) for p in phrases]
    scored.sort(key=lambda item: (item.score, item.phrase.conf), reverse=True)
    return scored[:limit]


def metro_context_score(option: str, phrases: Sequence[OCRPhrase]) -> float:
    best = 0.0
    for phrase in phrases:
        if "metro" not in phrase.norm_text:
            continue
        best = max(best, text_match_score(option, phrase.text))
    return best


_PROXIMITY_RE = re.compile(
    r"\b(?:near|close\s+to|next\s+to|adjacent\s+to|closest\s+to)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)


def find_best_anchor(question: str, phrases: Sequence[OCRPhrase]) -> Match | None:
    match = _PROXIMITY_RE.search(question)
    if not match:
        return None
    anchor = match.group(1).strip()
    best = top_matches(anchor, phrases, limit=1)
    if not best or best[0].score < 0.55:
        return None
    return best[0]


def find_anchor_region(question: str, phrases: Sequence[OCRPhrase]) -> Tuple[Tuple[float, float], List[Match]] | None:
    match = _PROXIMITY_RE.search(question)
    if not match:
        return None
    anchor_text = match.group(1).strip()
    anchor_tokens = normalized_tokens(anchor_text)
    if not anchor_tokens:
        return None

    scored: List[Match] = []
    for phrase in phrases:
        phrase_tokens = normalized_tokens(phrase.text)
        content = content_overlap_score(anchor_tokens, phrase_tokens)
        if content <= 0:
            continue
        score = 0.65 * content + 0.35 * max(0.0, text_match_score(anchor_text, phrase.text))
        scored.append(Match(phrase=phrase, score=score))

    scored.sort(key=lambda item: (item.score, item.phrase.conf), reverse=True)
    scored = scored[:8]
    if not scored or scored[0].score < 0.18:
        return None

    best_cluster = scored
    best_cluster_score = -1.0
    for seed in scored:
        cluster = []
        for item in scored:
            dx = seed.phrase.center[0] - item.phrase.center[0]
            dy = seed.phrase.center[1] - item.phrase.center[1]
            if math.hypot(dx, dy) <= 180:
                cluster.append(item)
        cluster_score = sum(item.score for item in cluster)
        if cluster_score > best_cluster_score:
            best_cluster = cluster
            best_cluster_score = cluster_score

    total_weight = sum(max(item.score, 0.01) for item in best_cluster)
    center_x = sum(item.phrase.center[0] * max(item.score, 0.01) for item in best_cluster) / total_weight
    center_y = sum(item.phrase.center[1] * max(item.score, 0.01) for item in best_cluster) / total_weight
    return (center_x, center_y), best_cluster


def normalized_distance(a: Tuple[float, float], b: Tuple[float, float], width: int, height: int) -> float:
    dx = (a[0] - b[0]) / max(1.0, float(width))
    dy = (a[1] - b[1]) / max(1.0, float(height))
    return math.hypot(dx, dy)


def match_payload(match: Match) -> Dict[str, object]:
    return {
        "text": match.phrase.text,
        "norm_text": match.phrase.norm_text,
        "score": match.score,
        "conf": match.phrase.conf,
        "left": match.phrase.left,
        "top": match.phrase.top,
        "width": match.phrase.width,
        "height": match.phrase.height,
        "center_x": match.phrase.center[0],
        "center_y": match.phrase.center[1],
        "types": phrase_types(match.phrase.text),
    }


def build_question_evidence(
    question: str,
    options: Sequence[str],
    phrases: Sequence[OCRPhrase],
    image_width: int,
    image_height: int,
) -> Dict[str, object]:
    question_lower = question.lower()
    anchor_match = find_best_anchor(question, phrases)
    anchor_region = find_anchor_region(question, phrases)
    option_evidence = []

    for option_idx, option in enumerate(options, start=1):
        matches = top_matches(option, phrases, limit=6)
        if "metro" in question_lower:
            metro_matches = top_matches(f"{option} metro", phrases, limit=3)
            unique: Dict[Tuple[str, int, int], Match] = {}
            for match in matches + metro_matches:
                key = (match.phrase.norm_text, match.phrase.left, match.phrase.top)
                prev = unique.get(key)
                if prev is None or match.score > prev.score:
                    unique[key] = match
            matches = sorted(unique.values(), key=lambda item: (item.score, item.phrase.conf), reverse=True)[:6]

        best_text_score = matches[0].score if matches else 0.0
        spatial = {}
        if matches:
            top_match = matches[0]
            spatial["normalized_top"] = top_match.phrase.center[1] / max(1.0, image_height)
            spatial["normalized_left"] = top_match.phrase.center[0] / max(1.0, image_width)
        if anchor_region is not None and matches:
            anchor_center, _ = anchor_region
            dists = [
                normalized_distance(anchor_center, match.phrase.center, image_width, image_height)
                for match in matches
            ]
            spatial["min_anchor_distance"] = min(dists)

        option_evidence.append(
            {
                "option_index": option_idx,
                "option_text": option,
                "option_types": phrase_types(option),
                "best_text_score": best_text_score,
                "metro_context_score": metro_context_score(option, phrases) if "metro" in question_lower else 0.0,
                "matches": [match_payload(match) for match in matches[:3]],
                "spatial": spatial,
            }
        )

    return {
        "question": question,
        "question_types": phrase_types(question),
        "question_family": (
            "near_anchor"
            if _PROXIMITY_RE.search(question)
            else "spatial_direction"
            if any(word in question_lower for word in ("north", "south", "east", "west"))
            else "visible_lookup"
            if "visible" in question_lower
            else "typed_entity_lookup"
        ),
        "options": option_evidence,
        "anchor": None if anchor_match is None else match_payload(anchor_match),
        "anchor_region": None
        if anchor_region is None
        else {
            "center_x": anchor_region[0][0],
            "center_y": anchor_region[0][1],
            "support": [match_payload(match) for match in anchor_region[1]],
        },
    }


def answer_question(
    question: str,
    options: Sequence[str],
    phrases: Sequence[OCRPhrase],
    image_width: int,
    image_height: int,
    vlm: Optional[object] = None,
    image_path: Optional[Path] = None,
) -> Dict[str, object]:
    question_lower = question.lower()
    evidence = build_question_evidence(question, options, phrases, image_width, image_height)
    anchor_region = evidence["anchor_region"]
    option_scores: List[Dict[str, object]] = []

    for option_evidence in evidence["options"]:
        option_idx = int(option_evidence["option_index"])
        option = str(option_evidence["option_text"])
        matches = [
            Match(
                phrase=OCRPhrase(
                    text=str(m["text"]),
                    norm_text=str(m["norm_text"]),
                    words=tuple(),
                    conf=float(m["conf"]),
                    left=int(m["left"]),
                    top=int(m["top"]),
                    width=int(m["width"]),
                    height=int(m["height"]),
                ),
                score=float(m["score"]),
            )
            for m in option_evidence["matches"]
        ]
        best_text = float(option_evidence["best_text_score"])
        score = best_text
        reason = "text"

        if "north" in question_lower and matches:
            north_bonus = max(0.0, 1.0 - (matches[0].phrase.center[1] / max(1.0, image_height)))
            score += 0.45 * north_bonus
            reason = "north"

        if anchor_region is not None and matches:
            anchor_center = (float(anchor_region["center_x"]), float(anchor_region["center_y"]))
            dists = [normalized_distance(anchor_center, m.phrase.center, image_width, image_height) for m in matches]
            near_radius = 0.40 if "metro" in question_lower else 0.35
            near_weight = 1.00 if "metro" in question_lower else 0.55
            near_bonus = max(0.0, 1.0 - min(dists) / near_radius)
            score += near_weight * near_bonus
            reason = "near"

        if "metro" in question_lower:
            score += 0.45 * float(option_evidence["metro_context_score"])
            reason = "metro-near" if anchor_region is not None else "metro"

        if "dam" in question_lower and matches:
            top_tokens = set(normalized_tokens(matches[0].phrase.text))
            option_content = [tok for tok in normalized_tokens(option) if tok not in TYPE_WORDS]
            if option_content and not (set(option_content) & top_tokens):
                score -= 0.12

        if "visible" in question_lower:
            score += 0.10 * min(1.0, len(matches) / 3.0)

        option_scores.append(
            {
                "option_index": option_idx,
                "option_text": option,
                "score": score,
                "best_text_score": best_text,
                "reason": reason,
                "matches": [
                    {"text": m.phrase.text, "score": m.score, "left": m.phrase.left, "top": m.phrase.top}
                    for m in matches[:3]
                ],
            }
        )

    option_scores.sort(key=lambda item: item["score"], reverse=True)
    best = option_scores[0]
    second = option_scores[1] if len(option_scores) > 1 else None
    margin = best["score"] - (second["score"] if second else 0.0)

    answered = float(best["score"]) > 0.20
    predicted_option = int(best["option_index"]) if answered else 5

    vlm_option: Optional[int] = None
    if vlm is not None and image_path is not None:
        try:
            vlm_option = vlm.answer(  # type: ignore[union-attr]
                image_path=image_path,
                question=question,
                options=list(options),
                option_scores=option_scores,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[VLM] call failed: {exc}")

        if vlm_option is not None and vlm_option in {1, 2, 3, 4}:
            predicted_option = vlm_option
        elif vlm_option == 5:
            predicted_option = 5

    if predicted_option not in {1, 2, 3, 4, 5}:
        predicted_option = 5

    return {
        "predicted_option": predicted_option,
        "confidence": float(best["score"]),
        "margin": float(margin),
        "option_scores": option_scores,
        "anchor": evidence["anchor"],
        "anchor_region": anchor_region,
        "evidence": evidence,
        "abstain_reason": None if answered else "ev_negative",
        "vlm_option": vlm_option,
    }


def load_questions(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_submission(rows: Sequence[Dict[str, str]], answers: Sequence[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question_num", "option"])
        for row, answer in zip(rows, answers):
            qid = row["id"]
            safe_answer = int(answer) if int(answer) in {1, 2, 3, 4, 5} else 5
            writer.writerow([qid, qid, safe_answer])


def save_debug(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
