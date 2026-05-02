#!/usr/bin/env python3
"""EasyOCR-based OCR module for stitched OSM map images."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Weights are pre-downloaded by setup.bash into <submission_root>/easyocr_models/
_MODEL_DIR = str(Path(__file__).resolve().parent.parent / "easyocr_models")


def _assign_line_keys(centers_y: List[float], tolerance: int = 15) -> List[Tuple[int, int, int]]:
    if not centers_y:
        return []
    sorted_order = sorted(range(len(centers_y)), key=lambda i: centers_y[i])
    line_keys: List[Tuple[int, int, int]] = [(0, 0, 0)] * len(centers_y)
    current_line = 0
    prev_y = centers_y[sorted_order[0]]
    line_keys[sorted_order[0]] = (0, 0, 0)
    for pos in sorted_order[1:]:
        if centers_y[pos] - prev_y > tolerance:
            current_line += 1
        line_keys[pos] = (0, 0, current_line)
        prev_y = centers_y[pos]
    return line_keys


def extract_ocr_easyocr(image_path: Path) -> List:
    """Run EasyOCR on *image_path* and return List[OCRPhrase]."""
    import easyocr  # type: ignore[import]
    from PIL import Image

    _parent = str(Path(__file__).resolve().parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from map_qa import (
        OCRWord,
        build_phrases,
        dedupe_words,
        is_reasonable_ocr_text,
        normalize_text,
    )

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False,
        model_storage_directory=_MODEL_DIR,
        download_enabled=False,
    )

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    all_words: List[OCRWord] = []

    for scale in (1.0, 1.5):
        if abs(scale - 1.0) < 1e-6:
            scaled = img
        else:
            from PIL import Image as _PIL
            pil = _PIL.fromarray(img)
            pil = pil.resize((int(w * scale), int(h * scale)), _PIL.Resampling.LANCZOS)
            scaled = np.array(pil)

        detections = reader.readtext(scaled, detail=1, paragraph=False)

        centers_y: List[float] = []
        valid = []
        for bbox, text, conf in detections:
            text = (text or "").strip()
            if not text or not is_reasonable_ocr_text(text):
                continue
            if conf < 0.10:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, x2 = int(min(xs) / scale), int(max(xs) / scale)
            y1, y2 = int(min(ys) / scale), int(max(ys) / scale)
            cy = (y1 + y2) / 2.0
            valid.append((text, conf, x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
            centers_y.append(cy)

        line_keys = _assign_line_keys(centers_y)

        for det_idx, (text, conf, left, top, width, height) in enumerate(valid):
            tokens = text.split()
            if not tokens:
                continue
            tok_w = width // max(1, len(tokens))
            for tok_idx, token in enumerate(tokens):
                all_words.append(
                    OCRWord(
                        text=token,
                        norm_text=normalize_text(token),
                        conf=conf * 100.0,
                        left=left + tok_idx * tok_w,
                        top=top,
                        width=max(1, tok_w),
                        height=height,
                        source_scale=scale,
                        line_key=line_keys[det_idx],
                    )
                )

    return build_phrases(dedupe_words(all_words))
