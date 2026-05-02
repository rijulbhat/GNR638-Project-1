#!/usr/bin/env python3
"""Qwen2-VL-2B-Instruct based final decider for map MCQ answering.

Design
------
The ``VLMAnswerer`` class wraps ``Qwen/Qwen2-VL-2B-Instruct`` (or any local
checkpoint of the same architecture).  It receives the full-resolution
reconstructed map image together with the question, the four answer options,
and an optional OCR evidence summary produced by the heuristic pipeline in
``map_qa.py``.  It returns a single digit in {1, 2, 3, 4} or 5 to indicate
abstention.

A strict ``re`` guard on the output means any hallucination or non-digit
response automatically falls back to 5, preventing the −1 penalty.

Usage
-----
    from ocr_vlm.vlm_answerer import VLMAnswerer

    vlm = VLMAnswerer()                      # loads model once
    answer = vlm.answer(
        image_path=Path("part2_outputs/reconstructed.png"),
        question="Which major water body is visible in the map?",
        options=["Powai Lake", "Vihar Lake", "Tulsi Lake", "Arabian Sea"],
        option_scores=[...],                 # from map_qa answer_question()
    )
    # answer ∈ {1, 2, 3, 4, 5}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches the *first* standalone digit 1–4 anywhere in the model output.
# Anything that doesn't contain 1–4 (e.g. "I don't know", "5", "Option A")
# returns 5.
_ANSWER_RE = re.compile(r"\b([1-4])\b")

_PROMPT_TEMPLATE = """\
You are an expert at reading and analysing map images. \
Answer the multiple-choice question below about the map shown.

Question: {question}

Options:
1. {opt1}
2. {opt2}
3. {opt3}
4. {opt4}

OCR evidence extracted from the map:
{evidence}

Instructions:
- Study the map image carefully, paying attention to labels, icons, and spatial layout.
- Use both the visual content and the OCR evidence to decide which option is correct.
- Reply with ONLY a single digit: 1, 2, 3, or 4.
- If you genuinely cannot determine the answer, reply with 5 to abstain.
Do NOT include any explanation — just the digit.\
"""


# ---------------------------------------------------------------------------
# Evidence formatter
# ---------------------------------------------------------------------------

def _format_evidence(option_scores: List[Dict]) -> str:
    """Convert the ``option_scores`` list from ``answer_question()`` into a
    compact plain-text summary suitable for the VLM prompt."""
    lines: List[str] = []
    for item in option_scores:
        opt_text = item.get("option_text", "?")
        matches = item.get("matches", [])
        score = item.get("score", 0.0)
        if matches:
            top = matches[0]
            lines.append(
                f'  Option "{opt_text}": best OCR match = "{top["text"]}" '
                f'(text_score={top["score"]:.2f}, composite_score={score:.2f})'
            )
        else:
            lines.append(
                f'  Option "{opt_text}": no OCR match found (composite_score={score:.2f})'
            )
    return "\n".join(lines) if lines else "  No OCR evidence available."


# ---------------------------------------------------------------------------
# VLMAnswerer
# ---------------------------------------------------------------------------

class VLMAnswerer:
    """Wraps Qwen2-VL-2B-Instruct for GPU-accelerated MCQ answering.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace repo ID or absolute local path.  Default is the 2 B
        variant which fits comfortably in the 48 GB L40s VRAM alongside the
        rest of the pipeline.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    ) -> None:
        print(f"[VLM] Loading model from '{model_name_or_path}' …")
        from transformers import AutoProcessor  # type: ignore

        # Qwen3-VL uses Qwen3VLForConditionalGeneration; Qwen2-VL (and
        # Qwen2.5-VL) use Qwen2VLForConditionalGeneration.  Pick the right
        # class based on the model name so both generations work transparently.
        name_lower = model_name_or_path.lower()
        if "qwen3" in name_lower:
            from transformers import Qwen3VLForConditionalGeneration as _ModelCls  # type: ignore
        else:
            from transformers import Qwen2VLForConditionalGeneration as _ModelCls  # type: ignore

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = _ModelCls.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        print("[VLM] Model ready.")

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def answer(
        self,
        image_path: Path,
        question: str,
        options: List[str],
        option_scores: Optional[List[Dict]] = None,
    ) -> int:
        """Return a predicted option index in {1, 2, 3, 4}, or 5 to abstain.

        Parameters
        ----------
        image_path:
            Path to the full-resolution reconstructed map image.
        question:
            The MCQ question string.
        options:
            Exactly four answer option strings (index 0 → option 1, etc.).
        option_scores:
            The ``option_scores`` list from ``map_qa.answer_question()``.
            Used to build the OCR evidence summary injected into the prompt.
            Pass ``None`` if unavailable.

        Returns
        -------
        int
            1–4 for a definite answer, 5 to abstain / on any parse failure.
        """
        # Pad to exactly 4 options so the template never has blank slots.
        opts = list(options)[:4]
        while len(opts) < 4:
            opts.append("(none)")

        evidence = _format_evidence(option_scores) if option_scores else "  No OCR evidence available."

        prompt_text = _PROMPT_TEMPLATE.format(
            question=question,
            opt1=opts[0],
            opt2=opts[1],
            opt3=opts[2],
            opt4=opts[3],
            evidence=evidence,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            return self._run_inference(messages)
        except Exception as exc:  # noqa: BLE001
            print(f"[VLM] Inference error: {exc}")
            return 5

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_inference(self, messages: List[Dict]) -> int:
        """Run the model and parse a digit from its output."""
        # Try the official qwen_vl_utils helper first; fall back to direct
        # PIL loading if the package is not installed.
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            from PIL import Image  # type: ignore

            # Extract image path from the message content
            img_path = next(
                c["image"]
                for c in messages[0]["content"]
                if c["type"] == "image"
            )
            img = Image.open(img_path).convert("RGB")
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input],
                images=[img],
                padding=True,
                return_tensors="pt",
            )

        # Move tensors to the same device as the model.
        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,   # We only need a single digit.
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Trim the prompt tokens from the output.
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print(f"[VLM] Raw output: {output_text!r}")

        m = _ANSWER_RE.search(output_text)
        if m:
            return int(m.group(1))
        # Strict fallback — treat anything not in {1,2,3,4} as abstain.
        return 5
