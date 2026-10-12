"""r7 R7-30 D1-30 evaluators (pure Python).
- ChartQA relaxed accuracy, pinned (frozen protocol OPEN-7): google-research/pix2struct pix2struct/metrics.py::relaxed_correctness at commit
  6fe25c1dc8151823ee3b479519d8d5948812fee4 (identical logic in EvolvingLMMs-Lab/lmms-eval 19085f3 lmms_eval/tasks/chartqa/utils.py, which
  cites Qwen-VL 34b4c0e).  Numeric answers (incl. 'x%') within 5 % relative change of a non-zero numeric target; otherwise case-insensitive
  exact match.  The only pre-processing here is .strip() of the generated text.
- GQA exact match and TextVQA m4c soft accuracy are reused from scripts/r7_vl_common.py (LLaVA's convert_gqa_for_eval / m4c_evaluator).
"""
from __future__ import annotations

from typing import Optional

PINNED = {"source": "google-research/pix2struct pix2struct/metrics.py::relaxed_correctness", "commit": "6fe25c1dc8151823ee3b479519d8d5948812fee4",
          "copy_verified_against": "EvolvingLMMs-Lab/lmms-eval 19085f3ac07e4631fd2f0218786741b6bbc40f69 lmms_eval/tasks/chartqa/utils.py",
          "prompt": "question + '\\nAnswer the question with a single word.' (lmms-eval 19085f3 chartqa.yaml default)", "max_relative_change": 0.05}


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """Verbatim logic of the pinned pix2struct function (see module docstring)."""
    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    return prediction.lower() == target.lower()


def chartqa_score(prediction: str, label: str) -> float:
    return float(relaxed_correctness(label, prediction.strip()))
