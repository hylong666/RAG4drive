"""Drop-in replacement for ``language_evaluation.CocoEvaluator`` using
``pycocoevalcap`` (already required by RAGdrive).

The official DriveLM evaluator calls::

    CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
    results = evaluator.run_evaluation(answers, ground_truths)

and then consumes ``results`` as an ordered dict whose first 4 keys are
BLEU_1..BLEU_4, the 5th is ROUGE_L, and the rest (CIDEr) are handled by the
``idx > 4`` branch. We reproduce exactly that key ordering so the downstream
final-score aggregation in eval_drivelm_official.py stays identical.

Output keys (in order): BLEU_1, BLEU_2, BLEU_3, BLEU_4, ROUGE_L, CIDEr
Values are on a 0-100 scale for BLEU/ROUGE and 0-10-ish for CIDEr, matching
the scaling assumptions in the official aggregation (divide-by-4/3 for BLEU,
divide-by-3 for ROUGE, divide-by-10/3 for CIDEr).
"""

from __future__ import annotations

from typing import Dict, List

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class CocoLanguageEvaluator:
    def run_evaluation(self, answers: List[str], ground_truths: List[str]) -> Dict[str, float]:
        # pycocoevalcap expects {id: [caption]} dicts.
        gts = {i: [gt] for i, gt in enumerate(ground_truths)}
        res = {i: [ans] for i, ans in enumerate(answers)}

        out: Dict[str, float] = {}

        # BLEU-1..4 -> 0-100 scale (official language_evaluation reports *100).
        bleu_scores, _ = Bleu(4).compute_score(gts, res)
        for n in range(4):
            out[f"BLEU_{n + 1}"] = bleu_scores[n] * 100.0

        # ROUGE-L -> 0-100 scale.
        rouge_score, _ = Rouge().compute_score(gts, res)
        out["ROUGE_L"] = rouge_score * 100.0

        # CIDEr -> native scale (~0-10). The aggregator divides it by 10/3.
        cider_score, _ = Cider().compute_score(gts, res)
        out["CIDEr"] = cider_score * 10.0

        return out
