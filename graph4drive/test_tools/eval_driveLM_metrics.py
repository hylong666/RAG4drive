import re
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import torch

class DriveLMEvaluator:
    def __init__(self, judge_tokenizer=None, judge_model=None, device="cuda"):
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self._cider_scorer = Cider()
        self.judge_tok = judge_tokenizer
        self.judge_mod = judge_model
        self.smooth = SmoothingFunction().method1

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def bleu4(self, preds, refs):
        # refs: list[list[str]] (DriveLM 格式)
        pred_tok = [self._tokenize(p) for p in preds]
        ref_tok = [[self._tokenize(r) for r in rl] for rl in refs]
        return corpus_bleu(ref_tok, pred_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=self.smooth) * 100

    def meteor(self, preds, refs):
        scores = []
        for p, rl in zip(preds, refs):
            p_tok = self._tokenize(p)
            scores.append(max(meteor_score([self._tokenize(r)], p_tok) for r in rl))
        return np.mean(scores) * 100

    def cider(self, preds, refs):
        gts = {i: [r.replace('\n',' ') for r in rl] for i, rl in enumerate(refs)}
        res = {i: [p.replace('\n',' ')] for i, p in enumerate(preds)}
        score, _ = self._cider_scorer.compute_score(gts, res)
        return score * 100

    def rougel(self, preds, refs):
        scores = []
        for p, rl in zip(preds, refs):
            scores.append(max(self.scorer.score(r, p)['rougeL'].fmeasure for r in rl))
        return np.mean(scores) * 100

    def accuracy_local_judge(self, preds, refs, threshold=60):
        if not self.judge_mod or not self.judge_tok:
            # 降级：用 BLEU-4 > 15 近似判定
            return sum(1 for p, rl in zip(preds, refs) 
                      if self.bleu4([p], [rl]) > 15.0) / len(preds) * 100
        
        scores = []
        prompt_template = (
            "You are an autonomous driving safety expert. Rate the following answer against the ground truth "
            "on a scale of 0-100 based on factual correctness, safety compliance, and logical coherence. "
            "Output ONLY the integer score.\n"
            "Ground Truth: {ref}\nModel Answer: {pred}\nScore:"
        )
        with torch.no_grad():
            for p, rl in zip(preds, refs):
                ref = rl[0]  # 单参考
                txt = prompt_template.format(ref=ref, pred=p)
                inputs = self.judge_tok(txt, return_tensors="pt").to(self.device)
                out = self.judge_mod.generate(**inputs, max_new_tokens=5, do_sample=False)
                decoded = self.judge_tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                match = re.search(r'\b(100|[1-9]?\d)\b', decoded)
                scores.append(int(match.group(1)) if match else 50)
        
        return sum(1 for s in scores if s >= threshold) / len(preds) * 100

    def evaluate(self, preds, refs):
        acc = self.accuracy_local_judge(preds, refs)
        return {
            "BLEU-4": float(round(self.bleu4(preds, refs), 2)),
            "METEOR": float(round(self.meteor(preds, refs), 2)),
            "CIDEr": float(round(self.cider(preds, refs), 2)),
            "ROUGE-L": float(round(self.rougel(preds, refs), 2)),
            "Accuracy": float(round(acc, 2))
        }