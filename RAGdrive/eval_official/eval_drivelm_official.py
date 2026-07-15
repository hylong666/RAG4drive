"""DriveLM official-standard evaluation for RAGdrive predictions.

This is a faithful port of ``DriveLM/challenge/evaluation.py`` with two
environment-driven changes so it runs inside RAGdrive's own conda env
(``drivegraphLM``) without the hard-to-install ``language_evaluation`` package:

* **Language metrics** (BLEU-1..4, ROUGE-L, CIDEr) use ``pycocoevalcap``
  (already in RAGdrive/requirements.txt) instead of ``language_evaluation``.
  The scoring/aggregation math and the final-score weights are unchanged.
* **ChatGPT judge** reads the key from the ``OPENAI_API_KEY`` env var and
  honours ``OPENAI_BASE_URL`` for local/compatible endpoints.

Scoring (identical to official):
    final = 0.4*chatgpt + 0.2*language + 0.2*match + 0.2*accuracy

tags: 0=accuracy(MCQ/exact), 1=chatgpt, 2=language(BLEU/ROUGE/CIDEr),
      3=match(F1 on object coords) + chatgpt.

Usage
-----
::

    export OPENAI_API_KEY=sk-...
    python eval_drivelm_official.py \\
        --pred  preds_official.json \\
        --test  my_test_eval.json
"""

from __future__ import annotations

import re
import os
import argparse
import json
from multiprocessing import Pool

import numpy as np

from pycoco_language_eval import CocoLanguageEvaluator
from gpt_judge import GPTEvaluation


class evaluation_suit:
    def __init__(self):
        self.language_eval = CocoLanguageEvaluator()
        self.chatgpt_eval = GPTEvaluation()
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

    def eval_acc(self):
        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            scores.append(1.0 if answer == GT else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def eval_chatGPT(self, data):
        if not data:
            return 0.0
        with Pool(8) as p:
            scores = p.map(self.chatgpt_eval.forward, data)
        scores = list(map(float, scores))
        return sum(scores) / len(scores)

    def eval_language(self):
        answer = self.language["answer"]
        GT = self.language["GT"]
        if not answer:
            return {}
        results_gen = self.language_eval.run_evaluation(answer, GT)
        return {f"val/{k}": v for k, v in results_gen.items()}

    def eval_match(self):
        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)
        outs1 = sum(outs1) / len(outs1) if outs1 else 0.0
        outs2 = self.eval_chatGPT(self.match["GPT"])
        return (outs1 + outs2) / 2.0

    def eval_graph(self, question):
        question_nums = re.findall(r"\d+\.\d+", question)
        question_nums = np.array(
            [list(map(float, x.split()))[0] for x in question_nums]
        ).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        answer_nums = re.findall(r"\d+\.\d+", answer)
        GT_nums = re.findall(r"\d+\.\d+", GT)
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        answer_nums = np.array(
            [list(map(float, x.split()))[0] for x in answer_nums]
        ).reshape(-1, 2)
        GT_nums = np.array(
            [list(map(float, x.split()))[0] for x in GT_nums]
        ).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        for pred in answer_nums:
            closest_distance = float("inf")
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i
            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)
                GT_nums = np.delete(GT_nums, closest_id, axis=0)
            else:
                false_positives += 1

        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
        if 1 in tag:
            self.GPT.append((answer, GT))
        if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
        if 3 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
            self.match["GPT"].append((answer, GT))

    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        scores["language"] = self.eval_language()
        scores["match"] = self.eval_match()
        return scores


def main():
    parser = argparse.ArgumentParser(description="DriveLM official-standard evaluation")
    parser.add_argument("--pred", required=True, help="preds_official.json (list of {id, answer})")
    parser.add_argument("--test", required=True, help="my_test_eval.json (GT + tags)")
    parser.add_argument("--no-chatgpt", action="store_true",
                        help="Skip ChatGPT judge (final score omits the 0.4 chatgpt term).")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        pred_file = json.load(f)
    pred_file = {pred_file[i]["id"]: pred_file[i] for i in range(len(pred_file))}

    with open(args.test, "r", encoding="utf-8") as f:
        test_file = json.load(f)

    evaluation = evaluation_suit()
    n_scored = 0
    for scene_id in test_file.keys():
        scene_data = test_file[scene_id]["key_frames"]
        for frame_id in scene_data.keys():
            frame_data_qa = scene_data[frame_id]["QA"]
            first_flag = True
            ordered = (frame_data_qa["perception"] + frame_data_qa["prediction"]
                       + frame_data_qa["planning"] + frame_data_qa["behavior"])
            for i, qa in enumerate(ordered):
                question = qa["Q"]
                GT = qa["A"]
                tag = qa["tag"]
                idx = scene_id + "_" + frame_id + "_" + str(i)
                if idx not in pred_file:
                    continue
                predict = pred_file[idx]["answer"]
                if first_flag:
                    first_flag = False
                    evaluation.set_graph(predict, GT)
                    evaluation.forward(tag, predict, GT)
                    n_scored += 1
                else:
                    if evaluation.eval_graph(question):
                        evaluation.forward(tag, predict, GT)
                        n_scored += 1

    print(f"[graph-gated] scored {n_scored} QA (rest pruned by graph gate)")
    output = evaluation.evaluation()
    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("match score: ", output["match"])
    print("language score: ", output["language"])

    # Normalize to 0-1 and combine: chatgpt, language, match, accuracy
    scores = []
    weights = [0.4, 0.2, 0.2, 0.2]

    # chatGPT
    scores.append(output["chatgpt"] / 100.0)

    # language
    score = 0
    lang = output["language"]
    for idx, key in enumerate(lang.keys()):
        if idx < 4:
            score += lang[key] / 4.0 / 3.0
        elif idx == 4:
            score += lang[key] / 3.0
        else:
            score += lang[key] / 10.0 / 3.0
    scores.append(score)

    # match
    scores.append(output["match"] / 100.0)

    # accuracy
    scores.append(output["accuracy"])

    final_score = sum([x * y for x, y in zip(scores, weights)])
    print("final score: ", final_score)


if __name__ == "__main__":
    main()

