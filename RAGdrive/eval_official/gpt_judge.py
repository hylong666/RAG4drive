"""ChatGPT-as-judge scorer, ported from DriveLM/challenge/gpt_eval.py.

Key is read from the ``OPENAI_API_KEY`` environment variable (never hard-coded).
``OPENAI_BASE_URL`` is honoured for local / OpenAI-compatible endpoints
(e.g. vLLM, one-api). Model defaults to gpt-3.5-turbo (as in the official
evaluator) but can be overridden with ``OPENAI_EVAL_MODEL``.
"""

from __future__ import annotations

import os
from openai import OpenAI


class GPTEvaluation:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")  # optional
        self.model = os.environ.get("OPENAI_EVAL_MODEL", "gpt-3.5-turbo")
        if not api_key:
            # Defer the error to call time so --no-chatgpt runs never need a key.
            self.client = None
        elif base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def call_chatgpt(self, chatgpt_messages, max_tokens=40):
        if self.client is None:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it, or run eval with --no-chatgpt."
            )
        response = self.client.chat.completions.create(
            model=self.model, messages=chatgpt_messages,
            temperature=0.6, max_tokens=max_tokens,
        )
        reply = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "an evaluator who rates my answer based on the correct answer"
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "{}".format(prompt)},
        ]

    def forward(self, data):
        answer, GT = data
        prompts = (
            "Rate my answer based on the correct answer out of 100, with higher "
            "scores indicating that the answer is closer to the correct answer, and "
            "you should be accurate to single digits like 62, 78, 41,etc. Output the "
            "number only"
        )
        prompts = prompts + "This is the correct answer: " + GT + "This is my answer: " + answer
        messages = self.prepare_chatgpt_message(prompts)
        reply, _ = self.call_chatgpt(messages, max_tokens=3000)
        output = reply + "\n\n"
        return output[:-2]
