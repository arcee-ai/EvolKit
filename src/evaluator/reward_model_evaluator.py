from .base_evaluator import BaseEvaluator
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer


class RewardModelEvaluator(BaseEvaluator):
    def __init__(self, model_name: str = "internlm/internlm2-1_8b-reward"):
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def get_score(self, instruction: str, response: str) -> float:
        chat = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        return self.model.get_score(self.tokenizer, chat)

    def evaluate(self, instructions: List[str], responses: List[str]) -> float:
        scores = [self.get_score(instruction, response) for instruction, response in zip(instructions, responses)]
        return sum(scores) / len(scores)

    def select_best_method(self, methods: List[str], instructions: List[str], responses: List[List[str]]) -> tuple:
        best_method = None
        highest_score = float('-inf')

        for method, method_responses in zip(methods, responses):
            avg_score = self.evaluate(instructions, method_responses)
            if avg_score > highest_score:
                highest_score = avg_score
                best_method = method

        return best_method, highest_score