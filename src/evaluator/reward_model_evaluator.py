from .base_evaluator import BaseEvaluator
from typing import List, Optional
import torch
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import asyncio


class RewardModelEvaluator(BaseEvaluator):
    def __init__(self, model: str = "internlm/internlm2-1_8b-reward"):
        self.model = AutoModel.from_pretrained(
            model,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your GPU

    async def get_score(self, instruction: str, response: str) -> float:
        chat = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.model.get_score, self.tokenizer, chat)

    async def evaluate(self, instructions: List[str], responses: List[str]) -> float:
        scores = await asyncio.gather(*[self.get_score(instruction, response) 
                                        for instruction, response in zip(instructions, responses)])
        return sum(scores) / len(scores)

    async def select_best_method(self, methods: List[str], instructions: List[str], responses: List[List[str]]) -> tuple:
        evaluation_tasks = [self.evaluate(instructions, method_responses) 
                            for method_responses in responses]
        scores = await asyncio.gather(*evaluation_tasks)
        
        best_index = max(range(len(scores)), key=scores.__getitem__)
        return methods[best_index], scores[best_index]