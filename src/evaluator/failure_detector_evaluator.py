from .base_evaluator import BaseEvaluator
from typing import List

class FailureDetectorEvaluator(BaseEvaluator):
    def __init__(self):
        self.stagnant_keywords = ["understood", "thank you", "noted", "got it", "okay", "alright"]
        self.insufficient_keywords = ["sure", "certainly", "of course", "happy to help"]
        self.loss_keywords = ["please provide", "need more information", "could you clarify", "what exactly"]

    def is_failure(self, response: str) -> bool:
        response = response.lower()
        return (
            self._is_stagnant_complexity(response) or
            self._is_insufficient_qualification(response) or
            self._is_loss_of_information(response)
        )

    def _is_stagnant_complexity(self, response: str) -> bool:
        return any(keyword in response for keyword in self.stagnant_keywords) and response.endswith("?")

    def _is_insufficient_qualification(self, response: str) -> bool:
        return (
            (any(keyword in response for keyword in self.insufficient_keywords) and response.endswith("?")) or
            ("what do you mean" in response or "could you explain" in response)
        )

    def _is_loss_of_information(self, response: str) -> bool:
        return any(keyword in response for keyword in self.loss_keywords)

    def evaluate(self, instructions: List[str], responses: List[str]) -> float:
        failures = sum(self.is_failure(response) for response in responses)
        return failures / len(responses)

    def select_best_method(self, methods: List[str], instructions: List[str], responses: List[List[str]]) -> tuple:
        best_method = None
        lowest_failure_rate = float('inf')

        for method, method_responses in zip(methods, responses):
            failure_rate = self.evaluate(instructions, method_responses)
            if failure_rate < lowest_failure_rate:
                lowest_failure_rate = failure_rate
                best_method = method

        return best_method, lowest_failure_rate