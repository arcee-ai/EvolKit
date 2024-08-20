from .base_evaluator import BaseEvaluator
from typing import List, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class FailureDetectorEvaluator(BaseEvaluator):
    def __init__(self, max_workers: int = 4):
        self.stagnant_pattern = re.compile(r'\b(understood|thank you|noted|got it|okay|alright)\b.*\?$', re.IGNORECASE)
        self.insufficient_pattern = re.compile(r'\b(sure|certainly|of course|happy to help)\b.*\?$|what do you mean|could you explain', re.IGNORECASE)
        self.loss_pattern = re.compile(r'please provide|need more information|could you clarify|what exactly', re.IGNORECASE)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def is_failure(self, response: str) -> bool:
        return (
            bool(self.stagnant_pattern.search(response)) or
            bool(self.insufficient_pattern.search(response)) or
            bool(self.loss_pattern.search(response))
        )

    def evaluate(self, instructions: List[str], responses: List[str]) -> float:
        with self.executor as executor:
            failure_futures = [executor.submit(self.is_failure, response) for response in responses]
            failures = sum(future.result() for future in as_completed(failure_futures))
        return failures / len(responses)

    async def select_best_method(self, methods: List[str], instructions: List[str], responses: List[List[str]]) -> Tuple[str, float]:
        evaluation_results = []
        
        with self.executor as executor:
            futures = [executor.submit(self.evaluate, instructions, method_responses) 
                       for method_responses in responses]
            
            for method, future in zip(methods, futures):
                failure_rate = future.result()
                evaluation_results.append((method, failure_rate))
        
        best_method, lowest_failure_rate = min(evaluation_results, key=lambda x: x[1])
        return best_method, lowest_failure_rate