from abc import ABC, abstractmethod
from typing import List, Optional

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, instructions: List[str], responses: List[str]) -> float:
        pass

    @abstractmethod
    def select_best_method(self, methods: List[str], instructions: List[str], responses: List[List[str]]) -> tuple:
        pass