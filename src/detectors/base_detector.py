from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, original_instruction, evolved_instruction, response):
        pass