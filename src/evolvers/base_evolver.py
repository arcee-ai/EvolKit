from abc import ABC, abstractmethod

class BaseEvolver(ABC):
    @abstractmethod
    def evolve(self, instruction, evolving_method):
        pass