from abc import ABC, abstractmethod
from typing import Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = "You are a helpful AI assistant.", temperature: Optional[float] = 0.5) -> str:
        pass
    
    async def agenerate(self, prompt: str, system_prompt: Optional[str] = "You are a helpful AI assistant.", temperature: Optional[float] = 0.5):
        pass