from typing import Optional
from os import getenv

from openai import OpenAI

from .base_generator import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key if api_key else getenv("OPENAI_API_KEY")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        self.client = OpenAI()

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", temperature: float = 0.7):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,)
        # print(response.choices[0].message.content) # For Debuging
        return response.choices[0].message.content