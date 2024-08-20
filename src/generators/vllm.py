from typing import Optional
from os import getenv

from openai import OpenAI, AsyncOpenAI
from .openai import OpenAIGenerator

class VLLMGenerator(OpenAIGenerator):
    def __init__(self, model: str = "deepseek/deepseek-chat", base_url: str = 'http://localhost:8000/v1') -> None:
        self.model = model
        # self.api_key = api_key if api_key else 'test-abc1'
        self.client = OpenAI(base_url=base_url, api_key='test-abc1')
        
        self.aclient = AsyncOpenAI(base_url=base_url, api_key='test-abc1')
        
    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", temperature: float = 0.5):
        return super().generate(prompt, system_prompt, temperature)
    
    async def agenerate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", temperature: float = 0.5):
        response = await self.aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,)
        # print(response.choices[0].message.content) # For Debuging
        return response.choices[0].message.content