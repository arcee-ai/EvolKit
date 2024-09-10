from typing import Optional
from os import getenv

from openai import OpenAI, AsyncOpenAI
from .openai import OpenAIGenerator

class OpenRouterGenerator(OpenAIGenerator):
    def __init__(self, model: str = "deepseek/deepseek-chat", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key if api_key else getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=self.api_key,)
        
        self.aclient = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=self.api_key,)
        
    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", temperature: float = 0.5):
        return super().generate(prompt, system_prompt, temperature)
    
    async def agenerate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", temperature: float = 0.2):
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,)
            # print(response.choices[0].message.content) # For Debuging
            return response.choices[0].message.content
        except:
            return 'error'