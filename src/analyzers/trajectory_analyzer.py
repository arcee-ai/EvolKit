import asyncio
from typing import List

from .base_analyzer import BaseAnalyzer
from src.generators import BaseGenerator

TRAJECTORY_ANALYZER_SYSTEM_PROMPT = """
You are an expert at analyzing the evolution of a given instruction. You will look at the trajectory of the evolution from an initial instruction and make feedbacks based on how the complexity is being increased in each stage.
"""

TRAJECTORY_ANALYZER_PROMPT = """
The following list shows cases where an Instruction evolves into a more complex version of an Instruction.
For each case, stage 0 represents the Instruction in its initial state, and stage 1 requires an increase in complexity based on the previous stage.

Please identify cases that failed to evolve, and provide the reason why it fails.

Please strictly output using the following format, do not add anything else to the response:

***FORMAT INSTRUCTION***
Choose one of the two options:
Option 1 - If all cases are evolving correctly, please strictly output:
### PASSED

Option 2 - If you identify cases that did not evolve correctly, please strictly output:
### FAILED - Reason: [reason_of_fail]
and so on...
***END OF FORMAT INSTRUCTION***

Evolution Trajectory:
{evol_trajectory}
"""

class TrajectoryAnalyzer(BaseAnalyzer):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator
        
    async def analyze_async(self, init_instruction: str, evolved_instructions: List[str]) -> List[str]:
        async def generate_single(evolved_instruction):
            trajectory_str = f"""
            Stage 0: {init_instruction}
            Stage 1: {evolved_instruction}
            """
            trajectory_prompt = TRAJECTORY_ANALYZER_PROMPT.format(evol_trajectory=trajectory_str)
            feedback = await self.generator.agenerate(prompt=trajectory_prompt, system_prompt=TRAJECTORY_ANALYZER_SYSTEM_PROMPT, temperature=0.2)
            
            return feedback
        
        tasks = [generate_single(evolved_instruction) for evolved_instruction in evolved_instructions]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def analyze(self, init_instruction: str, evolved_instructions: List[str]) -> List[str]:
        return asyncio.run(self.analyze_async(init_instruction, evolved_instructions))