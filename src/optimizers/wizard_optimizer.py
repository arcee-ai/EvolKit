from .base_optimizer import BaseOptimizer
from src.evolvers import BaseEvolver, RecurrentEvolver
from src.detectors import EvolutionFailureDetector
from src.generators import OpenAIGenerator, OpenRouterGenerator, BaseGenerator
from src.utils import parse_steps

import concurrent.futures
from typing import List, Optional
import asyncio
import random

METHOD_EVOL_PROMPT = """
Feedback: {feedback}
You are an Instruction Method Optimizer. Based on the feedback from the evolution failure case, optimize the method below to create a more effective instruction rewriting process without negatively impacting performance on other cases. Ensure that the complexity of the optimized method is not lower than the previous method.
If the feedback is "### PASSED", then come up with a better method than the current one to create a more complex and effective instruction rewriting process. Remember that the new method should not be very similar to the current method, be creative with new steps for the new method.

Current Method:
{current_method}

Please generate the optimized method strictly using ONLY the following format:

```Optimized Method
Step 1: #Methods List#
Describe how to generate a list of methods to make instructions more complex, incorporating the feedback

Step 2: #Plan#
Explain how to create a comprehensive plan based on the Methods List

[Note]Add as many steps here as you want to achieve the best method but N cannot be more than 7. The steps should align with the instruction domain/topic, and should not involve any tools or visualization, it should be text-only methods. The last step should always be #Finally Rewritten Instruction#.

Step N-1: #Rewritten Instruction#
Do not generate new Instruction here, but please provide a detailed the process of executing the plan to rewrite the instruction. You are generating a guide to write a better instruction, NOT THE INSTRUCTION ITSELF.

Step N: #Finally Rewritten Instruction#
Do not generate new Instruction here, but please provide the process to write the final rewritten instruction. You are generating a guide to write a better instruction, NOT THE INSTRUCTION ITSELF.```
"""

class WizardOptimizer(BaseOptimizer):
    def __init__(self, generator: BaseGenerator, failure_detector: EvolutionFailureDetector) -> None:
        self.generator = generator
        self.failure_detector = failure_detector

    async def optimize(self, current_method: str, feedback: List[str], evolver: RecurrentEvolver, development_set: Optional[List] = None):
        async def generate_and_evaluate(feedback_item):
            # Generate evolved method
            optimized_prompt = METHOD_EVOL_PROMPT.format(feedback=feedback_item, current_method=current_method)
            evolved_method = await self.generator.agenerate(optimized_prompt)

            # Generate responses for all instructions in parallel
            async def process_instruction(instruction):
                parsed_steps = parse_steps(evolved_method)
                new_method = evolver.build_new_method(parsed_steps, instruction)
                evolved_instruction = await self.generator.agenerate(prompt=new_method, temperature=0.5)
                parsed_evolved_instruction = parse_steps(evolved_instruction)[-1]['step_instruction']
                response = await self.generator.agenerate(parsed_evolved_instruction)
                return response

            responses = await asyncio.gather(*[process_instruction(instruction) for instruction in development_set])

            # Calculate failure rate
            failure_rate = self.failure_detector.calculate_failure_rate(responses)

            return evolved_method, failure_rate

        # Run all generations and evaluations in parallel
        results = await asyncio.gather(*[generate_and_evaluate(item) for item in feedback])

        # Find the best method
        evolved_methods, failure_rates = zip(*results)
        min_index = failure_rates.index(min(failure_rates))

        return evolved_methods[min_index], list(evolved_methods)