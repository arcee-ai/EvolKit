from .base_evolver import BaseEvolver
from src.generators.base_generator import BaseGenerator

import concurrent.futures
from typing import List

INITIAL_EVOLVE_METHOD = """
You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" below carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any explanation.

#Instruction#: {instruction}

**Output formatting instructions**
Please generate the optimized method strictly using ONLY the given below format, do not add anything else.
Always start and end the message with '```'

```Optimized Insutrction
Step 1:
#Methods List#

Step 2:
#Plan#

Step 3:
#Rewritten Instruction#

Step 4:
#Finally Rewritten Instruction#
```
"""

INTERATIVE_EVOLVE_METHOD = """
You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

{steps}

#Instruction#: {instruction}

**Output formatting instructions**
Please generate the optimized method strictly using ONLY the given below format, do not add anything else.
Always start and end the message with '```'

```Optimized Instruction
{format_steps}
```
"""

class RecurrentEvolver(BaseEvolver):
    def __init__(self, generator: BaseGenerator) -> None:
        self.generator = generator
        
    def evolve(self, instruction: str, evolving_method: str = None, n: int = 1) -> List[str]: # type: ignore
        evol_method = evolving_method.format(instruction=instruction) if evolving_method else INITIAL_EVOLVE_METHOD.format(instruction=instruction)
        
        def generate_single():
            return self.generator.generate(evol_method)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_single) for _ in range(n)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results
    
    def build_new_method(self, steps, instruction):
        
        step_details = ""
        format_steps = ""
        
        for i, step in enumerate(steps, start=1):
            step_name = step['step_name']
            step_instruction = step['step_instruction']
                
            step_details += f"Step {i}: {step_instruction}\n\n"
            format_steps += f"Step {i}: #{step_name}#\n\n"
        
        new_method = INTERATIVE_EVOLVE_METHOD.format(steps=step_details.strip(), instruction=instruction, format_steps=format_steps.strip())
        return new_method