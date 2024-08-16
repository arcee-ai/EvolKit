import asyncio
import time
from typing import List, Dict, Any
from src.evolvers.recurrent_evolver import INITIAL_EVOLVE_METHOD
from .utils import parse_steps
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class AutoEvol:
    def __init__(self, components: Dict[str, Any]):
        self.components = components

    async def process_instruction(self, instruction: str, num_methods: int, evolve_epoch: int = 2) -> Dict[str, Any]:
        start_time = time.time()
        instruction_stages = [instruction]
        methods = [INITIAL_EVOLVE_METHOD.format(instruction=instruction_stages[0])]
        current_method = methods[0]

        result = {
            "original_instruction": instruction,
            "stages": []
        }

        for i in range(evolve_epoch):
            stage_start_time = time.time()
            stage_result = {
                "stage": i + 1,
                "input_instruction": instruction_stages[-1],
                "method": current_method,
                "evolved_instructions": [],
                "feedbacks": [],
                "optimized_method": "",
                "final_evolved_instruction": ""
            }
            
            evolved_instructions = await self.components['evolver'].evolve_async(instruction_stages[-1], current_method, n=num_methods)
            
            feedbacks = await self.components['analyzer'].analyze_async(instruction_stages[-1], evolved_instructions)
            
            stage_result["evolved_instructions"] = evolved_instructions
            stage_result["feedbacks"] = feedbacks

            optimized_method, _ = await self.components['optimizer'].optimize(
                current_method, 
                feedback=feedbacks, 
                evolver=self.components['evolver'], 
                development_set=self.components['dev_set']
            )
            
            optimized_method_steps = parse_steps(optimized_method)
            optimized_method = self.components['evolver'].build_new_method(optimized_method_steps, instruction_stages[-1])
            
            stage_result["optimized_method"] = optimized_method

            evolved_instruction = await self.components['generator'].agenerate(prompt=optimized_method, temperature=0.2)
            evolved_instruction_steps = parse_steps(evolved_instruction)
            
            if evolved_instruction_steps[-1]['step_name'] == 'Finally Rewritten Instruction':
                evolved_instruction = evolved_instruction_steps[-1]['step_instruction']
            else:
                print('Error: Unexpected step name in evolved instruction')
                evolved_instruction = instruction_stages[-1]  # Append the same instruction as before

            instruction_stages.append(evolved_instruction)
            methods.append(optimized_method)
            current_method = self.components['evolver'].build_new_method(optimized_method_steps, evolved_instruction)

            stage_result["final_evolved_instruction"] = evolved_instruction
            stage_end_time = time.time()
            stage_result["stage_time"] = stage_end_time - stage_start_time
            result["stages"].append(stage_result)

        result["final_instruction"] = instruction_stages[-1]
        end_time = time.time()
        result["total_time"] = end_time - start_time
        return result
    
    async def process_batch(self, batch: List[str], num_methods: int, evolve_epoch: int, pbar: tqdm) -> List[Dict[str, Any]]:
        batch_results = await asyncio.gather(*[self.process_instruction(instruction, num_methods, evolve_epoch) for instruction in batch])
        pbar.update(len(batch))
        return batch_results

    async def run(self, dataset: List[str], batch_size: int = 10, num_methods: int = 5, max_concurrent_batches: int = 2, evolve_epoch: int = 2) -> List[Dict[str, Any]]:
        print(f"Starting dataset processing. Dataset size: {len(dataset)}, Max concurrent batches: {max_concurrent_batches}")
        start_time = time.time()

        batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
        pbar = tqdm(total=len(dataset), desc="Processing instructions")

        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_batch_with_semaphore(batch):
            async with semaphore:
                return await self.process_batch(batch, num_methods, evolve_epoch, pbar)

        results = await asyncio.gather(*[process_batch_with_semaphore(batch) for batch in batches])

        pbar.close()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nDataset processing complete. Total time: {total_time:.2f} seconds")
        return [item for sublist in results for item in sublist]  # Flatten the list of batch results