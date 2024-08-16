import asyncio
from typing import List
from src.generators import OpenRouterGenerator
from src.optimizers.wizard_optimizer import WizardOptimizer
from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer
from src.evolvers.recurrent_evolver import RecurrentEvolver, INITIAL_EVOLVE_METHOD
from src.evaluator.failure_detector_evaluator import FailureDetectorEvaluator
from src.utils import parse_steps

async def process_instruction(instruction: str, components: dict) -> List[str]:
    instruction_stages = [instruction]
    current_method = INITIAL_EVOLVE_METHOD.format(instruction=instruction_stages[0])

    for i in range(2):
        evolved_instructions = components['evolver'].evolve(instruction_stages[-1], current_method, n=5)
        feedbacks = components['analyzer'].analyze(instruction_stages[-1], evolved_instructions)
        
        optimized_method, _ = await components['optimizer'].optimize(
            current_method, 
            feedback=feedbacks, 
            evolver=components['evolver'], 
            development_set=components['dev_set']
        )
        
        optimized_method_steps = parse_steps(optimized_method)
        optimized_method = components['evolver'].build_new_method(optimized_method_steps, instruction_stages[-1])
        
        evolved_instruction = await components['generator'].agenerate(prompt=optimized_method, temperature=0.2)
        evolved_instruction_steps = parse_steps(evolved_instruction)
        
        if evolved_instruction_steps[-1]['step_name'] == 'Finally Rewritten Instruction':
            evolved_instruction = evolved_instruction_steps[-1]['step_instruction']
            instruction_stages.append(evolved_instruction)
            current_method = components['evolver'].build_new_method(optimized_method_steps, evolved_instruction)
        else:
            print(evolved_instruction)
            print('Error: Unexpected step name in evolved instruction')

    return instruction_stages

async def process_dataset(dataset: List[str], dev_set: List[str]) -> List[List[str]]:
    print(f"Starting dataset processing. Dataset size: {len(dataset)}")
    
    components = {
        'generator': OpenRouterGenerator(model='openai/gpt-4o'),
        'evolver': RecurrentEvolver(OpenRouterGenerator(model='openai/gpt-4o')),
        'analyzer': TrajectoryAnalyzer(OpenRouterGenerator(model='openai/gpt-4o')),
        'detector': FailureDetectorEvaluator(),
        'dev_set': dev_set
    }
    components['optimizer'] = WizardOptimizer(components['generator'], components['detector'])
    
    print("Initialized all components")

    async def process_batch(batch: List[str]) -> List[List[str]]:
        return await asyncio.gather(*[process_instruction(instruction, components) for instruction in batch])

    batch_size = 10
    results = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}")
        results.extend(await process_batch(batch))

    print("Dataset processing complete")
    return results

import pytest

@pytest.mark.asyncio
async def test_process_dataset():
    print("Starting test_process_dataset")
    dataset = [
        "Write a function to calculate the factorial of a number",
        "Explain the concept of recursion in programming",
        "Design a simple to-do list application",
        "Describe the process of photosynthesis",
        "Write a short story about a time traveler",
        "Explain the theory of relativity",
        "Create a recipe for a three-course meal",
        "Write a short story about a time traveler",
        "Explain the theory of relativity",
        "Create a recipe for a three-course meal"
    ]
    dev_set = [
        'Write a python function to perform bubble sort',
        'Write a letter to my headmaster asking for a day off.',
        'Write a python function to perform bubble sort',
        'Write a letter to my headmaster asking for a day off.',
        'Write a python function to perform bubble sort',
    ]
    
    print(f"Dataset size: {len(dataset)}, Dev set size: {len(dev_set)}")
    
    evolved_dataset = await process_dataset(dataset, dev_set)
    
    print("Dataset processing complete. Running assertions.")
    
    assert len(evolved_dataset) == len(dataset)
    for i, evolved_instructions in enumerate(evolved_dataset):
        print(f"Checking evolved instructions for dataset item {i+1}")
        assert len(evolved_instructions) == 3 # Original + 2 evolutions
        assert evolved_instructions[0] in dataset
        
        for j in range(1, len(evolved_instructions)):
            assert evolved_instructions[j] != evolved_instructions[j-1]
        print(f"All checks passed for dataset item {i+1}")

    for i in range(len(evolved_dataset)):
        for j in range(i+1, len(evolved_dataset)):
            assert evolved_dataset[i] != evolved_dataset[j]
    
    print("All assertions passed. Test complete.")

if __name__ == "__main__":
    pytest.main([__file__])