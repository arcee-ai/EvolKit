import asyncio
import json
import time
from typing import List, Dict, Any
from src.generators import OpenRouterGenerator
from src.optimizers.wizard_optimizer import WizardOptimizer
from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer
from src.evolvers.recurrent_evolver import RecurrentEvolver, INITIAL_EVOLVE_METHOD
from src.evaluator.failure_detector_evaluator import FailureDetectorEvaluator
from src.utils import parse_steps

async def process_instruction(instruction: str, components: dict) -> Dict[str, Any]:
    start_time = time.time()
    instruction_stages = [instruction]
    methods = [INITIAL_EVOLVE_METHOD.format(instruction=instruction_stages[0])]
    current_method = methods[0]

    result = {
        "original_instruction": instruction,
        "stages": []
    }

    for i in range(2):
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
        
        evolved_instructions = await components['evolver'].evolve_async(instruction_stages[-1], current_method, n=5)
        
        feedbacks = await components['analyzer'].analyze_async(instruction_stages[-1], evolved_instructions)
        
        stage_result["evolved_instructions"] = evolved_instructions
        stage_result["feedbacks"] = feedbacks

        optimized_method, _ = await components['optimizer'].optimize(
            current_method, 
            feedback=feedbacks, 
            evolver=components['evolver'], 
            development_set=components['dev_set']
        )
        
        optimized_method_steps = parse_steps(optimized_method)
        optimized_method = components['evolver'].build_new_method(optimized_method_steps, instruction_stages[-1])
        
        stage_result["optimized_method"] = optimized_method

        evolved_instruction = await components['generator'].agenerate(prompt=optimized_method, temperature=0.2)
        evolved_instruction_steps = parse_steps(evolved_instruction)
        
        if evolved_instruction_steps[-1]['step_name'] == 'Finally Rewritten Instruction':
            evolved_instruction = evolved_instruction_steps[-1]['step_instruction']
        else:
            print('Error: Unexpected step name in evolved instruction')
            evolved_instruction = instruction_stages[-1]  # Append the same instruction as before

        instruction_stages.append(evolved_instruction)
        methods.append(optimized_method)
        current_method = components['evolver'].build_new_method(optimized_method_steps, evolved_instruction)

        stage_result["final_evolved_instruction"] = evolved_instruction
        stage_end_time = time.time()
        stage_result["stage_time"] = stage_end_time - stage_start_time
        result["stages"].append(stage_result)

    result["final_instruction"] = instruction_stages[-1]
    end_time = time.time()
    result["total_time"] = end_time - start_time
    return result

async def process_dataset(dataset: List[str], dev_set: List[str]) -> List[Dict[str, Any]]:
    print(f"Starting dataset processing. Dataset size: {len(dataset)}")
    start_time = time.time()
    
    components = {
        'generator': OpenRouterGenerator(model='anthropic/claude-3.5-sonnet:beta'),
        'evolver': RecurrentEvolver(OpenRouterGenerator(model='anthropic/claude-3.5-sonnet:beta')),
        'analyzer': TrajectoryAnalyzer(OpenRouterGenerator(model='openai/gpt-4o')),
        'detector': FailureDetectorEvaluator(),
        'dev_set': dev_set
    }
    components['optimizer'] = WizardOptimizer(OpenRouterGenerator(model='openai/gpt-4o'), components['detector'])
    
    print("Initialized all components")

    async def process_batch(batch: List[str]) -> List[Dict[str, Any]]:
        return await asyncio.gather(*[process_instruction(instruction, components) for instruction in batch])

    batch_size = 10
    results = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_start_time = time.time()
        print(f"Processing batch {i//batch_size + 1}")
        batch_results = await process_batch(batch)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        print(f"Batch {i//batch_size + 1} completed in {batch_time:.2f} seconds")
        results.extend(batch_results)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Dataset processing complete. Total time: {total_time:.2f} seconds")
    return results

async def main():
    dataset = [
        "Explain the concept of quantum entanglement in simple terms",
        "Write a Python function to implement a binary search algorithm",
        "Describe the process of photosynthesis in plants",
        "Compose a haiku about artificial intelligence",
        "Explain the importance of the gut microbiome in human health",
        "Design a basic relational database schema for a library management system",
        "Summarize the main arguments for and against universal basic income",
        "Write a short story about a world where dreams are shared",
        "Explain the concept of blockchain technology and its potential applications",
        "Describe the key features of Renaissance art and its impact on Western culture"
    ]
    dev_set = [
        "Explain the theory of evolution by natural selection",
        "Write a JavaScript function to reverse a string without using built-in methods",
        "Describe the water cycle and its importance to Earth's ecosystems",
        "Create a marketing strategy for a new eco-friendly product",
        "Explain the basics of machine learning and its applications in daily life"
    ]
    
    print(f"Dataset size: {len(dataset)}, Dev set size: {len(dev_set)}")
    
    start_time = time.time()
    results = await process_dataset(dataset, dev_set)
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("Writing results to JSON file")
    with open('instruction_evolution_results-sonnet-gpt4o-1.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("Results written to instruction_evolution_results-sonnet-gpt4o-1.json")

if __name__ == "__main__":
    asyncio.run(main())