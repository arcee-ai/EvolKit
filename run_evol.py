import time
import json
import asyncio
import argparse
from datasets import load_dataset
from src.generators import OpenRouterGenerator, VLLMGenerator
from src.evolvers import RecurrentEvolver
from src.analyzers import TrajectoryAnalyzer
from src.evaluator import FailureDetectorEvaluator, RewardModelEvaluator
from src.optimizers.evol_optimizer import EvolOptimizer
from src import AutoEvol
from os import getenv

def load_and_process_dataset(dataset_name, dev_set_size=5):
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Assuming the dataset has a 'train' split. Adjust if needed.
    if 'train' not in dataset:
        raise ValueError(f"The dataset {dataset_name} does not have a 'train' split.")
    
    full_dataset = dataset['train']
    
    # Shuffle the dataset
    full_dataset = full_dataset.shuffle(seed=42)
    
    # Ensure dev_set_size is not larger than the dataset
    if dev_set_size > len(full_dataset):
        raise ValueError(f"Specified dev set size ({dev_set_size}) is larger than the dataset size ({len(full_dataset)})")
    full_filtered = []
    for sample in full_dataset['conversations']:
        for turn in sample:
            if turn['from'] == 'system':
                break
            if turn['from'] == 'human':
                full_filtered.append(turn['value'])
                break
    
    if dev_set_size != -1:
        # Split the dataset
        train_instructions = full_filtered[dev_set_size:]
        dev_instructions = full_filtered[:dev_set_size]
        
        return train_instructions, dev_instructions
    else: 
        return full_filtered, []
    
async def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

async def main():
    parser = argparse.ArgumentParser(description="Run AutoEvol with specified parameters")
    parser.add_argument("--dataset", required=True, help="Name of the dataset on Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model use to evol instructions.")
    parser.add_argument("--generator", type=str, required=True, choices=['openrouter', 'vllm'], help="Type of generator to use.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing")
    parser.add_argument("--num_methods", type=int, required=True, help="Number of methods to use")
    parser.add_argument("--max_concurrent_batches", type=int, required=True, help="Maximum number of concurrent batches")
    parser.add_argument("--evolve_epoch", type=int, required=True, help="Maximum number of epoch for each instruction")
    parser.add_argument("--output_file", type=str, required=True, help="Name of output file")
    
    # Optional arguments
    parser.add_argument("--dev_set_size", type=int, default=-1, help="Maximum samples for dev set. Use -1 for no dev set.")
    parser.add_argument("--use_reward_model", action="store_true", help="Use reward model for evaluation")
    
    args = parser.parse_args()
    
    # Load and process the dataset
    train_set, dev_set = load_and_process_dataset(args.dataset, args.dev_set_size)
    
    generator = (
    VLLMGenerator(model=args.model, base_url=getenv('VLLM_BACKEND') or 'http://localhost:8000/v1') 
    if args.generator == 'vllm' 
    else OpenRouterGenerator(model=args.model)
    )


    components = {
        'generator': generator,
        'evolver': RecurrentEvolver(generator),
        'analyzer': TrajectoryAnalyzer(generator),
        'evaluator': RewardModelEvaluator() if args.use_reward_model else FailureDetectorEvaluator(),
        'dev_set': dev_set
    }
    components['optimizer'] = EvolOptimizer(generator, components['evaluator'])
    
    auto_evol = AutoEvol(components)
    
    print(f"Dataset: {args.dataset}")
    if args.dev_set_size != -1:
        print(f"Train set size: {len(train_set)}, Dev set size: {len(dev_set)}")
    else:
        print(f"Train set size: {len(train_set)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of methods: {args.num_methods}")
    print(f"Max concurrent batches: {args.max_concurrent_batches}")
    
    start_time = time.time()
    
    output_file = args.output_file
    all_results = []
    
    total_batches = (len(train_set) + args.batch_size - 1) // args.batch_size  # Calculate total number of batches

    for i in range(0, len(train_set), args.batch_size):
        batch = train_set[i:i+args.batch_size]
        batch_results = await auto_evol.run(batch, batch_size=args.batch_size, num_methods=args.num_methods, max_concurrent_batches=args.max_concurrent_batches, evolve_epoch=args.evolve_epoch)
        all_results.extend(batch_results)
        
        current_batch = i // args.batch_size + 1
        print(f"Done batch {current_batch}/{total_batches}")  # New print statement for batch progress
        print(f"Batch {current_batch} completed. Saving results...")
        await save_results(all_results, output_file)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())