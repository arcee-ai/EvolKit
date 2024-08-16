import time
import json
import asyncio
import argparse
from datasets import load_dataset
from src.generators import OpenRouterGenerator
from src.evolvers import RecurrentEvolver
from src.analyzers import TrajectoryAnalyzer
from src.evaluator import FailureDetectorEvaluator, RewardModelEvaluator
from src.optimizers.wizard_optimizer import WizardOptimizer
from src import AutoEvol

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
        
    # Split the dataset
    train_instructions = full_filtered[dev_set_size:]
    dev_instructions = full_filtered[:dev_set_size]
    
    return train_instructions, dev_instructions
    
async def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

async def main():
    parser = argparse.ArgumentParser(description="Run AutoEvol with specified parameters")
    parser.add_argument("--dataset", help="Name of the dataset on Hugging Face")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--mini_batch_size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--num_methods", type=int, default=3, help="Number of methods to use")
    parser.add_argument("--max_concurrent_batches", type=int, default=5, help="Maximum number of concurrent batches")
    parser.add_argument("--evolve_epoch", type=int, default=3, help="Maximum number of epoch for each instruction")
    parser.add_argument("--dev_set_size", type=int, default=5, help="Maximum samples for dev set")
    parser.add_argument("--output_file", type=str, default='output.json', help="Name of output file")
    parser.add_argument("--use_reward_model", action="store_true", help="just a flag argument")
    
    args = parser.parse_args()
    
    # Load and process the dataset
    train_set, dev_set = load_and_process_dataset(args.dataset, args.dev_set_size)
    
    components = {
        'generator': OpenRouterGenerator(model='anthropic/claude-3.5-sonnet:beta'),
        'evolver': RecurrentEvolver(OpenRouterGenerator(model='anthropic/claude-3.5-sonnet:beta')),
        'analyzer': TrajectoryAnalyzer(OpenRouterGenerator(model='openai/gpt-4o')),
        'evaluator': RewardModelEvaluator() if args.use_reward_model else FailureDetectorEvaluator(),
        'dev_set': dev_set
    }
    components['optimizer'] = WizardOptimizer(OpenRouterGenerator(model='openai/gpt-4o'), components['evaluator'])
    
    auto_evol = AutoEvol(components)
    
    print(f"Dataset: {args.dataset}")
    print(f"Train set size: {len(train_set)}, Dev set size: {len(dev_set)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of methods: {args.num_methods}")
    print(f"Max concurrent batches: {args.max_concurrent_batches}")
    
    start_time = time.time()
    
    output_file = args.output_file
    all_results = []
    
    total_batches = (len(train_set) + args.batch_size - 1) // args.batch_size  # Calculate total number of batches

    for i in range(0, len(train_set), args.batch_size):
        batch = train_set[i:i+args.batch_size]
        batch_results = await auto_evol.run(batch, batch_size=args.mini_batch_size, num_methods=args.num_methods, max_concurrent_batches=args.max_concurrent_batches, evolve_epoch=args.evolve_epoch)
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