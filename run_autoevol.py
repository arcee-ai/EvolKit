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
    
    # Split the dataset
    dev_set = full_dataset[:dev_set_size]
    train_set = full_dataset[dev_set_size:]
    
    train_instructions = []
    dev_instructions = []
    
    for train_sample in train_set:
        convo = train_sample['conversations']
        for turn in convo:
            if turn['from'] == 'human':
                train_instructions.append(turn['value'])
                break  # Only take the first human instruction from each conversation
    
    for dev_sample in dev_set:
        convo = dev_sample['conversations']
        for turn in convo:
            if turn['from'] == 'human':
                dev_instructions.append(turn['value'])
                break  # Only take the first human instruction from each conversation
    
    return train_instructions, dev_instructions
    
async def main():
    parser = argparse.ArgumentParser(description="Run AutoEvol with specified parameters")
    parser.add_argument("dataset", help="Name of the dataset on Hugging Face")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--num_methods", type=int, default=2, help="Number of methods to use")
    parser.add_argument("--max_concurrent_batches", type=int, default=2, help="Maximum number of concurrent batches")
    parser.add_argument("--use_reward_model",action="store_true",help="just a flag argument")
    
    args = parser.parse_args()
    
    # Load and process the dataset
    train_set, dev_set = load_and_process_dataset(args.dataset)
    
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
    results = await auto_evol.run(train_set, batch_size=args.batch_size, num_methods=args.num_methods, max_concurrent_batches=args.max_concurrent_batches)
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("Writing results to JSON file")
    output_file = f'instruction_evolution_results-{args.dataset.replace("/", "-")}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())