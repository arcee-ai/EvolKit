import asyncio
import json
import argparse
from typing import List, Dict
from src.generators import OpenRouterGenerator, VLLMGenerator, BaseGenerator
from datasets import load_dataset
from tqdm import tqdm
import time
from os import getenv

async def process_batch(generator: BaseGenerator, batch: List[str], system_prompt: str) -> List[Dict]:
    tasks = []
    for item in batch:
        try:
            task = generator.agenerate(item, system_prompt, temperature=0.5)
            tasks.append(task)
        except:
            task = 'error'
            tasks.append(task)
    results = await asyncio.gather(*tasks)
    
    processed_items = []
    for item, result in zip(batch, results):
        processed_item = {
            'conversations': [
                {"from": "human", "value": item},
                {"from": "gpt", "value": result}
            ]
        }
        processed_items.append(processed_item)
    
    return processed_items

async def process_data(model:str, generator_str: str, file_path: str, batch_size: int, output_file: str):
    # Initialize OpenRouterGenerator
    generator = (
        VLLMGenerator(model=model, base_url=getenv('VLLM_BACKEND') or 'http://localhost:8000/v1') 
        if generator_str == 'vllm' 
        else OpenRouterGenerator(model=model))
    
    # Load data
    if '.json' not in file_path:
        data = load_dataset(file_path)['train']  # Assuming the main split is named 'train'
    else:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
    start_idx = 0
    
    # Calculate total number of batches
    total_batches = (len(data) + batch_size - start_idx - 1) // batch_size
    
    instructions = []
    for sample in data:
        convo = sample['conversations']
        if convo[0]['from'] == 'human':
            user = convo[0]['value']
        else:
            user = convo[1]['value']
        
        instructions.append(user)            
        
        
    # Process in batches
    all_results = []
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, len(data), batch_size):
            batch = instructions[i+start_idx:i+batch_size+start_idx]
            processed_batch = await process_batch(generator, batch, "You are a helpful assistant. Answer the question from the user. Give full solution and explaination.")
            
            # Extend results and save
            all_results.extend(processed_batch)
            
            # Save and overwrite for each batch
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            pbar.update(1)
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser(description="Process data using OpenRouterGenerator")
    parser.add_argument("--model", type=str, required=True, help="Model use to evol instructions.")
    parser.add_argument("--generator", type=str, required=True, choices=['openrouter', 'vllm'], help="Type of generator to use.")
    parser.add_argument("--data_path", required=True, help="Path to the JSON file or Hugging Face dataset repo")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--output", default="final_evolved_data.json", help="Output file path")
    
    args = parser.parse_args()
    
    asyncio.run(process_data(args.model, args.generator, args.data_path, args.batch_size, args.output))

if __name__ == "__main__":
    main()