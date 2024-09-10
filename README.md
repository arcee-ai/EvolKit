# EvolKit

EvolKit is an framework for automatically enhancing the complexity of instructions used in fine-tuning Large Language Models (LLMs). Our project aims to revolutionize the evolution process by leveraging open-source LLMs, moving away from closed-source alternatives.

## Key Features

- Automatic instruction complexity enhancement
- Integration with open-source LLMs
- Streamlined fine-tuning process
- Support for various datasets from Hugging Face
- Flexible configuration options for optimization

## Installation

To set up EvolKit, follow these steps:

1. Clone the repository:
   
   ```
   git clone https://github.com/arcee-ai/EvolKit.git
   cd EvolKit
   ```

2. Install the required dependencies:
   
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the AutoEvol script, use the following command structure:

```
python run_evol.py --dataset <dataset_name> [options]
```

### Required Parameters:

- `--dataset <dataset_name>`: The name of the dataset on Hugging Face to use.
- `--model <model_name>`: Model to use for evolving instructions.
- `--generator <generator_type>`: Type of generator to use ('openrouter' or 'vllm').
- `--batch_size <int>`: Number of instructions to process in each batch.
- `--num_methods <int>`: Number of evolution methods to use.
- `--max_concurrent_batches <int>`: Maximum number of batches to process concurrently (in our experiment, a cluster of 8xH100 hosting Qwen2-72B-Instruct-GPTQ-Int8 can handle batch size of 50 concurrently).
- `--evolve_epoch <int>`: Maximum number of epochs for evolving each instruction.
- `--output_file <filename>`: Name of the output file to save results.

### Optional Parameters:

- `--dev_set_size <int>`: Number of samples to use in the development set. Use -1 for no dev set. Default is -1.
- `--use_reward_model`: Flag to use a reward model for evaluation. No value required.

### Models

We found 2 models that work very well with this pipeline:
- Qwen2-72B-Instruct and DeepSeek-V2.5 (GPTQ and AWQ versions are fine too).
- Other models might work but it has to be very good at generating structured content (in order to parse using parsing operations)

### VLLM Support

To use VLLM as the backend, set the `VLLM_BACKEND` environment variable:

```
export VLLM_BACKEND=http://your-vllm-backend-url:port/v1
```

If not set, it will default to 'http://localhost:8000/v1'.

### Example Usage:

To run AutoEvol on the 'small_tomb' dataset with custom parameters:

```
python run_evol.py --dataset qnguyen3/small_tomb --model Qwen/Qwen2-72B-Instruct-GPTQ-Int8 --generator vllm --batch_size 100 --num_methods 3 --max_concurrent_batches 10 --evolve_epoch 3 --output_file the_tomb_evolved-3e-batch100.json --dev_set_size 5 --use_reward_model
```

This command will:
1. Load the 'qnguyen3/small_tomb' dataset from Hugging Face.
2. Use the Qwen2-72B-Instruct model with VLLM as the generator.
3. Process samples in batches of 100.
4. Apply 3 evolution methods for each instruction.
5. Process 10 batches concurrently.
6. Evolve each instruction for up to 3 epochs.
7. Use 5 samples for the development set.
8. Use the reward model for evaluation.
9. Output the final evolved instructions to the_tomb_evolved-3e-batch100.json.

After evolving the instructions, you can generate answers using:

```
python gen_answers.py --data_path the_tomb_evolved-3e-batch100.json --batch_size 50 --output completed_evol_data.json
```

The final dataset will be saved to completed_evol_data.json in ShareGPT format.

## Components

EvolKit consists of several key components:

- **Generator**: Uses an LLM for generating initial instructions (OpenRouter or VLLM).
- **Evolver**: Employs a recurrent evolution strategy.
- **Analyzer**: Utilizes trajectory analysis.
- **Evaluator**: Offers two options:
  - Reward Model Evaluator
  - Failure Detector Evaluator
- **Optimizer**: Optimizes the evolution method for the next round.

## Output

The script saves the results in JSON format to the specified output file. Each entry in the JSON file represents an evolved instruction along with relevant metadata.

Find a 20k subset of a dataset generated using EvolKit [here](https://huggingface.co/datasets/arcee-ai/EvolKit-20k)

## Acknowledgement
- Microsoft's WizardLM team for the inspiration from the [AutoEvol paper](https://arxiv.org/pdf/2406.00770).
