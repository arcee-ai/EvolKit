# OpenAutoEvol

OpenAutoEvol is an innovative framework for automatically enhancing the complexity of instructions used in fine-tuning Large Language Models (LLMs). Our project aims to revolutionize the evolution process by leveraging open-source LLMs, moving away from closed-source alternatives.

## Key Features

- Automatic instruction complexity enhancement
- Integration with open-source LLMs
- Streamlined fine-tuning process
- Support for various datasets from Hugging Face
- Flexible configuration options for optimization

## Installation

To set up OpenAutoEvol, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/OpenAutoEvol.git
   cd OpenAutoEvol
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the AutoEvol script, use the following command structure:

```bash
python run_autoevol.py --dataset <dataset_name> [options]
```

### Required Parameters:

- `--dataset <dataset_name>`: The name of the dataset on Hugging Face to use.

### Optional Parameters:

- `--batch_size <int>`: Number of instructions to process in each batch. Default is 5.
- `--mini_batch_size <int>`: Number of instructions to process in each mini-batch. Default is 5.
- `--num_methods <int>`: Number of evolution methods to use. Default is 3.
- `--max_concurrent_batches <int>`: Maximum number of batches to process concurrently. Default is 5.
- `--evolve_epoch <int>`: Maximum number of epochs for evolving each instruction. Default is 3.
- `--dev_set_size <int>`: Number of samples to use in the development set. Default is 5.
- `--output_file <filename>`: Name of the output file to save results. Default is 'output.json'.
- `--use_reward_model`: Flag to use a reward model for finding the best method each round. No value required.

### Example Usage:

To run AutoEvol on the 'small_tomb' dataset with custom parameters:

```bash
python run_autoevol.py --dataset qnguyen3/small_tomb --batch_size 100 --mini_batch_size 10 --num_methods 3 --max_concurrent_batches 10 --evolve_epoch 3 --dev_set_size 3 --output_file the_tomb_evolved-3e-batch100.json --use_reward_model
```

This command will:
1. Load the 'qnguyen3/small_tomb' dataset from Hugging Face.
2. Use 3 samples for the development set.
3. Process the remaining samples in the training set.
4. Use a batch size of 100 for processing, with mini-batches of 10.
5. Apply 3 evolution methods for each sample in the development set to find the best method.
6. Process 10 mini-batches (10*10=100 samples) concurrently.
7. Use the reward model for evaluation.
8. Output the final evolved instructions to `the_tomb_evolved-3e-batch100.json`.

## Components

OpenAutoEvol consists of several key components:

- **Generator**: Uses the Claude 3.5 Sonnet model for generating initial instructions.
- **Evolver**: Employs a recurrent evolution strategy with Claude 3.5 Sonnet.
- **Analyzer**: Utilizes GPT-4 for trajectory analysis.
- **Evaluator**: Offers two options:
  - Reward Model Evaluator
  - Failure Detector Evaluator
- **Optimizer**: Implements a Wizard Optimizer using GPT-4.

## Output

The script saves the results in JSON format to the specified output file. Each entry in the JSON file represents an evolved instruction along with relevant metadata.

## License

[Specify your chosen license information here]