# OpenAutoEvol

OpenAutoEvol is an innovative framework for automatically enhancing the complexity of instructions used in fine-tuning Large Language Models (LLMs). Our project aims to revolutionize the evolution process by leveraging open-source LLMs, moving away from closed-source alternatives.

## Key Features

- Automatic instruction complexity enhancement
- Integration with open-source LLMs
- Streamlined fine-tuning process

## Running AutoEvol

To run the AutoEvol script, use the following command structure:

```bash
python run_autoevol.py <dataset_name> [options]
```

### Parameters:

- `<dataset_name>`: (Required) The name of the dataset on Hugging Face to use.

### Options:

- `--batch_size <int>`: Number of instructions to process in each batch. Default is 5.
- `--num_methods <int>`: Number of evolution methods to use. Default is 2.
- `--max_concurrent_batches <int>`: Maximum number of batches to process concurrently. Default is 2.
- `--dev_set_size <int>`: Number of samples to use in the development set. Default is 5.
- `--evolve_epoch <int>`: Max epoch for each instruction when evolve.
- `--use_reward_model <store_true>`: whether or not using a reward model to find the best method each round.

### Example Usage:

To run AutoEvol on the 'small_tomb' dataset with custom parameters:

```bash
python main.py qnguyen3/small_tomb --batch_size 10 --num_methods 3 --max_concurrent_batches 2 --dev_set_size 3
```

This command will:
1. Load the 'qnguyen3/small_tomb' dataset from Hugging Face.
2. Use 3 samples for the development set.
3. Process the remaining samples in the training set.
4. Use a batch size of 10 for processing.
5. Apply 3 evolution methods for each sample in development set to find the best method 
6. Process 2 batches concurrently.

### Output:

The script will output progress information to the console and save the results in a JSON file named `instruction_evolution_results-<dataset_name>.json` in the current directory.

### Note:

Ensure you have the required dependencies installed and have access to the specified Hugging Face dataset before running the script.

## Contributing

We welcome contributions from the community! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get involved.

## License

[Your chosen license information]

## Contact

[Your preferred method of contact or support channels]
