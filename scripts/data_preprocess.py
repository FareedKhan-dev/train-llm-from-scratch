import argparse
import json
import os

import h5py
import tiktoken
import zstandard as zstd
from tqdm import tqdm


def process_files(input_dir: str, output_file: str, tokenizer_name: str, max_data: int | None = None) -> None:
    """
    Process a specified number of lines from each .jsonl.zst file in the input directory
    and save encoded tokens to an HDF5 file.

    Args:
        input_dir (str): Directory containing input .jsonl.zst files.
        output_file (str): Path to the output HDF5 file.
        tokenizer_name (str): Name of the tiktoken tokenizer to use (e.g., 'r50k_base').
        max_data (int, optional): Maximum number of lines to process from each file.
                                  If None, process all lines.
    """
    # Print processing strategy based on max_data
    if max_data is not None:
        print(f"You have chosen max_data = {max_data}. Processing only the top {max_data} JSON objects from each file.")
    else:
        print("Processing all available JSON objects from each file.")

    # Load the tokenizer using the provided tokenizer name
    enc = tiktoken.get_encoding(tokenizer_name)

    # Create an HDF5 file for output
    with h5py.File(output_file, 'w') as out_f:
        # Initialize the dataset for storing tokenized data
        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0  # Track the starting index for the next batch of tokens

        # Process each .jsonl.zst file in the input directory
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".jsonl.zst"):  # Only process .jsonl.zst files
                in_file = os.path.join(input_dir, filename)
                print(f"Processing: {in_file}")

                processed_lines = 0  # Counter for processed lines in the current file

                # Open the compressed .jsonl.zst file for reading
                with zstd.open(in_file, 'rt', encoding='utf-8') as in_f:
                    # Iterate over each line in the file
                    for line in tqdm(in_f, desc=f"Processing {filename}", total=max_data if max_data is not None else None):
                        try:
                            # Parse the line as JSON
                            data = json.loads(line)
                            text = data.get('text')  # Extract the 'text' field from the JSON object

                            if text:
                                # Tokenize the text and append an end-of-text token
                                encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
                                encoded_len = len(encoded)

                                # Resize the dataset to accommodate new tokens
                                end_index = start_index + encoded_len
                                dataset.resize(dataset.shape[0] + encoded_len, axis=0)

                                # Store the encoded tokens in the dataset
                                dataset[start_index:end_index] = encoded
                                start_index = end_index  # Update the start index
                            else:
                                # Warn if 'text' key is missing in the JSON object
                                print(f"Warning: 'text' key missing in line from {filename}")
                        except json.JSONDecodeError:
                            # Handle JSON decoding errors
                            print(f"Warning: Could not decode JSON from line in {filename}")
                        except Exception as e:
                            # Handle any other errors
                            print(f"An error occurred while processing line in {filename}: {e}")

                        processed_lines += 1
                        # Stop processing if max_data limit is reached
                        if max_data is not None and processed_lines >= max_data:
                            break

def main():
    """
    Main function to parse arguments, validate directories, and process files.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess PILE dataset files and save tokens to HDF5.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory containing training .jsonl.zst files.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory containing validation .jsonl.zst files.")
    parser.add_argument("--out_train_file", type=str, default="data/train/pile_train.h5", help="Path to the output training HDF5 file.")
    parser.add_argument("--out_val_file", type=str, default="data/val/pile_dev.h5", help="Path to the output validation HDF5 file.")
    parser.add_argument("--tokenizer_name", type=str, default="r50k_base", help="Name of the tiktoken tokenizer to use.")
    parser.add_argument("--max_data", type=int, default=1000, help="Maximum number of json objects to process from each file in both train and val datasets (default: 1000).")

    args = parser.parse_args()

    # Validate the existence of the training and validation directories
    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        return
    if not os.path.isdir(args.val_dir):
        print(f"Error: Validation directory not found: {args.val_dir}")
        return

    # Process training data
    print("Starting training data preprocessing...")
    process_files(args.train_dir, args.out_train_file, args.tokenizer_name, args.max_data)
    print("Training data preprocessing complete.")

    # Process validation data
    print("Starting validation data preprocessing...")
    process_files(args.val_dir, args.out_val_file, args.tokenizer_name, args.max_data)
    print("Validation data preprocessing complete.")

# Entry point of the script
if __name__ == "__main__":
    main()
