#!/usr/bin/env python3
"""
prepare_data.py - Prepare a text dataset for training by tokenizing and storing it in an HDF5 (.h5) file.

This script is part of the beginner-friendly CPU/Low-Hardware Student Mode.
It takes a raw text file (e.g., data/student/train.txt), tokenizes it using tiktoken
(default is r50k_base, the OpenAI GPT-3 tokenizer), and outputs an HDF5 file
ready for our data loader.

If the input dataset is missing, it automatically downloads and splits the public-domain
Tiny Shakespeare dataset into training (90%) and validation (10%) datasets so students
can run the pipeline out of the box with zero manual downloads.

Usage:
    # Out of the box (auto-downloads and tokenizes)
    python prepare_data.py

    # Custom input/output paths
    python prepare_data.py --input data/student/train.txt --output data/student/train.h5
"""

import argparse
import os
import sys
import urllib.request
import tiktoken
import h5py
import numpy as np

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def ensure_datasets_exist(train_path: str, val_path: str):
    """
    Checks if the training and validation raw text files exist.
    If not, downloads Tiny Shakespeare and splits it 90/10 into train and validation files.
    """
    train_exists = os.path.exists(train_path)
    val_exists = os.path.exists(val_path)

    if train_exists and val_exists:
        return

    print("=" * 60)
    print("                AUTOMATIC DATASET PREPARATION")
    print("=" * 60)
    print("One or both raw text files (train.txt / val.txt) were not found.")
    print("We will automatically download and split the public-domain Tiny Shakespeare")
    print("dataset so you can start training immediately.")
    print("-" * 60)

    # Ensure student directory exists
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)

    temp_raw_path = "data/student/shakespeare_raw.txt"
    print(f"Downloading Tiny Shakespeare from: {SHAKESPEARE_URL}")
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, temp_raw_path)
        print("Download successful!")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        print("Please check your internet connection or download a raw text file manually.", file=sys.stderr)
        sys.exit(1)

    # Split into 90% train, 10% val
    print("Splitting dataset: 90% for training, 10% for validation...")
    try:
        with open(temp_raw_path, "r", encoding="utf-8") as f:
            text = f.read()

        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        with open(train_path, "w", encoding="utf-8") as f:
            f.write(train_text)
        with open(val_path, "w", encoding="utf-8") as f:
            f.write(val_text)

        print(f"Created: {train_path} ({len(train_text):,} characters)")
        print(f"Created: {val_path} ({len(val_text):,} characters)")

    except Exception as e:
        print(f"[ERROR] Failed to split dataset: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up temporary raw download file
        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

    print("Dataset setup complete!")
    print("=" * 60 + "\n")

def tokenize_text_file(input_path: str, output_path: str, tokenizer_name: str = "r50k_base"):
    """
    Reads a raw text file, tokenizes it using tiktoken, and saves the resulting token IDs
    into an HDF5 file under the dataset 'tokens'.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    print(f"Reading raw text from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer '{tokenizer_name}'. Falling back to default 'r50k_base'.")
        enc = tiktoken.get_encoding("r50k_base")

    print("Tokenizing raw text... (this may take a few seconds)")
    # Append the end-of-text special token so the model learns boundaries
    encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
    num_tokens = len(encoded)
    print(f"Successfully tokenized! Total tokens: {num_tokens:,}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Writing tokens to HDF5 file: {output_path}")
    # Write the tokens dataset to HDF5
    with h5py.File(output_path, 'w') as f:
        # Save as a flat 1D array of 32-bit integers
        dset = f.create_dataset('tokens', data=np.array(encoded, dtype=np.int32))
        print(f"Created dataset 'tokens' with shape {dset.shape} and dtype {dset.dtype}")

    print("Data preparation complete! Ready for training.\n")

def main():
    parser = argparse.ArgumentParser(
        description="Educational script to tokenize raw text and output HDF5 datasets for CPU training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/student/train.txt",
        help="Path to the input text file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/student/train.h5",
        help="Path to save the output HDF5 (.h5) file."
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="r50k_base",
        help="tiktoken tokenizer encoding to use."
    )

    args = parser.parse_args()

    # If the user is using the default data/student/ paths, automatically check and resolve missing files
    if args.input == "data/student/train.txt":
        ensure_datasets_exist("data/student/train.txt", "data/student/val.txt")

    try:
        tokenize_text_file(args.input, args.output, args.tokenizer)
    except Exception as e:
        print(f"Error during data preparation: {e}")

if __name__ == "__main__":
    main()
