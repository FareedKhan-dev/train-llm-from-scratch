import argparse
import os

import requests
from tqdm import tqdm

# Base URL for the dataset files
BASE_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"
VAL_URL = f"{BASE_URL}/val.jsonl.zst"  # URL for the validation dataset
TRAIN_URLS = [f"{BASE_URL}/train/{i:02d}.jsonl.zst" for i in range(65)]  # URLs for 65 training files (adjust the range if needed)

def download_file(url: str, file_name: str) -> None:
    """
    Downloads a file from the given URL and saves it with the specified file name.
    Displays a progress bar using tqdm.

    Args:
        url (str): The URL of the file to download.
        file_name (str): The local path where the file will be saved.
    """
    print(f"Downloading: {file_name}...")
    response = requests.get(url, stream=True)  # Stream the file content
    total_size = int(response.headers.get('content-length', 0))  # Get total file size if available
    block_size = 1024  # Size of each block for the progress bar
    with open(file_name, 'wb') as f:  # Open file for writing in binary mode
        for chunk in tqdm(response.iter_content(block_size), total=total_size // block_size, desc="Downloading", leave=True):
            f.write(chunk)  # Write each chunk to the file

def download_dataset(val_url: str, train_urls: list[str], val_dir: str, train_dir: str, max_train_files: int) -> None:
    """
    Manages downloading of the dataset, including both validation and training files.

    Args:
        val_url (str): URL for the validation dataset.
        train_urls (list): List of URLs for the training dataset files.
        val_dir (str): Directory where the validation file will be stored.
        train_dir (str): Directory where the training files will be stored.
        max_train_files (int): Maximum number of training files to download.
    """
    # Define the path for the validation file
    val_file_path = os.path.join(val_dir, "val.jsonl.zst")
    if not os.path.exists(val_file_path):  # Check if the validation file already exists
        print(f"Validation file not found. Downloading from {val_url}...")
        download_file(val_url, val_file_path)  # Download the validation file
    else:
        print("Validation data already present. Skipping download.")

    # Loop through the training file URLs and download if not already present
    for idx, url in enumerate(train_urls[:max_train_files]):  # Limit to max_train_files
        file_name = f"{idx:02d}.jsonl.zst"  # Format file name (e.g., 00.jsonl.zst)
        file_path = os.path.join(train_dir, file_name)  # Construct the full file path
        if not os.path.exists(file_path):  # Check if the file already exists
            print(f"Training file {file_name} not found. Downloading...")
            download_file(url, file_path)  # Download the training file
        else:
            print(f"Training file {file_name} already present. Skipping download.")

def main() -> None:
    """
    Main function to parse arguments and orchestrate the dataset download process.
    """
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Download PILE dataset.")  # Description of the script
    parser.add_argument('--train_max', type=int, default=1, help="Max number of training files to download.")  # Max training files
    parser.add_argument('--train_dir', default="data/train", help="Directory for storing training data.")  # Training directory
    parser.add_argument('--val_dir', default="data/val", help="Directory for storing validation data.")  # Validation directory

    args = parser.parse_args()  # Parse the arguments provided by the user

    # Ensure directories for training and validation data exist
    os.makedirs(args.train_dir, exist_ok=True)  # Create training directory if it doesn't exist
    os.makedirs(args.val_dir, exist_ok=True)  # Create validation directory if it doesn't exist

    # Start downloading the dataset
    download_dataset(VAL_URL, TRAIN_URLS, args.val_dir, args.train_dir, args.train_max)

    print("Dataset downloaded successfully.")  # Indicate successful download

if __name__ == "__main__":
    # Entry point of the script
    main()
