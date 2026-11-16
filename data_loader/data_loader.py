from collections.abc import Iterator

import h5py
import numpy as np
import torch


def get_batch_iterator(
    data_path: str,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
    *,
    seed: int | None = None,
    skip_batches: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates an iterator for generating batches of data from an HDF5 file.

    The file is cut into windows of ``context_length + 1`` tokens, the windows are shuffled,
    and each batch takes the next ``batch_size`` of them, reshuffling when an epoch runs out.

    Args:
        data_path (str): Path to the HDF5 file containing tokenized data.
        batch_size (int): Number of sequences in each batch.
        context_length (int): Length of each sequence.
        device (str, optional): Device to load the data onto ('cpu' or 'cuda'). Defaults to "cpu".
        seed (int, optional): Seed for the shuffle. The order then depends only on this seed,
            not on anything else that draws from numpy's global random state. Without it the
            seed is drawn from numpy's global state, so ``np.random.seed`` still makes a run
            repeatable.
        skip_batches (int, optional): Batches to skip before the first one is yielded, without
            reading them. A resumed run passes the number of batches it already trained on,
            together with the same seed, so it continues the data order instead of starting it
            again. Defaults to 0.
        rank (int, optional): This process's rank under DistributedDataParallel. Defaults to 0.
        world_size (int, optional): Number of processes. All ranks share one shuffle (so they
            must pass the same seed) and each takes its own slice of every global batch, so no
            two ranks see the same window in an epoch. Defaults to 1.

    Yields:
        tuple: A tuple containing input sequences (xb) and target sequences (yb).
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    if skip_batches < 0:
        raise ValueError(f"skip_batches must be 0 or more, got {skip_batches}")
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    global_batch = batch_size * world_size

    # Open the HDF5 file in read mode
    with h5py.File(data_path, 'r') as hdf5_file:

        # Extract the dataset of tokenized sequences
        dataset = hdf5_file['tokens']

        # Get the total size of the dataset
        dataset_size = dataset.shape[0]

        # Calculate the number of examples (sequences) that can be made from the data
        n_examples = (dataset_size - 1) // context_length

        # Create an array of indices for examples and shuffle them for randomness
        example_idxs = np.arange(n_examples)
        rng.shuffle(example_idxs)

        # Initialize epoch counter and example counter
        epochs = 0
        counter = 0
        to_skip = skip_batches

        while True:
            # Check if the current batch exceeds the number of available examples
            if counter + global_batch > n_examples and counter > 0:
                # Shuffle the indices again and reset the counter to 0
                rng.shuffle(example_idxs)
                counter = 0
                if to_skip == 0:
                    print(f"Finished epoch {epochs}")  # Print epoch number when an epoch finishes
                epochs += 1  # Increment the epoch counter

            if to_skip > 0:
                # Move past a batch the resumed run already trained on, without reading it.
                counter += global_batch
                to_skip -= 1
                continue

            # This rank's slice of the next global batch of window starts
            start = counter + rank * batch_size
            random_indices = example_idxs[start:start + batch_size] * context_length

            # Retrieve sequences from the dataset based on the random indices
            random_samples = torch.tensor(np.array([dataset[idx:idx+context_length+1] for idx in random_indices]))

            # Separate the input sequences (xb) and target sequences (yb)
            xb = random_samples[:, :context_length].to(device)  # Input sequence (first half of the random sample)
            yb = random_samples[:, 1:context_length+1].to(device)  # Target sequence (second half of the random sample)

            # Increment the counter to move to the next batch
            counter += global_batch

            # Yield the input and target sequences as a tuple for the current batch
            yield xb, yb

if __name__ == '__main__':
    # Example Usage (requires a dummy HDF5 file for testing)
    # Create a dummy HDF5 file
    import os
    dummy_data_path = "dummy_data.h5"
    if not os.path.exists(dummy_data_path):
        with h5py.File(dummy_data_path, 'w') as f:
            f.create_dataset('tokens', data=np.arange(1000))

    batch_size = 4
    context_length = 10
    for xb, yb in get_batch_iterator(dummy_data_path, batch_size, context_length):
        print("Input Batch Shape:", xb.shape)
        print("Target Batch Shape:", yb.shape)
        break
