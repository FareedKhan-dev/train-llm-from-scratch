import torch
import tiktoken
import argparse
from config.config import default_config as config
from src.models.transformer import Transformer  # Assuming your Transformer class is in this module


def load_checkpoint(model_path: str, device: str):
    try:
        return torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=torch.device(device))

def generate_text(model_path: str, input_text: str, max_new_tokens: int = 100, device: str = 'cuda') -> str:
    """
    Generates text using a pre-trained Transformer model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        input_text (str): The initial text to start generation from.
        max_new_tokens (int): The maximum number of new tokens to generate.
        device (str): 'cuda' or 'cpu', the device to run the model on.

    Returns:
        str: The generated text.
    """
    # Load the model checkpoint
    checkpoint = load_checkpoint(model_path, device)

    # Initialize the model using the configuration from config.py
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)

    # Load the tokenizer
    enc = tiktoken.get_encoding("r50k_base")

    start_ids = enc.encode_ordinary(input_text)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generation process
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()

    # Decode the generated tokens
    output_text = enc.decode(generated_tokens)

    return output_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained Transformer model.")
    parser.add_argument('--model_path', type=str, help='Path to the saved model checkpoint.')
    parser.add_argument('--input_text', type=str, help='The initial text to start generation from.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')

    args = parser.parse_args()

    generated = generate_text(args.model_path, args.input_text, args.max_new_tokens, config['device'])
    print(f"Generated text:\n{generated}")

if __name__ == "__main__":
    main()
