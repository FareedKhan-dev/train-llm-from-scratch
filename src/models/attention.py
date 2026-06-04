import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    """
    A single attention head.

    This module calculates attention scores and applies them to the values.
    It includes key, query, and value projections, and uses causal masking
    to prevent attending to future tokens.

    Args:
        head_size (int): The dimensionality of the key, query, and value projections.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence, used for causal masking.
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the attention head.

        Args:
            head_size (int): The dimensionality of the key, query, and value projections.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)   # Key projection
        self.query = nn.Linear(n_embed, head_size, bias=False) # Query projection
        self.value = nn.Linear(n_embed, head_size, bias=False) # Value projection
        # Lower triangular matrix for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        scale_factor = 1 / math.sqrt(head_size)
        # Calculate attention weights: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        # Apply causal masking
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)   # (B, T, head_size)
        # Apply attention weights to values
        out = attn_weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module combines multiple attention heads in parallel. The outputs of each head
    are concatenated and passed through a final linear projection to form the output.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the multi-head attention module.

        Args:
            n_head (int): The number of parallel attention heads.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after concatenating the heads and applying the output projection.
        """
        # Concatenate the output of each head along the last dimension (C)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply final linear projection
        x = self.proj(x)
        return x

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    multihead_attn = MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = multihead_attn(input_tensor)

    print("MultiHeadAttention Input Shape:", input_tensor.shape)
    print("MultiHeadAttention Output Shape:", output_tensor.shape)
