import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleLLM(nn.Module):
    """A simple implementation of a transformer-based language model for educational purposes."""
    def __init__(self, vocab_size=50257, d_model=256, num_heads=4, max_seq_len=1024):
        super().__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # Convert tokens to vectors
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))  # Add position information
        
        # Transformer components
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model)  # Separate MLP module
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)  # Pre-attention normalization
        self.ln2 = nn.LayerNorm(d_model)  # Pre-MLP normalization
        
        # Final output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # 1. Token and position embeddings
        h = self.token_embedding(x)
        h = h + self.position_embedding[:, :x.size(1)]
        
        # 2. Transformer block
        # Self-attention with residual connection
        h = h + self.attention(self.ln1(h), mask)
        # MLP with residual connection
        h = h + self.mlp(self.ln2(h))
        
        # 3. Project to vocabulary size
        return self.output_proj(h)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism that allows the model to jointly attend to information 
    from different representation subspaces at different positions."""
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Model dimensions
        self.d_model = d_model          # Total embedding dimension
        self.num_heads = num_heads      # Number of attention heads
        self.head_dim = d_model // num_heads  # Dimension per head
        
        # Separate projections for Q, K, V for clarity
        self.q_proj = nn.Linear(d_model, d_model)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model)  # Value projection
        
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project input into Q, K, V
        q = self.q_proj(x)  # Shape: (batch_size, seq_len, d_model)
        k = self.k_proj(x)  # Shape: (batch_size, seq_len, d_model)
        v = self.v_proj(x)  # Shape: (batch_size, seq_len, d_model)
        
        # 2. Reshape for multi-head attention
        # Split d_model into num_heads and head_dim, then move head dim to position 1
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute attention scores
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 4. Apply mask if provided (for causal attention)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 6. Apply attention to values
        # (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attention_weights, v)
        
        # 7. Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 8. Final output projection
        return self.out_proj(out)

class MLP(nn.Module):
    """Multi-Layer Perceptron block used in Transformer architecture.
    Consists of two linear transformations with a GELU activation in between."""
    def __init__(self, d_model=256, expansion_factor=4):
        super().__init__()
        d_ff = d_model * expansion_factor  # Hidden dimension of feed-forward network
        
        self.fc1 = nn.Linear(d_model, d_ff)    # First linear transformation
        self.activation = nn.GELU()             # Non-linear activation
        self.fc2 = nn.Linear(d_ff, d_model)     # Second linear transformation

    def forward(self, x):
        # 1. First linear layer with dimension expansion
        x = self.fc1(x)
        
        # 2. Apply GELU activation
        x = self.activation(x)
        
        # 3. Second linear layer projecting back to model dimension
        x = self.fc2(x)
        
        return x



def create_causal_mask(size):
    """Creates a causal attention mask to prevent attending to future tokens.
    Args:
        size (int): Sequence length
    Returns:
        torch.Tensor: Boolean mask where True values are positions to be masked
    """
    # Create upper triangular matrix (including diagonal)
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # False values will be attended to, True values will be masked
    return ~mask

def main():
    # Create model instance
    model = SimpleLLM()
    
    # Create sample input (batch_size=2, seq_len=16)
    x = torch.randint(0, 50257, (2, 16))
    
    # Create causal mask
    mask = create_causal_mask(16)
    
    # Forward pass
    logits = model(x, mask)
    
    # Print model statistics
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main() 