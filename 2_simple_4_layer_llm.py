import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniDeepSeek(nn.Module):
    def __init__(self, 
                 vocab_size=32000,  # Similar to DeepSeek's tokenizer
                 hidden_size=256,   # Scaled to 4096+ in real models
                 num_heads=4,       # Scaled to 32+ heads
                 mlp_ratio=4        # Standard MLP expansion ratio
                ):
        super().__init__()
        
        # 1. Embedding Layer (like DeepSeek's input processing)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # 2. Single Attention Layer (scaled-down version)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # 3. Single MLP Layer (same structure as DeepSeek's FFN)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),  # Expansion
            nn.GELU(),                                         # Activation
            nn.Linear(hidden_size * mlp_ratio, hidden_size)    # Compression
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
        # 4. Output Projection (like DeepSeek's prediction head)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # Embed tokens (input_ids -> hidden states)
        x = self.embed(x)
        
        # Attention Block (with residual)
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + attn_out)  # Pre-LN like DeepSeek
        
        # MLP Block (with residual)
        mlp_out = self.mlp(x)
        x = self.mlp_norm(x + mlp_out)
        
        return self.head(x)