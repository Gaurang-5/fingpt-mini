import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, context_length: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Single combined QKV projection: nn.Linear(d_model, 3 * d_model, bias=False)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection: nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (NOT a parameter)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('mask', mask)
        
        # COMMENT: Why -inf works vs 0
        # The attention scores are passed through a softmax function, which computes exp(x) / sum(exp(x)). 
        # exp(-inf) approaches exactly 0, so the softmax probabilities for the masked (future) positions 
        # become exactly 0. If we used 0 before softmax instead, exp(0) = 1, so the masked positions 
        # would still receive a share of the probability mass, which would violate the causal property.

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape
        
        # 1. Compute Q, K, V via the combined projection
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        
        # Reshape to (batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim)
        
        # Split into Q, K, V and permute to (batch, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 2. Attention scores: (Q @ K^T) / sqrt(head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, seq_len, seq_len)
        
        # 3. Add causal mask (slice to actual seq_len)
        scores = scores + self.mask[:seq_len, :seq_len]
        
        # 4. Softmax over last dim
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 5. Dropout on attention weights
        attn_weights_dropped = self.dropout(attn_weights)
        
        # 6. Weighted sum with V
        context = attn_weights_dropped @ v  # (batch, num_heads, seq_len, head_dim)
        
        # 7. Reshape back to (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        # 8. Output projection
        output = self.out_proj(context)
        
        return output, attn_weights

if __name__ == '__main__':
    # TARGET SPECS
    d_model = 128
    num_heads = 4
    context_length = 128
    dropout = 0.1
    
    # Instantiate with the specs above
    attention = CausalSelfAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        dropout=dropout, 
        context_length=context_length
    )
    
    # Forward pass with x = torch.randn(2, 128, 128)
    x = torch.randn(2, 128, 128)
    output, attn_weights = attention(x)
    
    # Assert output shape == (2, 128, 128)
    assert output.shape == (2, 128, 128), f"Expected (2, 128, 128), got {output.shape}"
    print(f"Output shape: {output.shape} -> SUCCESS")
    
    # Print attention weights for batch 0, head 0
    print("\nAttention weights [batch 0, head 0] (top-left 5x5):")
    print(attn_weights[0, 0, :5, :5])
    
    # Assert torch.allclose(attn_weights[0, 0].triu(1), torch.zeros(128,128), atol=1e-6)
    upper_triangle = attn_weights[0, 0].triu(1)
    zeros = torch.zeros(128, 128)
    assert torch.allclose(upper_triangle, zeros, atol=1e-6), "Causality violated: Upper triangle is not zero!"
    print("\nCausality assertion -> SUCCESS")
    
    # Print parameter count
    num_params = sum(p.numel() for p in attention.parameters() if p.requires_grad)
    print(f"\nParameter count: {num_params}")
    
    # Print the mask to show -inf upper triangle
    print("\nCausal mask (top-left 5x5):")
    print(attention.mask[:5, :5])
