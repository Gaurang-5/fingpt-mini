import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

try:
    from .attention import CausalSelfAttention
except ImportError:
    from attention import CausalSelfAttention

@dataclass
class ModelConfig:
    """Configuration for FinGPT-Mini architecture."""
    vocab_size: int = 5000       # Size of the vocabulary (override at runtime)
    d_model: int = 128           # Dimensionality of the embeddings and hidden states
    num_heads: int = 4           # Number of attention heads
    num_layers: int = 4          # Number of Transformer blocks
    ffn_dim: int = 512           # Dimensionality of the feed-forward network hidden layer
    context_length: int = 128    # Maximum sequence length (context size)
    dropout: float = 0.1         # Dropout probability

class GELU(nn.Module):
    """
    GELU activation function with tanh approximation (same as GPT-2).
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x):
        # COMMENT: GELU is smooth near x=0 unlike ReLU's hard cutoff at 0.
        # This smoothness lets gradients flow even for small negative inputs,
        # which benefits deep networks. GPT uses GELU; ReLU is for CNNs.
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForward(nn.Module):
    """
    Feed-forward network for Transformer block.
    Linear(d_model → ffn_dim) → GELU → Dropout → Linear(ffn_dim → d_model) → Dropout
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-norm architecture.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            context_length=config.context_length
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # PRE-NORM architecture (LayerNorm before sublayer, not after):
        # COMMENT: pre-norm stabilises training in deep models vs post-norm
        # x = x + Attention(LayerNorm(x))
        attn_out, _ = self.attn(self.ln_1(x))
        x = x + attn_out
        
        # x = x + FFN(LayerNorm(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    """
    The full FinGPT-Mini model architecture.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # token_emb: nn.Embedding(vocab_size, d_model)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        # pos_emb: nn.Embedding(context_length, d_model) — LEARNED, not sinusoidal
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # blocks: nn.ModuleList of num_layers TransformerBlocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        
        # ln_final: nn.LayerNorm(d_model)
        self.ln_final = nn.LayerNorm(config.d_model)
        
        # lm_head: nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # WEIGHT TYING: self.lm_head.weight = self.token_emb.weight
        self.lm_head.weight = self.token_emb.weight
        # COMMENT: weight tying halves embedding params and improves coherence
        # because the model learns consistent input/output token representations

    def forward(self, idx, targets=None):
        # idx: (batch, seq_len) of token ids
        batch, seq_len = idx.shape
        
        # pos: torch.arange(seq_len)
        pos = torch.arange(seq_len, dtype=torch.long, device=idx.device)
        
        # x = token_emb(idx) + pos_emb(pos)  → (batch, seq_len, d_model)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # pass through all blocks
        for block in self.blocks:
            x = block(x)
            
        # apply ln_final
        x = self.ln_final(x)
        
        # logits = lm_head(x)  → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        # if targets provided: loss = F.cross_entropy(...)
        loss = None
        if targets is not None:
            # Shift targets isn't necessary here since we already shift inputs/targets in the dataloader
            # The dataloader yields input_ids and target_ids of the same length
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss

if __name__ == '__main__':
    # Instantiate with default config (set vocab_size=5000)
    config = ModelConfig(vocab_size=5000)
    model = MiniGPT(config)
    
    # Forward pass: idx = torch.randint(0, 5000, (2, 128))
    idx = torch.randint(0, 5000, (2, 128))
    logits, loss = model(idx)
    
    # Print output logits shape (expect: 2, 128, 5000)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 128, 5000), f"Expected shape (2, 128, 5000), got {logits.shape}"
    
    # Print total trainable parameters (target: 1–10M)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Confirm lm_head.weight is token_emb.weight (same object)
    is_tied = (model.lm_head.weight is model.token_emb.weight)
    print(f"Weight tying confirmed (lm_head.weight is token_emb.weight): {is_tied}")
    assert is_tied, "Weight tying failed!"
