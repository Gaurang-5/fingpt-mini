import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

try:
    from model import MiniGPT, ModelConfig
    from tokenizer import Tokenizer
except ImportError:
    from .model import MiniGPT, ModelConfig
    from .tokenizer import Tokenizer

def extract_attention_weights(model: MiniGPT, tokenizer: Tokenizer, text: str, device='cpu') -> Dict[str, torch.Tensor]:
    model.eval()
    model.to(device)
    
    token_ids = tokenizer.encode(text)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    attention_weights = {}
    hooks = []
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            # output is (out, attn_weights)
            _, attn_weights = output
            attention_weights[f'layer_{layer_idx}'] = attn_weights[0].detach().cpu()
        return hook

    try:
        for i, block in enumerate(model.blocks):
            h = block.attn.register_forward_hook(get_hook(i))
            hooks.append(h)
            
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
            
    return attention_weights

def plot_attention_heatmap(attn_matrix: np.ndarray, tokens: List[str], title: str, ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    # Standardize string representations for tokens (e.g. space -> ' ')
    tokens = [t if t.strip() else f"'{t}'" for t in tokens]
        
    sns.heatmap(attn_matrix, cmap='Blues', ax=ax, 
                xticklabels=tokens, yticklabels=tokens,
                vmin=0, vmax=attn_matrix.max(), cbar=False)
    
    ax.set_title(title, pad=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def plot_sentence_attention(attn_weights: Dict[str, torch.Tensor], tokens: List[str], save_path: str):
    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Layer 0 (Early)
    layer_early = attn_weights['layer_0'] # shape (num_heads, seq_len, seq_len)
    for head_idx in range(4):
        ax = axes[0, head_idx]
        matrix = layer_early[head_idx].numpy()
        plot_attention_heatmap(matrix, tokens, f"Layer 1, Head {head_idx+1}", ax=ax)
        
    # Row 2: Layer 3 (Late)
    layer_late = attn_weights['layer_3']
    for head_idx in range(4):
        ax = axes[1, head_idx]
        matrix = layer_late[head_idx].numpy()
        plot_attention_heatmap(matrix, tokens, f"Layer 4, Head {head_idx+1}", ax=ax)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def compute_mean_attention(attn_weights: Dict[str, torch.Tensor]) -> np.ndarray:
    all_matrices = []
    for layer_name, tensor in attn_weights.items():
        # tensor: (num_heads, seq_len, seq_len)
        all_matrices.append(tensor.numpy())
    
    # shape: (num_layers, num_heads, seq_len, seq_len)
    stacked = np.stack(all_matrices)
    return stacked.mean(axis=(0, 1))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, 'data', 'char_vocab.json')
    checkpoint_path = os.path.join(base_dir, 'checkpoints', 'checkpoint_step1000.pt')
    results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer(vocab_path, mode='char')
    
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        context_length=128
    )
    
    model = MiniGPT(config).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}. Using random weights.")
        
    sentences = [
        "RBI raised interest rates by 50 basis points",
        "The Sensex fell after poor quarterly results",
        "SEBI issued guidelines for mutual fund disclosure",
        "Inflation rose to 6.5 percent in October"
    ]
    
    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(15, 12))
    axes_summary = axes_summary.flatten()
    
    for i, text in enumerate(sentences):
        print(f"Processing S{i+1}: '{text}'")
        attn_weights = extract_attention_weights(model, tokenizer, text, device)
        
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        # 2x4 grid for this sentence
        s_save_path = os.path.join(results_dir, f'attention_s{i+1}.png')
        plot_sentence_attention(attn_weights, tokens, s_save_path)
        
        # Mean attention plot
        mean_attn = compute_mean_attention(attn_weights)
        mean_save_path = os.path.join(results_dir, f'attention_mean_s{i+1}.png')
        plot_attention_heatmap(mean_attn, tokens, f"Mean Attention — S{i+1}", save_path=mean_save_path)
        
        # Add to summary grid
        plot_attention_heatmap(mean_attn, tokens, f"S{i+1} Mean Attention", ax=axes_summary[i])
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'attention_summary.png'), bbox_inches='tight', dpi=300)
    plt.close(fig_summary)
    
    print("Done generating attention visualizations.")

if __name__ == '__main__':
    main()
