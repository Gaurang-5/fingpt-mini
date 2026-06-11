import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List
import os

try:
    from model import MiniGPT, ModelConfig
    from tokenizer import Tokenizer
except ImportError:
    from .model import MiniGPT, ModelConfig
    from .tokenizer import Tokenizer

@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 50
    device: str = 'cpu'

def generate(model, tokenizer, prompt: str, config: GenerationConfig) -> str:
    model.eval()
    device = config.device
    
    # a. Encode prompt → token_ids tensor of shape (1, seq_len)
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    context_length = model.config.context_length
    
    # b. Autoregressive loop
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            # i. If seq_len > context_length, crop to last context_length tokens
            idx_cond = token_ids if token_ids.size(1) <= context_length else token_ids[:, -context_length:]
            
            # ii. Forward pass: logits = model(token_ids)
            logits, _ = model(idx_cond)
            
            # iii. Take logits at LAST position only
            logits = logits[0, -1, :]
            
            # iv. TEMPERATURE: logits = logits / temperature
            # COMMENT: dividing by T < 1 sharpens the distribution (model picks
            # its top choice more often). T > 1 flattens it (more randomness/creativity).
            # T → 0 becomes greedy argmax. T → ∞ becomes uniform random.
            logits = logits / config.temperature
            
            # v. TOP-K
            # This prevents the model from sampling very low-probability garbage tokens
            # that sneak through after temperature scaling
            v, _ = torch.topk(logits, config.top_k)
            logits[logits < v[-1]] = float('-inf')
            
            # vi. Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # vii. Sample
            next_token = torch.multinomial(probs, num_samples=1)
            
            # viii. Append next_token to token_ids
            token_ids = torch.cat((token_ids, next_token.unsqueeze(0)), dim=1)
            
    # c. Decode full sequence
    full_sequence = tokenizer.decode(token_ids[0].tolist())
    
    # d. Return only the generated portion
    return full_sequence[len(prompt):]

def batch_generate(model, tokenizer, prompts: List[str], config: GenerationConfig) -> List[str]:
    return [generate(model, tokenizer, p, config) for p in prompts]

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    vocab_path = os.path.join(base_dir, 'data', 'char_vocab.json')
    
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer(vocab_path, mode='char')
    
    # Find the latest checkpoint
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.pt'))
    if not ckpts:
        print("No checkpoints found! Using an untrained model for generation.")
        model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
        model = MiniGPT(model_config).to(device)
    else:
        # Sort by modification time to get the latest
        latest_ckpt = max(ckpts, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model_config = checkpoint.get('config', ModelConfig(vocab_size=tokenizer.vocab_size))
        model = MiniGPT(model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    
    prompts = [
        "The Reserve Bank of India today announced",
        "SEBI has issued a circular regarding",
        "Stock markets fell sharply after",
        "The Sensex rose by 500 points as",
        "Inflation in India reached",
        "The Finance Minister stated that",
        "FII outflows from Indian markets",
        "RBI Governor in his speech mentioned",
        "The rupee depreciated against the dollar because",
        "Interest rates were hiked by"
    ]
    
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'generated_samples.txt')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("FinGPT-Mini Generated Samples\n")
        f.write("==============================\n\n")
        
        for temp in temperatures:
            gen_config = GenerationConfig(max_new_tokens=200, temperature=temp, top_k=50, device=device)
            print(f"Generating at Temperature = {temp}...")
            
            for p in prompts:
                generated = generate(model, tokenizer, p, gen_config)
                full_text = p + generated
                
                output_str = f"=== Prompt: \"{p}\" | Temperature: {temp} | top_k: {gen_config.top_k} ===\n"
                output_str += full_text + "\n===\n\n"
                
                f.write(output_str)
                print(f"Done: {p[:30]}...")

    print(f"\nSaved all generated samples to {out_file}")
