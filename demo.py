import sys, os
import torch

# Ensure we can import from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from generate import generate, GenerationConfig
from model import MiniGPT, ModelConfig
from tokenizer import Tokenizer

# Detect device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load Tokenizer
vocab_path = os.path.join(current_dir, 'data', 'char_vocab.json')
tokenizer = Tokenizer(vocab_path, mode='char')
print("Loading latest checkpoint...")

import glob
ckpts = glob.glob(os.path.join(current_dir, 'checkpoints', '*.pt'))
if not ckpts:
    print("No checkpoints found!")
    sys.exit(1)
    
latest_ckpt = max(ckpts, key=os.path.getmtime)
print(f"Loaded: {os.path.basename(latest_ckpt)}")

checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
model_config = checkpoint.get('config', ModelConfig(vocab_size=tokenizer.vocab_size))
model = MiniGPT(model_config).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

while True:
    try:
        prompt = input("\n[Enter Prompt] (or type 'exit' to quit): ")
        if prompt.lower() in ['exit', 'quit']:
            break
        
        print("\n[GENERATING...]")
        gen_config = GenerationConfig(max_new_tokens=100, temperature=0.7, top_k=50, device=device)
        output = generate(model, tokenizer, prompt, gen_config)
        print(f"[GENERATED] {output}\n")
    except KeyboardInterrupt:
        break
