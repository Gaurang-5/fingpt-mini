import math
import time
import os
import argparse
import csv
import torch
import matplotlib.pyplot as plt

try:
    from tokenizer import Tokenizer
    from dataset import create_dataloaders
    from model import ModelConfig, MiniGPT
except ImportError:
    from .tokenizer import Tokenizer
    from .dataset import create_dataloaders
    from .model import ModelConfig, MiniGPT

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    # Linear warmup: for step < warmup_steps, lr = max_lr * (step / warmup_steps)
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    # Cosine decay: for step >= warmup_steps
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    return max(min_lr, lr)

def get_optimizer(model, lr, weight_decay=0.1):
    # Separate param groups — weight decay only on 2D+ tensors (weight matrices)
    # NO weight decay on: bias, LayerNorm weights, embeddings
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad: continue
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, torch.nn.Linear):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (torch.nn.LayerNorm, torch.nn.Embedding)):
                no_decay.add(fpn)

    # weight tying: lm_head.weight is token_emb.weight.
    if 'lm_head.weight' in decay:
        decay.remove('lm_head.weight')
    if 'lm_head.weight' in no_decay:
        no_decay.remove('lm_head.weight')
        
    param_dict = dict(model.named_parameters())
    # filter out duplicate pointers due to weight tying
    unique_ids = set()
    decay_params = []
    for pn in sorted(list(decay)):
        p = param_dict[pn]
        if id(p) not in unique_ids:
            unique_ids.add(id(p))
            decay_params.append(p)
            
    no_decay_params = []
    for pn in sorted(list(no_decay)):
        p = param_dict[pn]
        if id(p) not in unique_ids:
            unique_ids.add(id(p))
            no_decay_params.append(p)

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    # Use AdamW, betas=(0.9, 0.95)
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    tokenizer = Tokenizer(args.char_vocab, mode='char')
    
    train_loader, val_loader = create_dataloaders(
        corpus_path=args.corpus, 
        tokenizer=tokenizer, 
        context_length=args.context_length, 
        batch_size=args.batch_size
    )
    
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, 
        context_length=args.context_length
    )
    model = MiniGPT(config).to(device)
    optimizer = get_optimizer(model, lr=args.max_lr)
    
    step = 0
    best_val_perplexity = float('inf')
    max_steps = args.epochs * len(train_loader)
    min_lr = 1e-5
    
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'val_perplexity', 'lr', 'tok_per_sec'])
        
    start_time = time.time()
    
    # LOOP STRUCTURE
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 1. Update LR manually
            lr = get_lr(step, args.max_lr, min_lr, args.warmup_steps, max_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                
            # 2. zero grad
            optimizer.zero_grad()
            
            # 3. logits, loss
            t0 = time.time()
            logits, loss = model(x, y)
            
            # 4. backward
            loss.backward()
            
            # 5. clip grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. step
            optimizer.step()
            
            t1 = time.time()
            tok_per_sec = (args.batch_size * args.context_length) / (t1 - t0)
            
            # 7. Log every 100 steps
            if step % 100 == 0:
                perplexity = math.exp(loss.item())
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | LR: {lr:.2e} | Tok/s: {tok_per_sec:.2f}")
                
            # VALIDATION (every 500 steps)
            if step > 0 and step % 500 == 0:
                model.eval()
                val_loss_sum = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        _, vloss = model(vx, vy)
                        val_loss_sum += vloss.item()
                val_loss = val_loss_sum / len(val_loader)
                val_perplexity = math.exp(val_loss)
                
                print(f"-- VAL | Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f} --")
                
                # Checkpointing
                if val_perplexity < best_val_perplexity:
                    best_val_perplexity = val_perplexity
                    ckpt_path = os.path.join(args.output_dir, f'checkpoint_step{step}.pt')
                    torch.save({
                        'step': step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'val_perplexity': val_perplexity
                    }, ckpt_path)
                    print(f"Saved best checkpoint to {ckpt_path}")
                
                # Log to CSV
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, loss.item(), val_loss, val_perplexity, lr, tok_per_sec])
                    
                model.train()
                
            step += 1

    total_time = time.time() - start_time
    print("\n--- Training Summary ---")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best Val Perplexity: {best_val_perplexity:.2f}")
    
    plot_training_curves(csv_path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def plot_training_curves(log_csv_path):
    steps = []
    train_losses = []
    val_losses = []
    val_perplexities = []
    
    if not os.path.exists(log_csv_path):
        print(f"Could not find {log_csv_path} to plot.")
        return
        
    with open(log_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            val_perplexities.append(float(row['val_perplexity']))
            
    if not steps:
        print("No validation data logged to plot yet.")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(steps, train_losses, label='Train Loss', marker='o')
    ax1.plot(steps, val_losses, label='Val Loss', marker='o')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    
    ax2.plot(steps, val_perplexities, color='red', label='Val Perplexity', marker='o')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Validation Perplexity')
    ax2.legend()
    
    plt.tight_layout()
    # Assuming running from fingpt-mini/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'results', 'training_curves.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved training curves to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='Path to cleaned corpus')
    parser.add_argument('--char-vocab', type=str, required=True, help='Path to character vocabulary')
    parser.add_argument('--context-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    train(args)
