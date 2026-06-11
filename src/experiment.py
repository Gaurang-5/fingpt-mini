import os
import json
import time
import torch
import logging
from dataclasses import dataclass
from typing import List

try:
    from model import MiniGPT, ModelConfig
    from tokenizer import Tokenizer
    from dataset import create_dataloaders
    from train import get_optimizer, get_lr
except ImportError:
    from .model import MiniGPT, ModelConfig
    from .tokenizer import Tokenizer
    from .dataset import create_dataloaders
    from .train import get_optimizer, get_lr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ExperimentConfig:
    name: str
    group: str
    num_layers: int
    vocab_mode: str
    max_vocab_size: int
    context_length: int
    
class ExperimentRunner:
    def __init__(self, configs: List[ExperimentConfig], corpus_path: str, char_vocab_path: str, word_vocab_path: str, results_path: str):
        self.configs = configs
        self.corpus_path = corpus_path
        self.char_vocab_path = char_vocab_path
        self.word_vocab_path = word_vocab_path
        self.results_path = results_path
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        
    def _load_results(self):
        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_result(self, config_name, result):
        results = self._load_results()
        results[config_name] = result
        os.makedirs(os.path.dirname(os.path.abspath(self.results_path)), exist_ok=True)
        with open(self.results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def _get_peak_memory_mb(self):
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif self.device == 'mps':
            if hasattr(torch.mps, 'current_allocated_memory'):
                return torch.mps.current_allocated_memory() / (1024 * 1024)
        return 0.0
        
    def _train_experiment(self, config: ExperimentConfig):
        vocab_path = self.char_vocab_path if config.vocab_mode == 'char' else self.word_vocab_path
        tokenizer = Tokenizer(vocab_path, mode=config.vocab_mode, max_vocab_size=config.max_vocab_size)
        
        batch_size = 32
        train_loader, val_loader = create_dataloaders(
            self.corpus_path, tokenizer, config.context_length, batch_size, val_fraction=0.1
        )
        
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=config.num_layers,
            context_length=config.context_length
        )
        model = MiniGPT(model_config).to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 3 Epochs
        epochs = 3
        max_lr = 3e-4
        warmup_steps = 100
        max_steps = epochs * len(train_loader)
        
        optimizer = get_optimizer(model, lr=max_lr)
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        
        global_step = 0
        final_val_loss = float('inf')
        val_history = []
        
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                lr = get_lr(global_step, max_lr, 1e-5, warmup_steps, max_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                    
                optimizer.zero_grad()
                logits, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                global_step += 1
                
            # Validation at end of epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(self.device), vy.to(self.device)
                    _, vloss = model(vx, vy)
                    val_loss += vloss.item()
            val_loss /= len(val_loader)
            final_val_loss = val_loss
            val_history.append(torch.exp(torch.tensor(val_loss)).item())
            
        train_time_mins = (time.time() - start_time) / 60.0
        val_perplexity = torch.exp(torch.tensor(final_val_loss)).item()
        peak_memory_mb = self._get_peak_memory_mb()
        
        return {
            "group": config.group,
            "num_layers": config.num_layers,
            "vocab_mode": config.vocab_mode,
            "max_vocab_size": config.max_vocab_size,
            "context_length": config.context_length,
            "final_val_perplexity": val_perplexity,
            "val_perplexity_history": val_history,
            "train_time_mins": train_time_mins,
            "param_count": param_count,
            "peak_memory_mb": peak_memory_mb
        }

    def run(self):
        results = self._load_results()
        for config in self.configs:
            if config.name in results:
                logging.info(f"Skipping {config.name}, already completed.")
                continue
                
            logging.info(f"Starting experiment {config.name}...")
            res = self._train_experiment(config)
            self._save_result(config.name, res)
            logging.info(f"Completed {config.name}. Val PPL: {res['final_val_perplexity']:.2f}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base_dir, 'data', 'cleaned_corpus.txt')
    char_vocab_path = os.path.join(base_dir, 'data', 'char_vocab.json')
    word_vocab_path = os.path.join(base_dir, 'data', 'word_vocab.json')
    results_path = os.path.join(base_dir, 'results', 'experiments.json')
    
    configs = [
        # EXPERIMENT A
        ExperimentConfig("A1", "A", 2, "word", None, 128),
        ExperimentConfig("A2", "A", 4, "word", None, 128),
        ExperimentConfig("A3", "A", 6, "word", None, 128),
        # EXPERIMENT B
        ExperimentConfig("B1", "B", 4, "char", None, 128),
        ExperimentConfig("B2", "B", 4, "word", None, 128),
        ExperimentConfig("B3", "B", 4, "word", 5000, 128),
        # EXPERIMENT C
        ExperimentConfig("C1", "C", 4, "word", None, 64),
        ExperimentConfig("C2", "C", 4, "word", None, 128),
        ExperimentConfig("C3", "C", 4, "word", None, 256),
    ]
    
    runner = ExperimentRunner(configs, corpus_path, char_vocab_path, word_vocab_path, results_path)
    runner.run()
