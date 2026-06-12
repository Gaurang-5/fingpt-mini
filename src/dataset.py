import os
import torch
from torch.utils.data import Dataset, DataLoader
import logging

try:
    from tokenizer import Tokenizer
except ImportError:
    from .tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialTextDataset(Dataset):
    def __init__(self, corpus_path: str = None, tokenizer: Tokenizer = None, context_length: int = 128, _text_data: str = None):
        self.context_length = context_length
        self.tokenizer = tokenizer
        
        if _text_data is not None:
            text = _text_data
        else:
            if corpus_path is None:
                raise ValueError("Must provide either corpus_path or _text_data")
            logging.info(f"Loading corpus from {corpus_path}...")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
        logging.info("Tokenizing corpus ONCE...")
        tokens = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        logging.info(f"Dataset created with {len(self.tokens)} total tokens.")

    def __getitem__(self, idx):
        # input_ids  = tokens[idx : idx + context_length]
        # target_ids = tokens[idx+1 : idx + context_length + 1]
        # The model learns to predict target given input — this IS language modeling
        input_ids = self.tokens[idx * self.context_length : (idx * self.context_length) + self.context_length]
        target_ids = self.tokens[(idx * self.context_length) + 1 : (idx * self.context_length) + self.context_length + 1]
        return input_ids, target_ids

    def __len__(self):
        # Subtract context_length to ensure we don't go out of bounds for the target_ids (idx + context_length + 1)
        return max(0, (len(self.tokens) - self.context_length) // self.context_length)

def create_dataloaders(corpus_path: str, tokenizer: Tokenizer, context_length: int, batch_size: int,
                       val_fraction: float = 0.1, num_workers: int = 2):
    
    logging.info(f"Splitting CORPUS from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Split CORPUS (not windows) at val_fraction for no data leakage
    split_idx = int(len(text) * (1 - val_fraction))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    logging.info("Creating train dataset...")
    train_dataset = FinancialTextDataset(_text_data=train_text, tokenizer=tokenizer, context_length=context_length)
    
    logging.info("Creating val dataset...")
    val_dataset = FinancialTextDataset(_text_data=val_text, tokenizer=tokenizer, context_length=context_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=num_workers)
                            
    return train_loader, val_loader

if __name__ == '__main__':
    # Test block
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vocab_path = os.path.join(base_dir, 'data', 'char_vocab.json')
    corpus_path = os.path.join(base_dir, 'data', 'cleaned_corpus.txt')
    
    # Create dummy data if it doesn't exist to allow the test to run out of the box
    if not os.path.exists(vocab_path) or not os.path.exists(corpus_path):
        logging.warning("Vocab or corpus missing. Creating dummy data for test...")
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(corpus_path, 'w') as f:
            f.write("this is a dummy financial corpus used for testing the dataset sliding window. it must be long enough to generate multiple windows.")
        with open(vocab_path, 'w') as f:
            import json
            chars = list(set("this is a dummy financial corpus used for testing the dataset sliding window. it must be long enough to generate multiple windows."))
            vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
            for i, c in enumerate(chars, start=4):
                vocab[c] = i
            json.dump(vocab, f)
            
    tokenizer = Tokenizer(vocab_path, mode='char')
    context_length = 16
    
    logging.info("--- Initializing Test Dataset ---")
    dataset = FinancialTextDataset(corpus_path=corpus_path, tokenizer=tokenizer, context_length=context_length)
    
    logging.info(f"Dataset size (number of windows): {len(dataset)}")
    logging.info(f"Vocab size: {tokenizer.vocab_size}")
    
    # Print 3 sample (input, target) pairs decoded to text
    for i in range(3):
        input_ids, target_ids = dataset[i]
        
        input_text = tokenizer.decode(input_ids.tolist())
        target_text = tokenizer.decode(target_ids.tolist())
        
        logging.info(f"\n--- Sample {i} ---")
        logging.info(f"Input text:  {repr(input_text)}")
        logging.info(f"Target text: {repr(target_text)}")
        
        # Confirm target is always input shifted by exactly 1 token
        is_shifted = (input_ids[1:].tolist() == target_ids[:-1].tolist())
        logging.info(f"Is shifted by 1: {is_shifted}")

    # Print a batch shape
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_in, batch_target in loader:
        logging.info(f"\nBatch input shape: {batch_in.shape}")
        logging.info(f"Batch target shape: {batch_target.shape}")
        break
