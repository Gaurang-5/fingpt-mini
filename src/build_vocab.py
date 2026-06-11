import argparse
import logging
import json
import os
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_vocabularies(corpus_path: str, char_out_path: str, word_out_path: str):
    if not os.path.exists(corpus_path):
        logging.error(f"Corpus file not found: {corpus_path}")
        return
        
    logging.info(f"Reading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    special_tokens = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    
    # Character-level vocab
    logging.info("Building character-level vocabulary...")
    unique_chars = sorted(list(set(text)))
    char_vocab = special_tokens.copy()
    idx = len(special_tokens)
    for c in unique_chars:
        if c not in char_vocab:
            char_vocab[c] = idx
            idx += 1
            
    # Word-level vocab
    logging.info("Building word-level vocabulary...")
    words = text.split()
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(15000)]
    
    word_vocab = special_tokens.copy()
    idx = len(special_tokens)
    for w in top_words:
        if w not in word_vocab:
            word_vocab[w] = idx
            idx += 1
            
    # Save vocabularies
    os.makedirs(os.path.dirname(os.path.abspath(char_out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(word_out_path)), exist_ok=True)
    
    with open(char_out_path, 'w', encoding='utf-8') as f:
        json.dump(char_vocab, f, ensure_ascii=False, indent=2)
    logging.info(f"Character vocabulary saved to {char_out_path} (size: {len(char_vocab)})")
    
    with open(word_out_path, 'w', encoding='utf-8') as f:
        json.dump(word_vocab, f, ensure_ascii=False, indent=2)
    logging.info(f"Word vocabulary saved to {word_out_path} (size: {len(word_vocab)})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build character and word vocabularies")
    parser.add_argument('--input', required=True, help="Path to cleaned corpus")
    parser.add_argument('--char_out', default='data/char_vocab.json', help="Output path for char vocab")
    parser.add_argument('--word_out', default='data/word_vocab.json', help="Output path for word vocab")
    args = parser.parse_args()
    
    build_vocabularies(args.input, args.char_out, args.word_out)
