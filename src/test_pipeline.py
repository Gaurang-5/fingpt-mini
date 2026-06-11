import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test(corpus_path: str, char_vocab_path: str, word_vocab_path: str):
    logging.info("--- Final Test ---")
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
            logging.info(f"First 500 characters of the cleaned corpus:\n{'-'*50}\n{text[:500]}\n{'-'*50}")
    except FileNotFoundError:
        logging.error(f"Cleaned corpus not found at {corpus_path}")

    try:
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            char_vocab = json.load(f)
            logging.info(f"Character vocabulary size: {len(char_vocab)}")
    except FileNotFoundError:
        logging.error(f"Character vocabulary not found at {char_vocab_path}")

    try:
        with open(word_vocab_path, 'r', encoding='utf-8') as f:
            word_vocab = json.load(f)
            logging.info(f"Word vocabulary size: {len(word_vocab)}")
    except FileNotFoundError:
        logging.error(f"Word vocabulary not found at {word_vocab_path}")

if __name__ == '__main__':
    run_test(
        corpus_path='data/cleaned_corpus.txt',
        char_vocab_path='data/char_vocab.json',
        word_vocab_path='data/word_vocab.json'
    )
