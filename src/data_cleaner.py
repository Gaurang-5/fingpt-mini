import argparse
import logging
import os
import re
import hashlib
from typing import List

try:
    import pdfplumber
    PDF_READER = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_READER = "pypdf2"
    except ImportError:
        PDF_READER = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_financial_text(text: str) -> str:
    """
    Cleans raw financial text by removing extra whitespaces and standardizing newlines.

    Parameters
    ----------
    text : str
        The raw input text to be cleaned.

    Returns
    -------
    str
        The cleaned and normalized text string.
    """
    # Lowercases text
    paragraph = text.lower()
    
    # Removes HTML tags
    paragraph = re.sub(r'<[^>]+>', ' ', paragraph)
    
    # Remove PDF artifacts, page numbers (heuristic)
    paragraph = re.sub(r'\bpage\s+\d+\b', '', paragraph)
    
    # Keep financial punctuation: % $ ₹ . , - ( )
    # Strip characters that are not a-z, 0-9, space, or the allowed punctuation
    paragraph = re.sub(r'[^a-z0-9\s%\$₹\.,\-\(\)]', ' ', paragraph)
    
    # Normalises whitespace (collapse multiple spaces/newlines)
    paragraph = re.sub(r'\s+', ' ', paragraph).strip()
    return paragraph

def has_high_non_ascii(text: str, threshold: float = 0.3) -> bool:
    if not text:
        return False
    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    return (non_ascii_count / len(text)) > threshold

def extract_from_pdf(pdf_path: str) -> str:
    """
    Extracts and returns the text content from a given PDF file.

    Parameters
    ----------
    pdf_path : str
        The absolute or relative path to the PDF file.

    Returns
    -------
    str
        The extracted text content from all pages of the PDF.
    """
    text = ""
    if PDF_READER == "pdfplumber":
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
            text = "\n".join(pages)
    elif PDF_READER == "pypdf2":
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
    else:
        logging.warning(f"Skipping {pdf_path}: No PDF reader installed.")
    return text

def extract_from_csv(csv_path: str) -> str:
    """
    Extracts relevant columns from a CSV file.

    Parameters
    ----------
    csv_path : str
        The path to the CSV file.

    Returns
    -------
    str
        The combined extracted text from specific columns.
    """
    import csv
    paragraphs = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            if len(row) >= 4:
                paragraphs.append(row[2])
                paragraphs.append(row[3])
    return "\n\n".join(paragraphs)

def process_raw_data(input_paths: List[str], output_path: str):
    """
    Processes all text, PDF, and CSV files and writes the cleaned text to an output file.

    Parameters
    ----------
    input_paths : List[str]
        The list of input file paths to process.
    output_path : str
        The path where the combined cleaned corpus will be saved.
    """
    seen_hashes = set()
    cleaned_paragraphs = []
    
    total_chars = 0
    unique_words = set()
    estimated_sentences = 0
    
    for path in input_paths:
        if not os.path.exists(path):
            logging.warning(f"Path not found: {path}")
            continue
            
        logging.info(f"Processing {path}...")
        raw_text = process_file(path)
        
        # Split into paragraphs (heuristic: double newline)
        paragraphs = re.split(r'\n\s*\n', raw_text)
        
        for p in paragraphs:
            # basic sentence estimate using '.' count
            sentence_count = p.count('.')
            if sentence_count < 3:
                continue # paragraph = 3+ sentences
            
            cleaned_p = clean_paragraph(p)
            
            if not cleaned_p:
                continue
                
            if has_high_non_ascii(cleaned_p, 0.30):
                continue
            
            # Deduplicate using sha256 hashing
            p_hash = hashlib.sha256(cleaned_p.encode('utf-8')).hexdigest()
            if p_hash in seen_hashes:
                continue
            seen_hashes.add(p_hash)
            
            cleaned_paragraphs.append(cleaned_p)
            
            # Update stats
            total_chars += len(cleaned_p)
            unique_words.update(cleaned_p.split())
            estimated_sentences += cleaned_p.count('.')
            
    final_text = "\n\n".join(cleaned_paragraphs)
    
    logging.info("--- Corpus Statistics ---")
    logging.info(f"Total characters: {total_chars}")
    logging.info(f"Unique words: {len(unique_words)}")
    logging.info(f"Estimated sentences: {estimated_sentences}")
    
    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    logging.info(f"Cleaned corpus saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean financial corpus")
    parser.add_argument('--input', nargs='+', required=True, help="Input file paths (.txt or .pdf)")
    parser.add_argument('--output', required=True, help="Output file path")
    args = parser.parse_args()
    
    clean_corpus(args.input, args.output)
