# FinGPT-Mini Data Preparation

This directory contains the vocabularies and instructions for downloading the required financial text data.

## Step 1: Download Kaggle "Indian Financial News" dataset

1. Ensure you have Kaggle CLI installed and configured.
2. Run the following command to download the dataset:
   ```bash
   kaggle datasets download -d hkapoor/indian-financial-news-articles-20032020
   unzip indian-financial-news-articles-20032020.zip
   ```

## Step 2: Download RBI Annual Reports

Download 2-3 recent Reserve Bank of India Annual Reports from their official website:
- [RBI Annual Report 2022-23 (PDF)](https://rbi.org.in/Scripts/AnnualReportPublications.aspx?Id=1353)
- [RBI Annual Report 2021-22 (PDF)](https://rbi.org.in/Scripts/AnnualReportPublications.aspx?Id=1334)
- [RBI Annual Report 2020-21 (PDF)](https://rbi.org.in/Scripts/AnnualReportPublications.aspx?Id=1305)

## Step 3: Download SEBI Annual Reports

Download 2 recent Securities and Exchange Board of India (SEBI) Annual Reports:
- [SEBI Annual Report 2022-23 (PDF)](https://www.sebi.gov.in/reports-and-statistics/publications/aug-2023/annual-report-2022-23_75283.html)
- [SEBI Annual Report 2021-22 (PDF)](https://www.sebi.gov.in/reports-and-statistics/publications/aug-2022/annual-report-2021-22_61868.html)

## Processing

Place all downloaded `.txt` and `.pdf` files in this directory (`data/`) or a subdirectory (e.g., `data/raw/`). Then you can use the scripts in `src/` to clean the corpus and build the vocabularies:

```bash
# Example
python src/data_cleaner.py --input data/raw/*.pdf data/raw/*.txt --output data/cleaned_corpus.txt
python src/build_vocab.py --input data/cleaned_corpus.txt --char_out data/char_vocab.json --word_out data/word_vocab.json
python src/test_pipeline.py
```
