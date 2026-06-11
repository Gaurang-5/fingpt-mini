<div align="center">
  
# 📉 FinGPT-Mini: Autoregressive Financial Language Model
**A Custom-Architected Generative Pre-Trained Transformer (GPT) Trained from Scratch**

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Architecture-Transformer-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Parameters-~800K-green?style=for-the-badge" />
</p>

</div>

---

## 📖 Abstract

**FinGPT-Mini** is a highly optimized, custom-architected generative language model designed specifically for the Indian financial domain. Built entirely from scratch in PyTorch, it features a complete end-to-end deep learning pipeline—from parsing unformatted raw PDFs and financial CSVs to an interactive autoregressive generation engine. 

Designed to demonstrate core Large Language Model (LLM) engineering principles, FinGPT-Mini features an ultra-efficient **~800K-parameter** GPT-style transformer architecture capable of rapid training on consumer hardware while successfully learning complex financial semantics, grammar, and long-range dependencies.

---

## 🧠 Architectural Deep Dive

Unlike fine-tuned wrapper models, FinGPT-Mini's core neural engine was written from scratch. It heavily borrows from the GPT-2/GPT-3 architectural philosophy with several modern training stability improvements.

### 1. Transformer Core & Pre-Norm Architecture
The model relies on a sequence of 4 Transformer decoding blocks (4 attention heads, 128 embedding dimensionality). To ensure deep gradient flow and mitigate vanishing gradients during optimization, we implemented a **Pre-Norm** layer normalization scheme:
```math
x_{l} = x_{l-1} + \text{Attention}(\text{LayerNorm}(x_{l-1}))
```
```math
x_{l} = x_{l} + \text{FFN}(\text{LayerNorm}(x_{l}))
```
*GELU (Gaussian Error Linear Unit) activations are utilized across all Feed-Forward Networks for smooth non-linearity.*

### 2. Multi-Head Causal Self-Attention
A hand-written scaled dot-product attention mechanism is utilized, featuring dynamic upper-triangular look-ahead masking. This mathematically prevents the model from "peeking" at future tokens during parallel training, forcing it to learn strictly autoregressive generation.

### 3. Latent Weight Tying
To drastically reduce the memory footprint and regularize the network, the input token embedding matrix ($E$) is mathematically tied to the final pre-softmax projection layer ($W_{out}$):
```python
self.lm_head.weight = self.token_emb.weight
```
This forces the model to learn a cohesive input-output latent representation, shedding ~20% of the total parameter count while improving semantic coherence.

---

## 📈 Hardware Optimization & Metrics

The PyTorch training loop utilizes automatic mixed precision, pinned memory tensors, and non-overlapping sliding-window dataloaders (`dataset.py`) to maximize I/O throughput to the GPU. 

When trained on an **NVIDIA T4 Cloud GPU**, the system achieves profound compute efficiency:

| Metric | Result |
| :--- | :--- |
| **Training Throughput** | > 10.8 Million tokens per second |
| **Validation Loss** | 1.04 |
| **Validation Perplexity (PPL)** | 2.84 |
| **Total Parameters** | ~813,000 |

*(Note: A validation perplexity of 2.84 is exceptionally strong for an 800K-parameter model, indicating the network has effectively internalized the syntax and terminology of the financial domain).*

---

## 📊 Interpretability & Attention Mapping

To ensure the model learned genuine syntactic relationships rather than memorizing localized sequences, PyTorch `register_forward_hook` methods were implemented to extract hidden attention matrices from the multi-head blocks.

The visualizations below confirm the model successfully learned strictly causal masking (the blank upper triangles), localized diagonal token prediction, and long-range vertical dependencies (e.g., verbs attending to historical subject tokens).

<div align="center">
  <img src="results/attention_summary.png" alt="Mean Attention Heatmaps" width="600"/>
</div>

---

## 💻 Quickstart & Pipeline Usage

The repository contains the entire pipeline required to recreate the model.

### 1. Data Pipeline
Parse raw PDFs, SEBI guidelines, and CSVs into a cleaned, tokenizable corpus:
```bash
python src/data_cleaner.py
```

### 2. GPU Training Loop
Initialize weights and train the model from scratch using Cross-Entropy tracking:
```bash
python src/train.py \
    --corpus data/cleaned_corpus.txt \
    --char-vocab data/char_vocab.json \
    --output-dir checkpoints \
    --epochs 10 \
    --batch-size 1024
```

### 3. Autoregressive Inference Engine
Interact with the trained neural network via a CLI prompt loop utilizing Temperature and Top-k sampling distributions:
```bash
python demo.py
```
> **Example Prompt:** `"The Reserve Bank of India announced today that"`  
> **Model Output:** `"an interest rates by 35 basis points... the reserve bank of india (rbi) on thursday said that up"`

---

## 🛠️ Repository Architecture

```text
fin_gpt/
├── src/
│   ├── model.py                # Core GPT Transformer topology
│   ├── attention.py            # Scaled Dot-Product Causal Attention
│   ├── dataset.py              # PyTorch Dataloader & sliding windows
│   ├── tokenizer.py            # Char/Word OOV-resistant tokenization
│   ├── data_cleaner.py         # Multi-format raw data extraction
│   ├── train.py                # AdamW optimizer & evaluation logic
│   ├── generate.py             # Top-k sampling & inference scripts
│   └── visualize_attention.py  # Forward hooks & Seaborn matrices
├── demo.py                     # Interactive CLI inference portal
└── README.md                   # System documentation
```
