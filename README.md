# FinGPT-Mini 📉🧠

FinGPT-Mini is a custom-built, miniature generative language model trained from scratch on Indian financial text (news, RBI circulars, SEBI guidelines, and financial reports). It is built entirely in PyTorch and features a GPT-style autoregressive transformer architecture with several modern architectural improvements.

## 🚀 Key Features
- **Custom Transformer Architecture**: Implements a Pre-Norm Transformer with GELU activations.
- **Causal Self-Attention**: Hand-written multi-head causal self-attention mechanism, featuring upper-triangular look-ahead masking.
- **Weight Tying**: Ties the token embedding matrix with the final `lm_head` projection layer, reducing total parameter count by ~20% and improving regularization.
- **Tokenization**: Supports both Character-level and Word-level tokenization strategies depending on the experiment configuration.
- **Interpretability Hooks**: Uses PyTorch `register_forward_hook` to extract hidden attention matrices from the multi-head attention blocks, allowing for visual analysis of semantic learned structures.

## 📊 Interpretability & Attention Visualizations

One of the standout features of this repository is the ability to visualize the model's internal attention mechanisms to ensure it is actually learning financial syntax rather than just memorizing sequences. 

The heatmaps below show the global mean attention across all layers and heads for financial sentences. You can see how the model correctly learns to prevent future-token-peeking (the blank upper triangle) while forming strong localized diagonal patterns to predict the next word, supplemented by long-range vertical stripes (e.g., verbs attending back to subjects).

*(See `notebooks/visualization.ipynb` for full layer-by-layer breakdowns)*

![Mean Attention Heatmaps](results/attention_summary.png)

## 💻 Usage

### 1. Training the Model
To train the model on the financial corpus using the character-level tokenizer:
```bash
python src/train.py \
    --corpus data/cleaned_corpus.txt \
    --char-vocab data/char_vocab.json \
    --output-dir checkpoints \
    --epochs 10
```
*Note: The dataset uses a sliding window of stride=1 to maximize training pairs. Training on an M-series Mac takes ~8 hours. For faster training, use Google Colab or an AWS GPU instance.*

### 2. Generating Text
Once trained, you can use the interactive generation script with adjustable temperature and Top-k sampling:
```bash
python src/generate.py
```

### 3. Visualizing Attention
To extract the attention weights and generate heatmaps from your trained checkpoint:
```bash
python src/visualize_attention.py
```

## 🛠️ Project Structure
- `src/model.py`: Core Transformer implementation
- `src/attention.py`: Causal Self-Attention block
- `src/tokenizer.py`: Character & Word Tokenizer logic
- `src/dataset.py`: PyTorch Dataloader and sliding window logic
- `src/train.py`: Main training loop with cosine learning rate decay
- `src/generate.py`: Autoregressive generation script with sampling
- `src/visualize_attention.py`: Hooks and Seaborn heatmap generation
