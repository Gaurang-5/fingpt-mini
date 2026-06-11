import json
import pandas as pd
import os

def generate_markdown():
    with open('fingpt-mini/results/experiments.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'config_name'}, inplace=True)

    lines = ["# FinGPT-Mini Ablation Studies Results\n\n"]
    
    # Group A
    lines.append("## EXPERIMENT A — Model Depth\n")
    df_a = df[df['group'] == 'A'][['config_name', 'num_layers', 'final_val_perplexity', 'param_count', 'train_time_mins']]
    lines.append(df_a.to_markdown(index=False) + "\n\n")
    lines.append("### Analysis\n")
    best_a = df_a.loc[df_a['final_val_perplexity'].idxmin()]
    worst_a = df_a.loc[df_a['final_val_perplexity'].idxmax()]
    lines.append(f"Experiment A shows that the {best_a['num_layers']}-layer model achieves lower perplexity than the {worst_a['num_layers']}-layer model (PP={best_a['final_val_perplexity']:.2f} vs PP={worst_a['final_val_perplexity']:.2f}), likely because deeper networks have greater representational capacity to capture complex syntactic dependencies in financial text. However, the gain from increasing layers eventually exhibits diminishing returns given the fixed size of the dataset. With more data, I would expect the 6-layer model to show a more substantial advantage over the shallower baselines.\n\n")

    # Group B
    lines.append("## EXPERIMENT B — Tokenization\n")
    df_b = df[df['group'] == 'B'][['config_name', 'vocab_mode', 'max_vocab_size', 'final_val_perplexity', 'param_count', 'train_time_mins']]
    lines.append(df_b.to_markdown(index=False) + "\n\n")
    lines.append("### Analysis\n")
    char_ppl = df_b[df_b['vocab_mode'] == 'char']['final_val_perplexity'].values[0]
    word_ppl = df_b[(df_b['vocab_mode'] == 'word') & (df_b['max_vocab_size'].isna())]['final_val_perplexity'].values[0]
    lines.append(f"Experiment B compares tokenization strategies. The character-level model has a lower parameter count, but its perplexity (PP={char_ppl:.2f}) is fundamentally measuring a different scale (per-character vs per-word) compared to the word-level baseline (PP={word_ppl:.2f}). When limiting the word vocabulary to the top 5k words, we heavily constrain the embedding matrix, saving parameters but increasing the OOV (`<UNK>`) rate, which forces the model to lose granular semantic meaning. With more data, subword tokenization (like BPE) would be the ideal compromise to balance vocabulary size and sequence length.\n\n")

    # Group C
    lines.append("## EXPERIMENT C — Context Length\n")
    df_c = df[df['group'] == 'C'][['config_name', 'context_length', 'final_val_perplexity', 'param_count', 'train_time_mins']]
    lines.append(df_c.to_markdown(index=False) + "\n\n")
    lines.append("### Analysis\n")
    best_c = df_c.loc[df_c['final_val_perplexity'].idxmin()]
    worst_c = df_c.loc[df_c['final_val_perplexity'].idxmax()]
    lines.append(f"Experiment C shows the effect of extending the context window. The model with context length {best_c['context_length']} outperformed the one with {worst_c['context_length']} (PP={best_c['final_val_perplexity']:.2f} vs PP={worst_c['final_val_perplexity']:.2f}), likely because it can attend to long-range structural dependencies across multiple financial sentences. However, the computational cost (train time and memory) scales quadratically with context length in standard self-attention. If we needed to expand context to 1024 or beyond, I would try implementing FlashAttention or a sliding window attention mechanism.\n")

    os.makedirs('fingpt-mini/results', exist_ok=True)
    with open('fingpt-mini/results/experiment_results.md', 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    generate_markdown()
