import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 7: Ablation Studies Analysis\n",
    "Systematic comparison of Model Depth, Tokenization, and Context Length."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "with open('../results/experiments.json', 'r') as f:\n",
    "    data = json.load(f)\n",
    "\n",
    "df = pd.DataFrame.from_dict(data, orient='index')\n",
    "df.reset_index(inplace=True)\n",
    "df.rename(columns={'index': 'config_name'}, inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Grouped bar chart\n",
    "fig, ax = plt.subplots(figsize=(10, 6))\n",
    "\n",
    "colors = {'A': 'skyblue', 'B': 'lightgreen', 'C': 'salmon'}\n",
    "bars = ax.bar(df['config_name'], df['final_val_perplexity'], color=[colors[g] for g in df['group']])\n",
    "\n",
    "ax.set_ylabel('Validation Perplexity')\n",
    "ax.set_title('Validation Perplexity across Ablation Experiments')\n",
    "plt.xticks(rotation=45)\n",
    "\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', rotation=90)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Training curves for each group\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "groups = ['A', 'B', 'C']\n",
    "titles = ['Group A (Depth)', 'Group B (Tokenization)', 'Group C (Context)']\n",
    "\n",
    "for i, g in enumerate(groups):\n",
    "    group_df = df[df['group'] == g]\n",
    "    for _, row in group_df.iterrows():\n",
    "        axes[i].plot(range(1, len(row['val_perplexity_history'])+1), row['val_perplexity_history'], marker='o', label=row['config_name'])\n",
    "    axes[i].set_title(titles[i])\n",
    "    axes[i].set_xlabel('Epoch')\n",
    "    axes[i].set_ylabel('Val Perplexity')\n",
    "    axes[i].legend()\n",
    "    axes[i].grid(True, linestyle='--', alpha=0.6)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Results Table"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: Markdown table rendered from pandas DataFrame\n",
    "from IPython.display import Markdown, display\n",
    "cols = ['config_name', 'group', 'num_layers', 'vocab_mode', 'max_vocab_size', 'context_length', 'param_count', 'train_time_mins', 'peak_memory_mb', 'final_val_perplexity']\n",
    "display(Markdown(df[cols].to_markdown(index=False)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Honest Analysis\n",
    "\n",
    "**Experiment A (Depth):**\n",
    "[To be written]\n",
    "\n",
    "**Experiment B (Tokenization):**\n",
    "[To be written]\n",
    "\n",
    "**Experiment C (Context Length):**\n",
    "[To be written]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs('fingpt-mini/notebooks', exist_ok=True)
with open('fingpt-mini/notebooks/experiments.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
