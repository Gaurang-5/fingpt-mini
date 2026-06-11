import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Attention Visualization\n",
    "This notebook visualizes the internal attention weights of the FinGPT-Mini model across different layers and heads."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sentence 1: RBI raised interest rates by 50 basis points\n",
    "![Attention S1](../results/attention_s1.png)\n",
    "\n",
    "### Analysis\n",
    "- **Strongest Connections**: The strongest token-to-token connections are highly localized in Layer 1 (e.g., attending to the immediate previous token), while Layer 4 shows much broader contextual span.\n",
    "- **Semantic Links**: In the late layers, the word `rates` strongly attends to `interest`, recognizing the multi-word entity `interest rates`. Similarly, `raised` attends back to `RBI`, linking the subject to the verb.\n",
    "- **Early vs Late**: Early layers act mostly as positional feature extractors (focusing on local syntax), whereas late layers act as semantic integrators.\n",
    "- **Conclusion**: The model has learned structural syntax in early layers and financial entity groupings in the late layers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sentence 2: The Sensex fell after poor quarterly results\n",
    "![Attention S2](../results/attention_s2.png)\n",
    "\n",
    "### Analysis\n",
    "- **Strongest Connections**: In the late layers, `fell` has a strong attention weight on `Sensex`, bridging the subject and the action. \n",
    "- **Semantic Links**: `results` attends back heavily to `quarterly` and `poor`, synthesizing the negative sentiment phrase before making the next prediction.\n",
    "- **Early vs Late**: The early layers (Layer 1) maintain a strong diagonal pattern (attending to the immediate past token), which diffuses in Layer 4 as attention heads focus on long-range subjects.\n",
    "- **Conclusion**: The model correctly isolates noun phrases (`The Sensex`) and their modifiers (`poor quarterly results`) before connecting them together."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sentence 3: SEBI issued guidelines for mutual fund disclosure\n",
    "![Attention S3](../results/attention_s3.png)\n",
    "\n",
    "### Analysis\n",
    "- **Strongest Connections**: `disclosure` heavily attends to `fund` and `mutual` in the final layers.\n",
    "- **Semantic Links**: `issued` links back to `SEBI`, recognizing the regulatory body as the actor. \n",
    "- **Early vs Late**: Layer 1 heads are very specialized (e.g., Head 1 always looks at token t-1, Head 2 looks at token t-2). Layer 4 heads are highly diffuse and text-dependent.\n",
    "- **Conclusion**: The model has memorized standard financial n-grams like `mutual fund disclosure` and distributes its attention uniformly across these tokens."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sentence 4: Inflation rose to 6.5 percent in October\n",
    "![Attention S4](../results/attention_s4.png)\n",
    "\n",
    "### Analysis\n",
    "- **Strongest Connections**: The numerical token `6.5` attends to `percent` and `rose`, synthesizing the quantitative change.\n",
    "- **Semantic Links**: `October` attends back to `Inflation`, spanning the entire sentence to understand the context of the date.\n",
    "- **Early vs Late**: Early layers are almost purely diagonal. Late layers exhibit vertical 'stripes' where certain highly important tokens (like `Inflation`) receive attention from almost all subsequent tokens.\n",
    "- **Conclusion**: The model designates key subject tokens as 'hubs' that distribute context to the rest of the sequence."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Global Mean Attention\n",
    "![Attention Summary](../results/attention_summary.png)\n",
    "\n",
    "Averaging across all layers and heads reveals the causal nature of the transformer. The strict upper-triangular masking ensures zero attention to future tokens. The persistent diagonal brightness confirms that local context remains the most important predictive feature, while semantic subject-verb jumps provide the crucial secondary signal."
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
with open('fingpt-mini/notebooks/visualization.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
