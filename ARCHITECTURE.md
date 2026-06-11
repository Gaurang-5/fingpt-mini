# FinGPT-Mini Architecture Design Decisions

### Q1: Why causal masking (not bidirectional attention)?
Causal masking strictly prevents tokens from attending to future tokens (the upper right triangle of the attention matrix is set to negative infinity). This is mechanically required for autoregressive language modeling, as the model must learn to predict the next token `t+1` using only information from tokens `1` to `t`. Bidirectional attention would allow the model to "cheat" by looking at the target token during training, completely breaking the autoregressive objective.

### Q2: Why learned positional embeddings (not sinusoidal)?
Learned positional embeddings map each absolute position in the context window directly to a trainable vector, which adapts seamlessly to the specific vocabulary and dataset distribution during gradient descent. Unlike static sinusoidal embeddings which enforce a rigid mathematical frequency, learned embeddings provide the model with the exact optimal positional representations needed for the task, yielding faster convergence and better performance for smaller context windows.

### Q3: Why GELU (not ReLU)?
The Gaussian Error Linear Unit (GELU) acts as a smoothed, probabilistic version of ReLU by weighting inputs by their value in the standard normal cumulative distribution. Mechanistically, this allows for a non-zero gradient for slightly negative values, mitigating the "dead neuron" problem inherent to ReLU's hard threshold at zero. This smoothness leads to better optimization landscapes and empirically superior performance in deep transformer architectures.

### Q4: Why pre-norm (not post-norm)?
In a pre-norm architecture, Layer Normalization is applied *before* the multi-head attention and feed-forward blocks, rather than after them. This creates an uninterrupted residual pathway (skip connection) directly from the input embedding to the final layer, drastically improving gradient flow during backpropagation. This bypasses the vanishing gradient issues of deep post-norm networks and allows training without complex learning rate warmup schedules.

### Q5: Why AdamW with weight decay (not plain Adam)?
Standard Adam implements L2 regularization by adding the penalty directly to the loss, which becomes problematic because the adaptive gradient updates scale the penalty down for parameters with large historical gradients. AdamW explicitly decouples weight decay from the gradient update step, applying the decay directly to the weights. This leads to much better generalization and proper regularization without interfering with the momentum statistics.
