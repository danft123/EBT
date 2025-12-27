# EnergyTransformerDecoder

This module implements an autoregressive energy transformer (decoder-only transformer).

## Iterative Inference

The module implements an iterative inference algorithm that allows for efficient inference.
Denote by B the batch size, S the sequence length and H the hidden dimension, denote also by V the vocabulary size.
The algorithm works as follows:

1. Initialize the hidden states with the context (hidden_states) of shape [B, S, H]
2. For the initial candidate we sample a random tensor from a uniform or normal distribution of shape [B, 1, V]
3. For each step t in 1:S do:
    1. Compute the energy of context + candidate of shape [B] (before we need to pass the candidate through the embedding so that it is of shape [B, 1, H])
    2. Compute the next candidate using gradient descent (Langevin Dynamics) using scalar energy over candidate [B,1,V]: next_candidate = current_candidate - step_size * grad_energy + noise where noise is sampled from N(0, sigma)

In the original paper the authors called the first option System II and the second option System I.
"System 1 thinking is characterized by quick, intuitive and automatic responses, relying on previous experience to solve simple or familiar problems. Alternatively, System 2 Thinking is slow, deliberate and analytical, requiring conscious effort and logical reasoning to process more complex information."
"For S2 models, we found that not detaching between steps was best, and similarly that calculating the loss only at the last step was best. For S1 models, we found the opposite to be most stable. Generally, if one is calculating the loss only at the last step, then one should not detach between steps as it’s best if the gradient propagates to previous steps in this case."