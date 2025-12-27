## EnergyTransformerDecoder

This is a decoder transformer that receives a context (hidden_states) and a candidate (input_injection) and computes an energy score for the pair. Here hidden_states has shape [B, S, H] and input_injection has shape [B, 1, H].

The current algorithm concatenates the context and candidate to form a tensor [B, S+1, H] and feeds it to the transformer. By choosing a rank k, the transformer sends its output to the energy head, which is a linear layer that projects the output to a tensor [B, S+1, k] so that at each timestep we have an associated energy vector of size k (which we call latent energy).

The latent energy is then sent to a tensor [B, S+1] by summing the squared values of each coordinate from latent energy. That is:
E = x^T B^T B x = ||Bx||^2.

At this step we have an energy scalar associated with each timestep. The final energy is defined as the sum of each energy for each timestep.