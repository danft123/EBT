## EnergyTransformerDecoder

This is a decoder transformer that receives a context (hidden_states) and a candidate (input_injection) and computes an energy score for the pair.
Here hidden_states has shape [B, S, H] and input_injection has shape [B, 1, H].