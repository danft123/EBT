from lightning import LightningModule
from typing import Any
from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra
import torch
from torch import Tensor
from torch.nn import Embedding

class EnergyTransformerDecoder(LightningModule):
    def __init__(self, model: dict[str,Any], candidate_step_size: float, candidate_sigma: float, tokenizer: dict[str,Any], *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = instantiate(model)
        self.candidate_step_size = candidate_step_size
        self.candidate_sigma = candidate_sigma
        self.tokenizer = instantiate(tokenizer)
    def forward(self, x: dict[str,Any]) -> Any:
        """
        Forward pass of the EnergyTransformer module.
        The initial candidate is a random tensor from a uniform or normal distribution.
        The forward then iterates over the energy model to refine the candidate.
        The refinement is done using Langevin dynamics or another MCMC method.
        """
    
    def next_candidate(self, current_candidate: Tensor, energy: Tensor) -> Tensor:
        """
        Given the current candidate and the energy, produce the next candidate.
        This can be done using Gradient Descent and Langevin dynamics: next_candidate = current_candidate - step_size * grad_energy + noise where noise is sampled from N(0, sigma)
        Args:
        current_candidate: The current candidate tensor of shape (bsz, 1, vocab_size)
        energy: The energy resulted from context and current_candidate of shape (bsz)
        Returns:
        next_candidate: The next candidate tensor of shape (bsz, 1, vocab_size)
        """
        
        # This is incorrect, instead of hidden_dim we should use vocab_size #TODO fix later
        # noise = torch.randn_like(current_candidate) * self.candidate_sigma
        # next_candidate = current_candidate - self.candidate_step_size * torch.autograd.grad(energy.sum(), current_candidate)[0]
        # return next_candidate + noise
        

    def training_step(self, batch: dict[str,Any], batch_idx: int) -> Tensor:
        """
        Training step of the EnergyTransformer module.
        """
        context_tokens = batch["context_tokens"]
        num_steps = batch["num_steps"]
        candidate = torch.randn(context_tokens.shape[0], 1, self.tokenizer.vocab_size) # TODO fix later
        for step in range(num_steps):
            ...
            
    



        

@hydra.main(config_path="../../conf/module/nlp/AR-EBT", config_name="num_layers=2_rank=1", version_base=None)
def main(cfg: DictConfig):
    """
    Script to test/debug the EnergyTransformer module. See README.md in Debug section for how to execute this script directly.
    """
    module = instantiate(cfg)
    print(module)

if __name__ == "__main__":
    main()