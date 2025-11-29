from typing import Any
from lightning import LightningModule
from torch import Tensor
from hydra.utils import instantiate

class BaseModule(LightningModule):
    def __init__(self, model: dict[str,Any]):
        super().__init__()
        self.save_hyperparameters()
        self.model = instantiate(**self.hparams.model)
    
    def forward(self, x: dict[str,Any]) -> Tensor:
        raise NotImplementedError
    
    def common_step(self, batch: dict[str,Any], batch_idx: int):
        x, target = batch['input'], batch['target']
        output = self.forward(x)
        loss = self.compute_loss(output, target)
        

    
    def training_step(self, batch: dict[str,Any], batch_idx: int):
        return self.common_step(batch, batch_idx)
    
    def validation_step(self, batch: dict[str,Any], batch_idx: int):
        return self.common_step(batch, batch_idx)
    
    def test_step(self, batch: dict[str,Any], batch_idx: int):
        return self.common_step(batch, batch_idx)
    
    def configure_optimizers(self):
        raise NotImplementedError
    
    