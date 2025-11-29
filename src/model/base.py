from torch.nn import Module
from torch import Tensor
from typing import Any


class BaseModel(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: dict[str,Any]) -> Tensor:
        raise NotImplementedError