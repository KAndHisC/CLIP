import poptorch
from torch import nn
from .model_ipu import *
from .ipu_options import get_options





class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        if hasattr(self.layer, 'weight'):
            self.weight = layer.weight

    def forward(self, *args, **kwargs):
        return poptorch.recomputationCheckpoint(self.layer(*args, **kwargs)) 


def recomputationCheckpointWrapper(layer):
    def recomputationLayer(*args, **kwargs):
        return poptorch.recomputationCheckpoint(layer(*args, **kwargs))
    return recomputationLayer

