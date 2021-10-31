import poptorch
from torch import nn
from torch.utils.data import Dataset
from .model_ipu import *


class SyntheticData(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image = image
        # self.texts = None
        # self.preprocess = None
        self.length = 1000000

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.image


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

