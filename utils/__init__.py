from .log import Logger
from .args import parse_args
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



