from .log import Logger
from .args import parse_args
from .optimization import get_lr_scheduler, get_optimizer
from .checkpoint import save_model, maybe_load_checkpoint_passing_constraints, prepare_checkpoint_metrics
from .utils import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



