# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_constant_schedule
from poptorch.optim import AdamW, Adam, LAMB, SGD
from torch import float16, float32
import math


from torch.optim.lr_scheduler import LambdaLR
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def get_lr_scheduler(optimizer,
                     scheduler_type,
                     warmup_steps=None,
                     num_steps=None):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_steps)
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "consine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps)
        # scheduler = WarmupCosineSchedule(optimizer, warmup_steps, num_steps)
    else:
        raise ValueError("Unknown scheduler_type:", scheduler_type)

    # Prevent warning about not calling optimizer.step()
    optimizer._step_count = 1
    return scheduler


def get_optimizer(config, model):

    # Do not apply weight_decay for one-dimensional parameters
    regularized_params = []
    non_regularized_params = []
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) == 1:
                non_regularized_params.append(param)
            else:
                regularized_params.append(param)

    params = [
        {"params": regularized_params, "weight_decay": config.weight_decay}
        # {"params": non_regularized_params, "weight_decay": 0}
    ]

    if config.optimizer == "AdamW":
        optimizer = AdamW(params,
                          lr=config.learning_rate,
                          weight_decay=config.weight_decay,
                          eps=1e-6,
                          loss_scaling=config.loss_scaling,
                          accum_type=float16,
                          first_order_momentum_accum_type=float32
                        #   second_order_momentum_accum_type=float32
                          )
    elif config.optimizer == "Adam":
        optimizer = Adam(params,
                         lr=config.learning_rate,
                         weight_decay=config.weight_decay,
                         betas=(config.beta1, config.beta2),
                         eps=config.eps,
                         loss_scaling=config.loss_scaling,
                         accum_type=float16)
    elif config.optimizer == "LAMBNoBias":
        optimizer = LAMB(params,
                         lr=config.learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=config.loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         bias_correction=False)
    elif config.optimizer == "LAMB":
        optimizer = LAMB(params,
                         lr=config.learning_rate,
                         weight_decay=0,
                         eps=1e-6,
                         loss_scaling=config.loss_scaling,
                         max_weight_norm=None,
                         accum_type=float16,
                         bias_correction=True)
    elif config.optimizer == "SGD":
        optimizer = SGD(params,
                         lr=config.learning_rate,
                         momentum=config.momentum,
                         weight_decay=config.weight_decay,
                         loss_scaling=config.loss_scaling,
                         use_combined_accum=True)
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)
    return optimizer
