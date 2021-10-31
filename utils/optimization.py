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

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule
from torch import float16, float32


def get_lr_scheduler(optimizer, steps_per_epoch, config):

    warmup_steps = config.warmup_epochs * steps_per_epoch // config.gradient_accumulation
    num_steps = config.epochs * steps_per_epoch // config.gradient_accumulation + 1
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=-1)

    if config.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_steps)
    elif config.lr_schedule == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif config.lr_schedule == "consine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=-1)
    else:
        raise ValueError("Unknown scheduler_type:", config.lr_schedule)

    # Prevent warning about not calling optimizer.step()
    optimizer._step_count = 1
    return scheduler


def get_optimizer(config, model):
    # TODO-- ??
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

    if config.device == 'ipu':
        from poptorch.optim import AdamW, Adam, LAMB, SGD
    else:
        from torch.optim import AdamW, Adam, LAMB, SGD
    
    if config.optimizer == "AdamW":
        optimizer = AdamW(params,
                        lr=config.learning_rate,
                        betas=config.adam_beta, 
                        eps=config.eps,
                        weight_decay=config.weight_decay,
                        #   loss_scaling=config.loss_scaling,
                        #   accum_type=float16,
                        #   first_order_momentum_accum_type=float32
                        #   second_order_momentum_accum_type=float32
                    )
    elif config.optimizer == "Adam":
        optimizer = Adam(params,
                        lr=config.learning_rate,
                        weight_decay=config.weight_decay,
                        betas=config.adam_beta,
                        eps=config.eps,
                        # loss_scaling=config.loss_scaling,
                        # accum_type=float16
                    )
    elif config.optimizer == "LAMBNoBias":
        optimizer = LAMB(params,
                        lr=config.learning_rate,
                        weight_decay=0,
                        eps=1e-6,
                        loss_scaling=config.loss_scaling,
                        max_weight_norm=None,
                        # accum_type=float16,
                        bias_correction=False # TODO--
                    )
    elif config.optimizer == "LAMB":
        optimizer = LAMB(params,
                        lr=config.learning_rate,
                        weight_decay=0,
                        eps=1e-6,
                        # loss_scaling=config.loss_scaling,
                        max_weight_norm=None,
                        # accum_type=float16,
                        bias_correction=True
                    )
    elif config.optimizer == "SGD":
        optimizer = SGD(params,
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay,
                        # loss_scaling=config.loss_scaling,
                        use_combined_accum=True
                    )
    else:
        raise ValueError("Unknown Optimizer:", config.optimizer)
    return optimizer
