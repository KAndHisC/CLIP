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

import os
import time
import datetime
import torch
import numpy as np

from transformers import BertConfig
import wandb
import warnings
from math import ceil
from args import parse_args

from model import PipelinedWithLoss

from optimization import get_lr_scheduler, get_optimizer
from checkpoint import save_model, maybe_load_checkpoint_passing_constraints, prepare_checkpoint_metrics

from datasets import build_loaders
from log import Logger

if __name__ == "__main__":
    # TODO -- auto choice IPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        import poptorch
        from poptorch.enums import DataLoaderMode
        from ipu_options import get_options

    # Ignore known warnings
    # warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = BertConfig(**(vars(parse_args())))

    # Check output dir 
    abs_pathd = os.path.abspath(config.checkpoint_dir)
    os.makedirs(abs_pathd, exist_ok=True)

    # logging.getLogger("poptorch::python").setLevel(logging.ERROR)
    log = Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',level='info')

    # W&B
    if config.wandb:
        wandb.init(project="CLIP", settings=wandb.Settings(console='off'))
        wandb.config.update(vars(config))

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    train_loader = build_loaders(mode="train", config=config, opts=opts, async_dataloader=True)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration")

    # IPU Model and Optimizer
    print("instance the model")
    model = PipelinedWithLoss(config).half().train()
    print(model)
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print("total_trainable_params:{}".format(total_trainable_params))
    optimizer = get_optimizer(config, model)
    scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                 config.warmup_steps, config.training_steps)

    #init from a checpoint
    if config.checkpoint_file:
        # model.load_from(np.load(config.checkpoint_file))
        pass

    # Restore model from checkpoint
    epochs_finished = 0
    if config.restore:
        # Retrieve relevant checkpoint
        checkpoint = maybe_load_checkpoint_passing_constraints(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.restore_epochs_and_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.last_epoch = epochs_finished = checkpoint["epoch"]
            checkpoint_metrics = checkpoint["metrics"]
        else:
            # Checkpoint model with epochs and optimizer state reset
            # for further training
            log.logger.info("Save checkpoint path: {}".format(save_model(config, model, optimizer, epochs_finished)))
    else:
        # Checkpoint model at start of run
        log.logger.info("Save checkpoint path: {}".format(save_model(config, model, optimizer, epochs_finished)))

    train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    # log.logger.info("---------- Compilation Started ---------")
    # start_compile = time.perf_counter()
    # datum = dataset.get_random_datum(config)
    # train_model.compile(*datum)
    # duration_compilation = time.perf_counter() - start_compile
    # log.logger.info(f"Compiled model in {duration_compilation} secs")
    # log.logger.info("---------------------------------------")

    #popdist 
    num_instances = config.popdist_size if config.use_popdist else 1

    # Training loop
    log.logger.info("---------- Training Started -----------")

    factor = config.gradient_accumulation * config.batches_per_step
    epochs = ceil(config.training_steps / steps_per_epoch) - epochs_finished
    training_steps = config.training_steps - (steps_per_epoch * epochs_finished)
    # start_train = time.perf_counter()
    # start_step = time.perf_counter()
    for epoch in range(epochs):
        # start_step = time.perf_counter()
        for step, (image, input_ids) in enumerate(train_loader):
            current_step = step + epoch * steps_per_epoch
            if current_step == 2:
                start_train = time.perf_counter()

            start_step = time.perf_counter()
            losses = train_model(image, input_ids)
            print("Losses: ", losses)
            scheduler.step()
            train_model.setOptimizer(optimizer)
            step_length = time.perf_counter() - start_step
            step_throughput = config.samples_per_step / step_length

            log.logger.info("Epoch: {:.2f}/{} Step: {}/{} Lr: {:.6f} Loss: {:.3f} Througput: {:.2f} samples/sec" \
                         .format(epoch, epochs, \
                         current_step, training_steps, \
                         scheduler.get_last_lr()[0], \
                         losses, \
                         step_throughput))
            if config.wandb:
                wandb.log({"LR": scheduler.get_last_lr()[0],
                           "Throughput": step_throughput,
                           "Loss": losses})

            # start_step = time.perf_counter()
            if current_step + 1 == training_steps:
                break  # Training finished mid-epoch
            if current_step%config.checkpoint_save_steps==0  and (current_step + 1 != training_steps):
                log.logger.info("Save checkpoint path: {}".format(save_model(config, model, optimizer, epoch + epochs_finished + 1,
                        metrics=prepare_checkpoint_metrics(losses, factor))))

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    save_path = save_model(config, model, optimizer, epoch + epochs_finished + 1,
               metrics=prepare_checkpoint_metrics(losses, factor))
    log.logger.info("Save checkpoint path: {}".format(save_path))
    log.logger.info("---------------------------------------")

    log.logger.info("---------- Training Metrics -----------")
    log.logger.info(f"global_batch_size: {config.global_batch_size}")
    log.logger.info(f"batches_per_step: {config.batches_per_step}")
    log.logger.info(f"training_steps: {training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * (training_steps-2)
    log.logger.info(f"Training time: {duration_run:.3f} secs")
    log.logger.info("Throughput: {:5f} samples/sec.".format(num_samples / duration_run))
    log.logger.info("---------------------------------------")
    