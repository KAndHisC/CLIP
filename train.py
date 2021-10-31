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
import clip
from transformers import BertConfig
import wandb
import warnings
from math import ceil

import utils

if __name__ == "__main__":
    # Ignore known warnings
    # warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

    # Build config from args
    config = BertConfig(**(vars(utils.parse_args())))
    
    # Check output dir 
    abs_pathd = os.path.abspath(config.checkpoint_dir)
    os.makedirs(abs_pathd, exist_ok=True)

    # logging.getLogger("poptorch::python").setLevel(logging.ERROR)
    log = utils.Logger(abs_pathd+"/"+datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')+'.log',level='info')

    # W&B
    if config.wandb:
        wandb.init(project="CLIP", settings=wandb.Settings(console='off'))
        wandb.config.update(vars(config))

    

    # Dataloader
    train_loader, test_loader = utils.build_loaders(config=config, async_dataloader=True, IPU_opts=opts,)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration")

    if config.device == "ipu":
        import poptorch
        from poptorch.enums import DataLoaderMode
        from clip_ipu import get_options
        from model import PipelinedWithLoss
        # Execution parameters
        opts = get_options(config)
        # IPU Model and Optimizer
        print("instance the model")
        model = PipelinedWithLoss(config).half().train()
        print(model)
        total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params:{}".format(total_trainable_params))
    else:
        model = utils.get_officail_CLIP_model(config)
    optimizer = utils.get_optimizer(config, model)
    scheduler = utils.get_lr_scheduler(optimizer, steps_per_epoch=steps_per_epoch, config=config)

    #init from a checpoint
    if config.checkpoint_file:
        # model.load_from(np.load(config.checkpoint_file))
        pass

    # Restore model from checkpoint TODO--
    epochs_finished = 0
    # if config.restore:
    #     # Retrieve relevant checkpoint
    #     checkpoint = maybe_load_checkpoint_passing_constraints(config)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     if config.restore_epochs_and_optimizer:
    #         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #         scheduler.last_epoch = epochs_finished = checkpoint["epoch"]
    #         checkpoint_metrics = checkpoint["metrics"]
    #     else:
    #         # Checkpoint model with epochs and optimizer state reset
    #         # for further training
    #         log.logger.info("Save checkpoint path: {}".format(save_model(config, model, optimizer, epochs_finished)))
    # else:
    #     # Checkpoint model at start of run
    #     log.logger.info("Save checkpoint path: {}".format(save_model(config, model, optimizer, epochs_finished)))
    
    if config.device == "ipu":
        train_model = poptorch.trainingModel(model, opts, optimizer=optimizer)

        # Compile model
        log.logger.info("---------- Compilation Started ---------")
        start_compile = time.perf_counter()
        datum = test_loader[0]
        train_model.compile(*datum)
        duration_compilation = time.perf_counter() - start_compile
        log.logger.info(f"Compiled model in {duration_compilation} secs")
        log.logger.info("---------------------------------------")

        #popdist 
        num_instances = config.popdist_size if config.use_popdist else 1

    # Training loop
    log.logger.info("---------- Training Started -----------")

    training_steps = steps_per_epoch * config.epochs
    # start_train = time.perf_counter()
    # start_step = time.perf_counter()
    for epoch in range(config.epochs):
        utils.train_epoch(model, train_loader, optimizer, scheduler, config, wandb=wandb)
        valid_loss = utils.valid_epoch(model, test_loader)
        wandb.log({"Valid_Loss": valid_loss.avg})
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./models/200m"+str(valid_loss.avg)[:5]+".pt")
            log.logger.info("Save checkpoint path: {}"
                            .format(
                                utils.save_model(config, model, optimizer, epoch)
                            )
                        )
            print("saved the best model! ")

            
            

    # stop_train = time.perf_counter()
    # # Checkpoint at end of run
    # save_path = save_model(config, model, optimizer, epoch + epochs_finished + 1,
    #            metrics=prepare_checkpoint_metrics(losses, factor))
    # log.logger.info("Save checkpoint path: {}".format(save_path))
    # log.logger.info("---------------------------------------")

    # log.logger.info("---------- Training Metrics -----------")
    # log.logger.info(f"global_batch_size: {config.global_batch_size}")
    # log.logger.info(f"batches_per_step: {config.batches_per_step}")
    # log.logger.info(f"training_steps: {training_steps}")
    # duration_run = stop_train - start_train
    # num_samples = config.samples_per_step * (training_steps-2)
    # log.logger.info(f"Training time: {duration_run:.3f} secs")
    # log.logger.info("Throughput: {:5f} samples/sec.".format(num_samples / duration_run))
    # log.logger.info("---------------------------------------")
    