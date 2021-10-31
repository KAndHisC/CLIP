# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from numpy.core.einsumfunc import _compute_size_by_dict
import yaml
import argparse


config_file = "./configs.yml"


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_args(args=None):
    pparser = argparse.ArgumentParser("CLIP Configuration name", add_help=False)
    pparser.add_argument("--config",
                         type=str,
                         help="Configuration Name",
                         default='default')
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "CLIP",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # This is here only for the help message
    parser.add_argument("--config", type=str, help="Configuration name")
    # Execution
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'ipu', 'gpu'], default='cpu', help="device for runing")
    
    # IPU
    ## ??TODO--
    parser.add_argument("--layers-per-ipu", type=int, nargs="+",
                        help="Number of encoders placed on each IPU. Can be a single number, for an equal number encoder layers per IPU.Or it can be a list of numbers, specifying number of encoder layers for each individual IPU.")
    parser.add_argument("--layers-on-ipu", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")

    parser.add_argument("--replication-factor", type=int, help="Number of replicas")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                        "If True the output of each encoder layer will be stashed keeping the max liveness "
                        "of activations to be at most one layer. "
                        "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                        "The added stash + code could be greater than the reduction in temporary memory.",)
    parser.add_argument("--ipus-per-replica", type=int, help="Number of IPUs required by each replica")
    parser.add_argument("--encoder-start-ipu", type=int, choices=[0, 1],
                        help="The index of the IPU that the first encoder will be placed on. Can be 0 or 1.")
    parser.add_argument("--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul") # AMP ? TODO--
    parser.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable asynchronous mode in the DataLoader")
    parser.add_argument("--file-buffer-size", type=int, help="Number of files to load into the Dataset internal buffer for shuffling.")
    parser.add_argument("--random-seed", type=int, help="Seed for RNG")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--normalization-location', choices=['host', 'ipu', 'none'], default='host', help='Location of the data normalization')
    parser.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['AdamW', 'Adam', 'SGD', 'LAMB', 'LAMBNoBias'], help="optimizer to use for the training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate value for constant schedule, maximum for linear schedule.")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"],
                        help="Type of learning rate schedule. --learning-rate will be used as the max value")
    parser.add_argument("--loss-scaling", type=float, help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--weight-decay", type=float, help="Set the weight decay")
    parser.add_argument("--momentum", type=float, help="The momentum factor of SGD optimizer")
    parser.add_argument("--warmup-epochs", type=int, help="TODO--") # TODO--
    parser.add_argument("--gradient-accumulation", type=int, help="Number of gradients to accumulate before updating the weights")
    parser.add_argument('--eps', type=float, default=1e-6, help="")
    parser.add_argument('--adam_beta', type=float, nargs="+", default=[0.9, 0.98], help="")
    ## ?? TODO--
    parser.add_argument("--lr-warmup", type=float, help="Proportion of lr-schedule spent in warm-up. Number in range [0.0, 1.0]")
    

    # Dataset
    parser.add_argument("--batch_size", type=int, help="Set the micro batch_size")
    parser.add_argument("--image-path", type=str, nargs="+", help="The path of image files")
    parser.add_argument("--captions-path", type=str, nargs="+", help="The path of text file")
    parser.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                        help="No Host/IPU I/O, random data created on device")

    # Misc
    parser.add_argument("--num-workers", type=int, help="The number of dataloader workers")
    parser.add_argument("--profile", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable profiling")
    parser.add_argument("--profile-dir", type=str, help="Directory for profiling results")
    parser.add_argument("--custom-ops", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable custom ops")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    ## ?? TODO-- ##
    parser.add_argument("--use-popdist", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling poprun function")
    parser.add_argument("--popdist-size", type=int, help="The popdist size")
    parser.add_argument("--enable-rts", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling RTS")
    parser.add_argument("--pred-head-transform", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable prediction head transform in the CLS layer during pretraining. This transform comes after\
                        the encoders but before the decoder projection for the MLM loss.")
    parser.add_argument("--embedding-serialization-factor", type=int, help="Matmul serialization factor the embedding layers")
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable half partials for matmuls and convolutions globally")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Directory where checkpoints will be saved and restored from.\
                             This can be either an absolute or relative path. If this is not specified, only end of run checkpoint is\
                             saved in an automatically generated directory at the root of this project. Specifying directory is\
                             recommended to keep track of checkpoints.")
    parser.add_argument("--checkpoint-every-epoch", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Option to checkpoint model after each epoch.")
    parser.add_argument("--checkpoint_save_epochs", type=int, default=1,
                        help="Option to checkpoint model after n epochs.")
    parser.add_argument("--checkpoint-file", type=str, default="", help="Checkpoint to be retrieved for further training. This can\
                              be either an absolute or relative path to the checkpoint file.")
    # restore TODO--
    parser.add_argument("--restore", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore a checkpoint model to continue training.")
    parser.add_argument("--restore-epochs-and-optimizer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore epoch and optimizer state to continue training. This should normally be True when resuming a\
                              previously stopped run, otherwise False.")

    # CLIP
    parser.add_argument("--epochs", type=int, help="")
    
    parser.add_argument('--image_resolution', type=int, default=224, help="The shape of image after preprocess")
    parser.add_argument('--context_length', type=int, default=77, help="The model can handle fix length text")
    parser.add_argument('--transformer_heads', type=int, default=8, help="Set the number of heads in self attention of  custom transformer")
    parser.add_argument('--transformer_width', type=int, default=512, help="The dim of custom transformer")
    parser.add_argument('--transformer_layers', type=int, default=12, help="The layers of custom transformer")
    parser.add_argument('--embed_dim', type=int, default=512, help="The dim of custom transformer")
    parser.add_argument('--vision_width', type=int, default=768, help="The dim of ViT")
    parser.add_argument('--vision_layers', type=int, default=12, help="The layers of ViT")
    parser.add_argument('--vision_heads', type=int, default=12, help="Set the number of heads in self attention of ViT")
    parser.add_argument('--vision_patch_size', type=int, default=32, help="The size of vit to patch")
    parser.add_argument('--grid_size', type=int, default=7, help="")
    parser.add_argument('--vocab_size', type=int, default=49408, help="The size of vocabulary")
    parser.add_argument('--truncate', type=bool, default=True, help="")
    
    

    parser.add_argument("--layer-norm-eps", type=float, help="The eps value for the layer norms")

    # compute
    # parser.add_argument("--training-steps", type=int, help="Number of training steps")
    # parser.add_argument("--batches-per-step", type=int, help="Number of batches per training step")

    

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        print(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args(remaining_args)

    # Expand matmul_proportion input into list representation
    if args.device == 'ipu':
        if isinstance(args.matmul_proportion, float):
            args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

        if len(args.matmul_proportion) != args.ipus_per_replica:
            if len(args.matmul_proportion) == 1:
                args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
            else:
                raise ValueError(f"Length of matmul_proportion doesn't match ipus_per_replica: {args.matmul_proportion} vs {args.ipus_per_replica}")

        args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size
        args.samples_per_step = args.global_batch_size * args.iterations
    else:
        args.global_batch_size = args.gradient_accumulation * args.batch_size
        args.samples_per_step = args.global_batch_size

    return args
