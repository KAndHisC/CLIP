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


import poptorch
import popart
import numpy as np
import ctypes
import os


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''

    # Custom ops
    if config.custom_ops is True:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
        if os.path.exists(CUSTOM_OP_PATH):
            ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
            ops_and_patterns.setVocabSize(config.vocab_size)
            ops_and_patterns.setEmbeddingSize(config.hidden_size)
            ops_and_patterns.setHiddenSize(config.hidden_size)
        else:
            print("Could not find custom_ops.so. Execute `make` before running this script.")
            exit()

    # Numpy options
    np.random.seed(config.random_seed)

    # Poptorch options
    opts = poptorch.Options()

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.batches_per_step)
    opts.replicationFactor(config.replication_factor)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.anchorMode(poptorch.AnchorMode.Final)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings().useOnChipStorage(True))
    if config.enable_rts:
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings().useReplicatedTensorSharding(True).minElementsForReplicatedTensorSharding(config.replication_factor))

    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    opts.randomSeed(config.random_seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.SameAsIpu))

    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    # PopART options
    opts._Popart.set("enableStochasticRounding", True)
    opts._Popart.set("virtualGraphMode", int(popart.VirtualGraphMode.Manual))
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("enableGradientAccumulation", True)
    opts._Popart.set("enableOutlining", True)
    opts._Popart.set("outlineThreshold", 10.0)
    # opts._Popart.set("enableGroupedMatmuls", False)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                    int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])

    if config.enable_half_partials:
        opts._Popart.set("partialsTypeMatMuls", "half")
        opts._Popart.set("convolutionOptions", {'partialsType': "half"})

    if config.synthetic_data:
        opts._Popart.set("syntheticDataMode", int(popart.SyntheticDataMode.RandomNormal))

    engine_options = {"target.syncReplicasIndependently": "true"}
    if config.profile:
        engine_options = {
            **engine_options,
            **{
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": config.profile_dir,
                "profiler.format": "v3",
                "autoReport.all": "true",
            }
        }
    opts._Popart.set("engineOptions", engine_options)

    return opts
