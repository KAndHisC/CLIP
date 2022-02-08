# create a env

```
conda create -n clip python=3.6
conda activate clip
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda install torchvision torchaudio cpuonly -c pytorch
conda install ftfy regex tqdm
conda install -c conda-forge scikit-learn
pip install transformers
```
cat>$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh<<EOF
#!/bin/bash
export GCDA_MONITOR=1
export TF_CPP_VMODULE="poplar_compiler=1"
export TF_POPLAR_FLAGS="--max_compilation_threads=40 --executable_cache_path=/localdata/takiw/cachedir"
export TMPDIR="/localdata/takiw/tmp"
export POPART_LOG_LEVEL=DEBUG
export POPTORCH_LOG_LEVEL=TRACE
export POPLAR_LOG_LEVEL=DEBUG
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./reports"}'

source /localdata/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0-EA.1+814-4c21ad5946/poplar-ubuntu_18_04-2.4.0+1998-b635644fb9/enable.sh
source /localdata/takiw/sdk/poplar_sdk-ubuntu_18_04-2.4.0-EA.1+814-4c21ad5946/popart-ubuntu_18_04-2.3.0+1998-b635644fb9/enable.sh
EOF


# get repo and setup

```
https://github.com/KAndHisC/CLIP.gitc
cd CLIP
git checkout takiw_test
python setup.py develop
```

# get datasets

poplar_sdk-ubuntu_18_04-2.2.0+688-7a4ab80373/popart-ubuntu_18_04-2.2.0+166889-feb7f3f2bb
poplar_sdk-ubuntu_18_04-2.2.0+688-7a4ab80373/poplar-ubuntu_18_04-2.2.0+166889-feb7f3f2bb