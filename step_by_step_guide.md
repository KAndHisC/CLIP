# create a env

```
conda create -n clip python=3.9
conda activate clip
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install ftfy regex tqdm
conda install -c conda-forge scikit-lear
pip install transformers
```

# get repo and setup

```
https://github.com/KAndHisC/CLIP.gitc
cd CLIP
git checkout takiw_test
python setup.py develop
```

# get datasets

