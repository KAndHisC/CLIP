import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import torch
import poptorch
from .simple_tokenizer import SimpleTokenizer
from typing import Union, List
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


_tokenizer = SimpleTokenizer()


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms, config):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.config  = config
        self.image_filenames = image_filenames
        self.captions = list(captions)

        self.encoded_captions = tokenize(list(captions), context_length=self.config.context_length, truncate=self.config.truncate)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {"input_ids": self.encoded_captions[idx]}

        image = Image.open(f"{self.config.image_path}/{self.image_filenames[idx]}")
        image = self.transforms(image)
        item['image'] = image.half()
        item['caption'] = self.captions[idx]
        
        return item['image'], item['input_ids']


    def __len__(self):
        return len(self.captions)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(mode="train", config=None):
    n_px = config.image_resolution

    return Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def make_train_valid_dfs(mode,config):
    image_names = []
    image_captions = []
    with open(f"{config.captions_path}/Flickr8k.token.txt") as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_name = line.split('\t')[0].split('#')[0]
            if os.path.exists(f"{config.image_path}/{image_name}") is False:
                print(f"{config.image_path}/{image_name} not exist")
                continue
            image_names.append(image_name)
            image_captions.append(line.split('\t')[1])
    # max_id = len(image_names) if not config.debug else 100
    max_id = len(image_names)
    image_ids = np.arange(0, max_id)
    dataframe = pd.DataFrame({"id":image_ids,"image":image_names[:max_id],"caption":image_captions[:max_id]})

    return dataframe

def build_loaders(mode,config,opts,async_dataloader):
    dataframe = make_train_valid_dfs(mode,config)
    transforms = get_transforms(mode=mode,config=config)
    
    train_dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        transforms=transforms,
        config=config
    )

    dataset_mode = poptorch.DataLoaderMode.Async if async_dataloader else poptorch.DataLoaderMode.Sync
    train_dataloader = poptorch.DataLoader(opts,
                                     train_dataset,
                                     batch_size=config.batch_size if not(isinstance(train_dataset, torch.utils.data.IterableDataset)) else None,
                                     num_workers=config.dataloader_workers,
                                     shuffle=not(isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                     drop_last=not(isinstance(train_dataset, torch.utils.data.IterableDataset)),
                                     persistent_workers = True,
                                     auto_distributed_partitioning = not isinstance(train_dataset, torch.utils.data.IterableDataset),
                                     worker_init_fn=None,
                                     mode=dataset_mode,
                                     async_options={'load_indefinitely': True})
    
    return train_dataloader


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


if __name__  == '__main__':
    a = tokenize(['A child in a pink dress is climbing up a set of stairs in an entry way .',
    'A girl going into a wooden building .'])
    print(a)
    print(a.size())
