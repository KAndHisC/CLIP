import os
from clip import _convert_image_to_rgb, _transform, tokenize
import numpy as np
import pandas as pd
import torch

from PIL import Image
from PIL import ImageFile

from tqdm.std import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, config):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        super().__init__() # TODO--

        self.config  = config
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.length = len(self.captions)
        print("The number of image-text pairs: ", self.length)

        self.preprocess = _transform(config.image_resolution)
        self.context_length = config.context_length
        self.truncate = config.truncate


    def __getitem__(self, idx):
        image = Image.open(f"{self.config.image_path}/{self.image_filenames[idx]}")
        texts = clip.tokenize(self.captions[idx], self.context_length, truncate=self.truncate).squeeze()
        # try:
        #     image = Image.open(f"{self.image_path}/{self.image_filenames[idx]}")
        # except:
        #     print(f"{self.image_path}/{self.image_filenames[idx]}")
        #     return torch.rand(3, 224, 224), texts
        # else:
        #     image = self.preprocess(image)
        
        image = self.preprocess(image)
        return image, texts

    def __len__(self):
        return self.length


def load_dfs(cfg):
    image_names = []
    image_captions = []
    # with open(f"{cfg.captions_path}/Flickr8k.token.txt") as f:
    with open(cfg.captions_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip('\n')
            # image_name = line.split('\t')[0].split('#')[0]
            image_name = line.split('\t')[0] # use mm dataset
            # if os.path.exists(f"{config.image_path}/{image_name}") is False:
            #     print(f"{config.image_path}/{image_name} not exist")
            #     continue
            image_names.append(image_name)
            image_captions.append(line.split('\t')[1])

    max_id = len(image_names)
    image_ids = np.arange(0, max_id)
    dataframe = pd.DataFrame({"id": image_ids, "image": image_names[:max_id], "caption": image_captions[:max_id]})
 
    return dataframe


def build_loaders(config, async_dataloader, IPU_opts=None):
    dataframe = load_dfs(config)
    
    datasets = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        config=config
    )

    test_size = int(len(datasets) * 0.2)
    # test_size = 1
    train_size = len(datasets) - test_size
    print("train_size: ", train_size)
    print("test_size: ", test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])

    if not IPU_opts:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False
        )
    else:
        import poptorch

        dataset_mode = poptorch.DataLoaderMode.Async if async_dataloader else poptorch.DataLoaderMode.Sync
        # isIterable = isinstance(train_dataset, torch.utils.data.IterableDataset)
        train_dataloader = poptorch.DataLoader(
            IPU_opts, train_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True,
            drop_last=False,
            # persistent_workers = True, # ?
            # auto_distributed_partitioning = False, # ?
            # worker_init_fn=None,
            mode=dataset_mode,
            # async_options={'load_indefinitely': True} # ?
        )
        test_dataloader = poptorch.DataLoader(
            IPU_opts, train_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False,
            drop_last=False,
            mode=dataset_mode,
        )
    
    return train_dataloader, test_dataloader


if __name__  == '__main__':
    a = tokenize(['A child in a pink dress is climbing up a set of stairs in an entry way .',
    'A girl going into a wooden building .'])
    print(a)
    print(a.size())
