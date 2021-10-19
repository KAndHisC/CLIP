from time import process_time
from clip.clip import tokenize
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.std import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_dfs(cfg):
    image_names = []
    image_captions = []
    # with open(f"{cfg.captions_path}/Flickr8k.token.txt") as f:
    with open(cfg.captions_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip('\n')
            # image_name = line.split('\t')[0].split('#')[0]
            image_name = line.split('\t')[0] # use mm dataset
            # if os.path.exists(f"{cfg.image_path}/{image_name}") is False:
            #     print(f"{cfg.image_path}/{image_name} not exist")
            #     continue
            image_names.append(image_name)
            image_captions.append(line.split('\t')[1])

    max_id = len(image_names)
    image_ids = np.arange(0, max_id)
    dataframe = pd.DataFrame({"id": image_ids, "image": image_names[:max_id], "caption": image_captions[:max_id]})
 
    return dataframe


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, cfg):
        super().__init__()
        self.image_path = cfg.image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.length = len(self.captions)
        print("The number of image-text pairs: ", self.length)

        self.preprocess = cfg.preprocess

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_path}/{self.image_filenames[idx]}")
        image = self.preprocess(image)
        return image, self.captions[idx]

    def __len__(self):
        return self.length


def build_loaders(cfg):
    dataframe = load_dfs(cfg)
    datasets = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        cfg=cfg
    )
    test_size = int(len(datasets) * 0.2)
    train_size = len(datasets) - test_size
    print("train_size: ", train_size)
    print("test_size: ", test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False
    )

    return train_dataloader, test_dataloader
