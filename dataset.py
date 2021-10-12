from clip.clip import tokenize
import torch
import os
import numpy as np
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


tokenizer = SimpleTokenizer()


def make_train_valid_dfs(cfg):
    image_names = []
    image_captions = []
    with open(f"{cfg.captions_path}/Flickr8k.token.txt") as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_name = line.split('\t')[0].split('#')[0]
            if os.path.exists(f"{cfg.image_path}/{image_name}") is False:
                print(f"{cfg.image_path}/{image_name} not exist")
                continue
            image_names.append(image_name)
            image_captions.append(line.split('\t')[1])

    max_id = len(image_names)
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    dataframe = pd.DataFrame({"id": image_ids, "image": image_names[:max_id], "caption": image_captions[:max_id]})
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe


def get_transforms(cfg):
    n_px = cfg.image_resolution

    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def tokenize(texts, cfg):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), cfg.context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > cfg.context_length:
            if cfg.truncat:
                tokens = tokens[: cfg.context_length]
                tokens[-1] = eot_token

            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {cfg.context_length}")

        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_filenames = image_filenames
        self.captions = list(captions)
        print("The number of image-text pairs: ", len(self.captions))
        self.encoded_captions = tokenize(list(captions), cfg)
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {"input_ids": self.encoded_captions[idx]}
        image = Image.open(f"{self.cfg.image_path}/{self.image_filenames[idx]}")
        image = self.transforms(image)
        item['image'] = image
        # item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def build_loaders(cfg):
    train_dataframe, test_dataframe = make_train_valid_dfs(cfg)
    transforms = get_transforms(cfg)
    train_dataset = CLIPDataset(
        train_dataframe["image"].values,
        train_dataframe["caption"].values,
        transforms=transforms,
        cfg=cfg
    )
    test_dataset = CLIPDataset(
        test_dataframe["image"].values,
        test_dataframe["caption"].values,
        transforms=transforms,
        cfg=cfg
    )

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
