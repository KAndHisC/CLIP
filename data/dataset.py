import os
from clip import _transform, tokenize
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
        texts = tokenize(self.captions[idx], self.context_length, truncate=self.truncate).squeeze()
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


class SyntheticData(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image = image
        # self.texts = None
        # self.preprocess = None
        self.length = 1000000

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.image

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


if __name__  == '__main__':
    a = tokenize(['A child in a pink dress is climbing up a set of stairs in an entry way .',
    'A girl going into a wooden building .'])
    print(a)
    print(a.size())
