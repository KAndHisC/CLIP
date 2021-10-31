import os
from PIL import Image
from tqdm.std import tqdm

cleand_path = "/fsx/home/minghongc/datasets/mm/img_cap_cleand.txt"
def clean_dataset(config, preprocess):
    with open(config.captions_path, "r", encoding="utf-8") as f:
        file = open(cleand_path, 'w')
        for raw_line in tqdm(f.readlines()):
            line = raw_line.strip('\n')
            # image_name = line.split('\t')[0].split('#')[0]
            image_name = line.split('\t')[0] # use mm dataset
            if os.path.exists(f"{config.image_path}/{image_name}") is False:
                print(f"{config.image_path}/{image_name} not exist")
                continue
            # try:
            #     Image.open(f"{config.image_path}/{image_name}")
            #     image = preprocess(Image.open(f"{config.image_path}/{image_name}"))
            # except OSError:
            #     print(f"{config.image_path}/{image_name}")
            else:
                file.write(raw_line)
        file.close()