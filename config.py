import torch

class CFG:
    debug = False
    image_path = "/fsx/home/minghongc/datasets/Flickr8k/Flicker8k_Dataset"
    captions_path = "/fsx/home/minghongc/datasets/Flickr8k"
    batch_size = 128
    num_workers = 4

    # optimizer
    # lr = 5e-4
    lr = 1e-6
    adam_beta = [0.9, 0.98]
    # weight_decay = 1e-3
    weight_decay=0.2

    # scheduler
    epochs = 20
    warmup_steps = 2000
    # num_steps = epochs * (num of data / batch size)
    patience = 2 # ?
    factor = 0.5 # ?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temperature = 1.0
    
    # CLIP
    embed_dim = 512
    transformer_width = 512
    transformer_heads = 8
    transformer_layers = 12

    vision_width = 768
    vision_layers = 12
    vision_heads = 12
    vision_patch_size = 32
    grid_size = 7
    image_resolution = 224

    vocab_size = 49408
    context_length = 77
    truncate = True 
