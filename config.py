import torch

class CFG:
    debug = False
    image_path = "/fsx/home/minghongc/datasets/Flickr8k/Flicker8k_Dataset"
    captions_path = "/fsx/home/minghongc/datasets/Flickr8k"
    batch_size = 32
    num_workers = 4
    lr = 5e-4
    weight_decay = 1e-3
    patience = 2
    factor = 0.5
    epochs = 20
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
