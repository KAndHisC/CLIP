from clip.model import CLIP
import torch
from data import CLIPDataset, load_dfs
from tqdm.autonotebook import tqdm
from utils.loss import paper_loss
from utils.metrics import AvgMeter
# from tqdm import tqdm_notebook as tqdm
best_loss = float('inf')

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
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True
        )
    else:
        import poptorch

        dataset_mode = poptorch.DataLoaderMode.Async if async_dataloader else poptorch.DataLoaderMode.Sync
        # isIterable = isinstance(train_dataset, torch.utils.data.IterableDataset)
        train_dataloader = poptorch.DataLoader(
            IPU_opts, train_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True,
            # persistent_workers = True, # ?
            # auto_distributed_partitioning = False, # ?
            # worker_init_fn=None,
            mode=dataset_mode,
            # async_options={'load_indefinitely': True} # ?
        )
        test_dataloader = poptorch.DataLoader(
            IPU_opts, train_dataset,
            batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True,
            mode=dataset_mode,
        )
    
    return train_dataloader, test_dataloader


# def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
def train_epoch(model, train_loader, optimizer, scheduler, config, wandb=None):
    tqdm_object = tqdm(train_loader, mininterval=5.0)
    i = 0
    for images, texts in tqdm_object:
        images = images.to(config.device)
        texts = texts.to(config.device)

        logits_per_image, logits_per_text = model(images, texts)
        # loss = custom_loss(logits_per_image, logits_per_text)
        loss = paper_loss(logits_per_image, logits_per_text)
        loss.backward()

        if (i+1) % config.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            # tqdm_object.set_postfix(train_loss=loss_meter.avg, learing_rate=cfg.lr)
            tqdm_object.set_postfix(train_loss=loss.item(), learing_rate=scheduler.get_last_lr()[0])
            if wandb:
                wandb.log({"LR": scheduler.get_last_lr()[0], "Loss": loss})
        i += 1

    return None

def valid_epoch(model, valid_loader, config):
    loss_meter = AvgMeter()
    model.eval()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for images, texts in tqdm_object:
            images = images.to(config.device)
            texts = texts.to(config.device)

            logits_per_image, logits_per_text = model(images, texts)
            # loss = custom_loss(logits_per_image, logits_per_text)
            loss = paper_loss(logits_per_image, logits_per_text)
            count = images.size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter

def get_officail_CLIP_model(config):
    return CLIP(embed_dim=config.embed_dim, 
                # vision
                image_resolution=config.image_resolution,
                vision_layers=config.vision_layers,
                vision_width=config.vision_width,
                vision_patch_size=config.vision_patch_size,
                context_length=config.context_length,
                vocab_size=config.vocab_size,
                # text
                transformer_width=config.transformer_width,
                transformer_heads=config.transformer_heads,
                transformer_layers=config.transformer_layers
            ).to(config.device)

def get_officail_CLIP_model(config):
    return CLIP(embed_dim=config.embed_dim, 
                # vision
                image_resolution=config.image_resolution,
                vision_layers=config.vision_layers,
                vision_width=config.vision_width,
                vision_patch_size=config.vision_patch_size,
                context_length=config.context_length,
                vocab_size=config.vocab_size,
                # text
                transformer_width=config.transformer_width,
                transformer_heads=config.transformer_heads,
                transformer_layers=config.transformer_layers
            ).to(config.device)

def save_best_model(model, valid_loss):
    if valid_loss < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./models/2m"+str(valid_loss.avg)[:5]+".pt")
            print("saved the best model! ")

