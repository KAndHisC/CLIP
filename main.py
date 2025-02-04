import clip
import torch
from dataset import build_loaders
from config import CFG
from clip.model import CLIP
from tqdm.autonotebook import tqdm
# from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import numpy as np

import wandb

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
def train_epoch(model, train_loader, optimizer):

    tqdm_object = tqdm(train_loader, mininterval=5.0)
    i = 0
    for images, texts in tqdm_object:
        
        images = images.to(cfg.device)
        texts = texts.to(cfg.device)

        logits_per_image, logits_per_text = model(images, texts)
        # loss = custom_loss(logits_per_image, logits_per_text)
        loss = paper_loss(logits_per_image, logits_per_text)
        loss.backward()

        if (i+1) % cfg.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            # tqdm_object.set_postfix(train_loss=loss_meter.avg, learing_rate=cfg.lr)
            tqdm_object.set_postfix(train_loss=loss.item(), learing_rate=scheduler.get_last_lr()[0])
            wandb.log({"LR": scheduler.get_last_lr()[0], "Loss": loss})
        i += 1

    return None


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for images, texts in tqdm_object:
        images = images.to(cfg.device)
        texts = texts.to(cfg.device)

        logits_per_image, logits_per_text = model(images, texts)
        # loss = custom_loss(logits_per_image, logits_per_text)
        loss = paper_loss(logits_per_image, logits_per_text)
        count = images.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter


torch_cross_entropy_func = torch.nn.CrossEntropyLoss()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()



def paper_loss(logits_per_image, logits_per_text):
    labels = torch.Tensor(np.arange(logits_per_image.size()[0])).long().to(cfg.device)
    # labels = torch.eye(logits_per_image.size()[0]).cuda()

    loss_text = torch_cross_entropy_func(logits_per_text, labels)
    loss_image = torch_cross_entropy_func(logits_per_image, labels)
    loss =  (loss_text + loss_image) / 2.0

    return loss.mean()

def custom_loss(logits_per_image, logits_per_text):
    
    logits = logits_per_text
    targets = F.softmax(
        (logits_per_image + logits_per_text) / 2, dim=-1
    )

    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.t(), targets.t(), reduction='none')

    loss = (images_loss + texts_loss) / 2.0

    return loss.mean()


if __name__ == '__main__':
    # config
    cfg = CFG()
    
    # wandb
    wandb.init(project="CLIP-GPU", settings=wandb.Settings(console='off'))
    wandb.config.update(vars(cfg))
    
    model = CLIP(embed_dim=cfg.embed_dim, 
                # vision
                image_resolution=cfg.image_resolution,
                vision_layers=cfg.vision_layers,
                vision_width=cfg.vision_width,
                vision_patch_size=cfg.vision_patch_size,
                context_length=cfg.context_length,
                vocab_size=cfg.vocab_size,
                # text
                transformer_width=cfg.transformer_width,
                transformer_heads=cfg.transformer_heads,
                transformer_layers=cfg.transformer_layers).to(cfg.device)
    
    # DataLoader
    train_loader, test_loader = build_loaders(cfg=cfg)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.adam_beta, eps=1e-6, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.adam_beta, eps=cfg.eps, weight_decay=cfg.weight_decay) # in paper
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=cfg.patience, factor=cfg.factor
    # )
    warmup_steps = cfg.warmup_epochs * len(train_loader) // cfg.gradient_accumulation
    num_steps = cfg.epochs * len(train_loader) // cfg.gradient_accumulation + 1
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_steps, last_epoch=-1)

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        # train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        train_loss = train_epoch(model, train_loader, optimizer)
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, test_loader)
            wandb.log({"Valid_Loss": valid_loss.avg})

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "./models/200m"+str(valid_loss.avg)[:5]+".pt")
            print("saved the best model! ")

        # lr_scheduler.step(valid_loss.avg)
        # print("lr: ", get_lr(optimizer))
    
