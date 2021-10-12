from functools import total_ordering
import torch
from dataset import build_loaders
from config import CFG
from clip.model import CLIP
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
import time


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
def train_epoch(model, train_loader, lr, optimizer):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        # start_step = time.perf_counter()
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        logits_per_image, logits_per_text = model(batch["image"], batch["input_ids"])
        loss = get_loss(logits_per_image, logits_per_text)
        # print("loss: ", loss)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # if step == "batch":
        #     lr_scheduler.step()
        # step_length = time.perf_counter() - start_step
        # step_throughput = len(batch) / step_length
        # print("tput: {}".format(step_throughput))
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        # tqdm_object.set_posefix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        tqdm_object.set_postfix(train_loss=loss_meter.avg, learing_rate=lr)

    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(cfg.device) for k, v in batch.items()}

        logits_per_image, logits_per_text = model(batch["image"], batch["input_ids"])
        loss = get_loss(logits_per_image, logits_per_text)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    return loss_meter


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()


def get_loss(logits_per_image, logits_per_text):
    # labels = torch.arange(logits_per_image.size()[0]).cuda()  # 使用这个作为labels造成的损失会非常非常大

    # labels = torch.eye(logits_per_image.size()[0]).cuda()

    # loss_text = cross_entropy(logits_per_text, labels, reduction='none')
    # loss_image = cross_entropy(logits_per_image, labels, reduction='none')
    # loss =  (loss_text + loss_image) / 2.0

    # return loss.mean()


    # custom loss
    logits = (logits_per_text @ logits_per_image.t())
    image_similarity = logits_per_image @ logits_per_image.t()
    text_similarity = logits_per_image @ logits_per_text.t()

    targets = F.softmax(
        (image_similarity + text_similarity) / 2, dim=-1
    )

    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.t(), targets.t(), reduction='none')

    loss = (images_loss + texts_loss) / 2.0

    return loss.mean()


if __name__ == '__main__':
    # config
    cfg = CFG()

    # DataLoader
    train_loader, test_loader = build_loaders(cfg=cfg)

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

    # print(model)

    # parameters
    # params = [
    #     {"params": model.image_encoder.parameters(), "lr": cfg.image_encoder_lr},
    #     {"params": model.text_encoder.paramters(), "lr": cfg.text_enocder_lr}
    # ]

    # optimizer
    # optimizer = torch.optim.AdamW(params, weight_decay=0.)
    # print("The parameters of CLIP: ")
    # print(list(model.parameters()))
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.patience, factor=cfg.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        # train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        train_loss = train_epoch(model, train_loader, cfg.lr, optimizer)
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, test_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("saved the best model! ")

        # lr_scheduler.step(valid_loss.avg)
        # print("lr: ", get_lr(optimizer))
    
