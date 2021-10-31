
import torch

from clip.model import CLIP

import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import numpy as np

import wandb

if __name__ == '__main__':

    
    
    
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=cfg.patience, factor=cfg.factor
    # )
    

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
    
