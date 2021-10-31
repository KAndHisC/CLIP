import torch 
import numpy as np
from torch import nn, F

torch_cross_entropy_func = torch.nn.CrossEntropyLoss()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()

# labels = torch.Tensor(np.arange(logits_per_image.size()[0])).long().to(cfg.device)
# labels = torch.eye(logits_per_image.size()[0]).cuda()
def paper_loss(logits_per_image, logits_per_text, labels):
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
