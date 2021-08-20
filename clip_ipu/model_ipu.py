

import torch
import clip
from torch import nn
import poptorch

class PipelinedWithLossForCLIP(nn.Module):
    def __init__(self, model_name, jit=False):
        super().__init__()

        device = "cpu"
        self.loss_fuc = nn.CrossEntropyLoss()

        # tmp test using a non-custom loss
        self.loss = nn.CrossEntropyLoss()
        if jit:
            model, preprocess = clip.load(model_name, device=device, jit=True)
        else:
            model, preprocess = clip.load(model_name, device=device)
        self.model = model
        self.preprocess = preprocess


    def forward(
        self,
        images, texts, labels = None
    ):
        # logits_per_image, logits_per_text = self.model(images, texts)
        logits_per_image, _ = self.model(images, texts)
        probs = logits_per_image.softmax(dim=-1)
        
        # loss = self.custom_loss(logits_per_image, logits_per_text, labels)
        # temporary using a sample loss to test
        if self.training:
            return probs, self.loss(logits_per_image, labels)
        return probs

    def custom_loss(self, logits_per_image, logits_per_text , labels):
        loss_i = self.loss_fuc(logits_per_image, labels)
        loss_t = self.loss_fuc(logits_per_text, labels)
        return poptorch.identity_loss( ((loss_i + loss_t)/2.0).mean() , reduction='none')
