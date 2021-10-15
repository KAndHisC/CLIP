


import clip
from clip_ipu import RecomputationCheckpoint, recomputationCheckpointWrapper
from torch import nn
import poptorch


class PipelinedWithLossForCLIP(nn.Module):
    def __init__(self, model_name, training=False):
        super().__init__()

        self.loss_fuc = nn.CrossEntropyLoss()

        # tmp test using a non-custom loss
        self.loss = nn.CrossEntropyLoss()

        model, preprocess = clip.load(model_name)
        self.model = model

        # preprocess
        self.preprocess = preprocess
        self.texts = None
        
        # Vision
        self.model.visual.conv1 = poptorch.BeginBlock(RecomputationCheckpoint(self.model.visual.conv1), "conv1", ipu_id=0)
        self.model.visual.ln_pre = poptorch.BeginBlock(self.model.visual.ln_pre, "ln_pre", ipu_id=0)
        if training:
            layers_on_ipu = [0,0,0,0,0,0,1,1,1,1,1,1]
        else:
            layers_on_ipu = [0,0,0,1,1,1,2,2,2,3,3,3]
        
        for index in range(len(self.model.visual.transformer.resblocks)):
            layer = self.model.visual.transformer.resblocks[index]
            layer = RecomputationCheckpoint(layer) 
            self.model.visual.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"visual_transformer{index}", ipu_id=layers_on_ipu[index])
            print(f"visual_transformer layer {index:<2} --> IPU {layers_on_ipu[index]}") 

        if training:
            self.model.visual.ln_post = poptorch.BeginBlock(self.model.visual.ln_post, "ln_post", ipu_id=2)
        else:
            self.model.visual.ln_post = poptorch.BeginBlock(self.model.visual.ln_post, "ln_post", ipu_id=3)

        if not training:
            return

        # # CLIP
        self.model.token_embedding = poptorch.BeginBlock(RecomputationCheckpoint(self.model.token_embedding), "token_embedding", ipu_id=2)

        # Text
        layers_on_ipu = [2,2,3,3,3,3,3,3,3,3,3,3]
        for index in range(len(self.model.transformer.resblocks)):
            layer = self.model.transformer.resblocks[index]
            layer = RecomputationCheckpoint(layer) 
            self.model.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"text_transformer{index}", ipu_id=layers_on_ipu[index])
            print(f"text_transformer layer {index:<2} --> IPU {layers_on_ipu[index]}") 
        
        # CLIP
        self.model.ln_final = poptorch.BeginBlock(self.model.ln_final, "ln_final", ipu_id=3)
        
        

    @recomputationCheckpointWrapper
    def forward(
        self,
        images, texts = None, labels = None
    ):
        # logits_per_image, logits_per_text = self.model(images, texts)
        image_features = self.model.encode_image(images)
        if self.training:
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = self.categories

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logit_scale * text_features @ image_features.t()

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
    
    def setTextAsCategories(self, texts):
        text_features = self.model.encode_text(texts)
        categories = text_features / text_features.norm(dim=-1, keepdim=True)
        self.categories = categories.detach()

