# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#https://github.com/huggingface/transformers/tree/9ee66adadb2a8d6e04e8b18a1c9ea0b57c80642e/examples/research_projects/jax-projects/hybrid_clip

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import poptorch
import numpy as np


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
        # return super().forward(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return poptorch.recomputationCheckpoint(self.layer(x))


class SerializedEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, serialization_factor=32):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Num embeddings should be divisible by the serialization factor
        assert self.vocab_size % self.serialization_factor == 0
        self.split_size = self.vocab_size // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding(self.split_size, self.embedding_dim)
            for i in range(self.serialization_factor)]
        )

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x

        return x_sum


class SerializedLinear(nn.Linear):
    def __init__(self, in_features=512, out_features=256, factor=8, bias=True,
                mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias

        if len(x.shape) > 2:
            return output.view(x.shape[0], -1, self.out_features)
    
        return output


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", SerializedLinear(d_model, d_model * 4, factor=1, bias=True)),
            ("gelu", QuickGELU()),
            ("c_proj", SerializedLinear(d_model * 4, d_model, factor=1, bias=True))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = SerializedLinear(width, output_dim, factor=8, bias=False)

        # scale the weight of self.proj
        self.proj.weight = nn.Parameter(scale * self.proj.weight.data)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:

            x = self.proj(x)

        return x


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, text_features, image_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # shape = [global_batch_size, global_batch_size]
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.Tensor(np.arange(logits_per_image.size()[0])).long()
        t_loss = self.cross_entropy(logits_per_text, labels)
        i_loss = self.cross_entropy(logits_per_image, labels)

        loss = (t_loss + i_loss) / 2.0

        return loss


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.context_length = config.context_length

        vision_heads = config.vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=config.image_resolution,
            patch_size=config.vision_patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=vision_heads,
            output_dim=config.embed_dim
        )

        self.transformer = Transformer(
            width=config.transformer_width,
            layers=config.transformer_layers,
            heads=config.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = config.vocab_size
        self.token_embedding = SerializedEmbedding(config.vocab_size, config.transformer_width, config.embedding_serialization_factor)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, config.transformer_width))
        self.ln_final = LayerNorm(config.transformer_width)

        self.text_projection = SerializedLinear(config.transformer_width, config.embed_dim, factor=2, bias=False)

        self.initialize_parameters()

        # loss
        self.loss = Loss()

    def initialize_parameters(self):
        for idx in self.token_embedding.split_embeddings:
            nn.init.normal_(idx.weight, std=0.02)

        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(-1e4)
        mask.triu_(1)  # zero out the lower diagonal

        return mask

    @property
    def dtype(self):

        return torch.float16

    def encode_image(self, image):

        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        # because the batch_first = False
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        loss = self.loss(text_features, image_features)

        return poptorch.identity_loss(loss, reduction="mean")


class PipelinedWithLoss(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = CLIP(config=config)

        # repipeline the model  参数量为：151277312
        print("---------- Device Allocation -----------")
        print("image_encoder 1 --> IPU 0")
        for index in range(5):
            layer = RecomputationCheckpoint(self.model.visual.transformer.resblocks[index])
            self.model.visual.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"image_encoder_layer{index}", ipu_id=0)

        print("image_encoder 2 --> IPU 1")
        for index in range(5, 11):
            layer = RecomputationCheckpoint(self.model.visual.transformer.resblocks[index])
            self.model.visual.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"image_encoder_layer{index}", ipu_id=1)

        print("image_encoder layer[11] --> IPU 2")
        layer = RecomputationCheckpoint(self.model.visual.transformer.resblocks[11])
        self.model.visual.transformer.resblocks[11] = poptorch.BeginBlock(layer, "image_encoder_layer11", ipu_id=2)

        print("token_embedding --> IPU 2")
        self.model.token_embedding = poptorch.BeginBlock(self.model.token_embedding, "embedding", ipu_id=2)
        print("text_enocder --> IPU 3")
        for index in range(0, 12):
            layer = RecomputationCheckpoint(self.model.transformer.resblocks[index])
            self.model.transformer.resblocks[index] = poptorch.BeginBlock(layer, f"text_encoder_layer{index}", ipu_id=3)

        print("loss --> IPU 3")
        self.model.loss = poptorch.BeginBlock(self.model.loss, f"loss", ipu_id=3)
        print("---------------------------------------")


    def forward(self, image, text):
        loss = self.model(image, text)
        return loss

    def load_from(self, weights):
        self.model.load_from(weights)
