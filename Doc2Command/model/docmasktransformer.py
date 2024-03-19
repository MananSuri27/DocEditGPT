import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch.nn.init import trunc_normal_

from blocks import Block 

def init_weights(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class DocMaskTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_size,
        encoder_dim,
        num_layers,
        num_heads,
        model_dim,
        feedforward_dim,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.scale = model_dim ** -0.5

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, feedforward_dim, dropout, drop_path_rates[i]) for i in range(num_layers)]
        )

        self.class_embedding = nn.Parameter(torch.randn(1, num_classes, model_dim))
        self.project_decoder = nn.Linear(encoder_dim, model_dim)

        self.project_patch = nn.Parameter(self.scale * torch.randn(model_dim, model_dim))
        self.project_classes = nn.Parameter(self.scale * torch.randn(model_dim, model_dim))

        self.decoder_norm = nn.LayerNorm(model_dim)
        self.mask_norm = nn.LayerNorm(num_classes)

        self.apply(init_weights)
        trunc_normal_(self.class_embedding, std=0.02)



    def forward(self, x, grid_size):
        x = self.project_decoder(x)
        class_embedding = self.class_embedding.expand(x.size(0), -1, -1)
        x = torch.cat((x, class_embedding), 1)
        for block in self.blocks:
            x = block(x)
        x = self.decoder_norm(x)

        patches, class_seg_feature = x[:, : -self.num_classes], x[:, -self.num_classes :]
        patches = patches @ self.project_patch
        class_seg_feature = class_seg_feature @ self.project_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        class_seg_feature = class_seg_feature / class_seg_feature.norm(dim=-1, keepdim=True)

        masks = patches @ class_seg_feature.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(grid_size))

        return masks


