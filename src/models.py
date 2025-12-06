from typing import Tuple
import torch
import torch.nn as nn
import timm


class HierarchicalClassifier(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, freeze_encoder: bool, dropout: float, super_classes: int, sub_classes: int):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        in_features = self.encoder.num_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.super_head = nn.Linear(in_features, super_classes)
        self.sub_head = nn.Linear(in_features, sub_classes)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = self.dropout(feats)
        super_logits = self.super_head(feats)
        sub_logits = self.sub_head(feats)
        return super_logits, sub_logits
