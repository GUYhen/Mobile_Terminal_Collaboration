from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, ResNet101_Weights, mobilenet_v2, resnet101
from torchvision.models._utils import IntermediateLayerGetter


class SeparableConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, padding=dilation, dilation=dilation, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_ch: int, rates: Sequence[int], out_ch: int = 256):
        super().__init__()
        branches = [nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))]
        branches += [SeparableConv(in_ch, out_ch, r) for r in rates]
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class Decoder(nn.Module):
    def __init__(self, low_ch: int, num_classes: int, aspp_ch: int = 256):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(SeparableConv(aspp_ch + 48, 256), SeparableConv(256, 256))
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.fuse(torch.cat([self.reduce(low), high], dim=1)))


class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        features = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.low = features[:4]
        self.high = features[4:]

    def forward(self, x: torch.Tensor) -> dict:
        low = self.low(x)
        return {"low": low, "out": self.high(low)}


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int, backbone: str, aspp_rates: Sequence[int]):
        super().__init__()
        if backbone == "resnet101":
            net = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
            self.backbone = IntermediateLayerGetter(net, return_layers={"layer1": "low", "layer4": "out"})
            low_ch, high_ch = 256, 2048
        elif backbone == "mobilenet_v2":
            self.backbone = MobileNetV2Backbone()
            low_ch, high_ch = 24, 1280
        else:
            raise ValueError(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.aspp = ASPP(high_ch, aspp_rates)
        self.decoder = Decoder(low_ch, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def federated_parameters(self) -> List[nn.Parameter]:
        return list(self.aspp.parameters()) + list(self.decoder.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decoder(feats["low"], self.aspp(feats["out"]))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
