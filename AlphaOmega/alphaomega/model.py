import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_bn_relu(c_in, c_out, k=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, padding=0 if k == 1 else dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(self, c_in, c_out, rates):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(c_in, c_out)] + [conv_bn_relu(c_in, c_out, 3, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_relu(c_in, c_out))
        self.project = nn.Sequential(conv_bn_relu(c_out * (len(rates) + 2), c_out), nn.Dropout(0.1))

    def forward(self, x):
        out = [branch(x) for branch in self.branches]
        out.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(out, dim=1))


class Decoder(nn.Module):
    def __init__(self, c_low, c_low_out, c_high, d):
        super().__init__()
        self.reduce = conv_bn_relu(c_low, c_low_out)
        self.fuse = nn.Sequential(conv_bn_relu(c_low_out + c_high, d, 3), conv_bn_relu(d, d, 3))

    def forward(self, low, high):
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.reduce(low), high], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, pretrained, output_stride, aspp_rates, aspp_channels, low_level_channels, d):
        super().__init__()
        dilation = {16: [False, False, True], 8: [False, True, True]}[output_stride]
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained == "imagenet" else None
        backbone = torchvision.models.resnet50(weights=weights, replace_stride_with_dilation=dilation)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.aspp = ASPP(2048, aspp_channels, aspp_rates)
        self.decoder = Decoder(256, low_level_channels, aspp_channels, d)
        self.classifier = nn.Conv2d(d, num_classes, 1)

    @property
    def d(self):
        return self.classifier.in_channels

    @property
    def num_classes(self):
        return self.classifier.out_channels

    def features(self, x):
        b = self.backbone
        low = b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x)))))
        high = b.layer4(b.layer3(b.layer2(low)))
        return self.decoder(low, self.aspp(high))

    def forward(self, x):
        return F.interpolate(self.classifier(self.features(x)), size=x.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def add_classes(self, C_t):
        old = self.classifier
        new = nn.Conv2d(old.in_channels, old.out_channels + C_t, 1).to(old.weight.device)
        new.weight[: old.out_channels].copy_(old.weight)
        new.bias[: old.out_channels].copy_(old.bias)
        self.classifier = new


def build_model(cfg, num_classes):
    return DeepLabV3Plus(
        num_classes,
        cfg.pretrained,
        cfg.output_stride,
        cfg.aspp_rates,
        cfg.aspp_channels,
        cfg.low_level_channels,
        cfg.d,
    )


def head_logits(f, W, b):
    return F.conv2d(f, W.reshape(W.shape[0], -1, 1, 1), b)


def group_of(name, prefixes):
    for l, names in prefixes.items():
        if any(name == q or name.startswith(q + ".") for q in names):
            return l
    return None


def layer_groups(model, prefixes):
    groups = {l: [] for l in prefixes}
    for name, param in model.named_parameters():
        groups[group_of(name, prefixes)].append((name, param))
    return groups


def model_like(model, Theta):
    clone = copy.deepcopy(model)
    clone.classifier = nn.Conv2d(clone.d, Theta["classifier.weight"].shape[0], 1)
    clone.load_state_dict(Theta)
    return clone.to(next(model.parameters()).device)
