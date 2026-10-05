import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, resnet18


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2 if img_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def MobileNetV3(in_channels, num_classes):
    model = mobilenet_v3_small(weights=None, num_classes=num_classes)
    c = model.features[0][0]
    model.features[0][0] = nn.Conv2d(in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    return model


def ResNet18(in_channels, num_classes):
    model = resnet18(weights=None, num_classes=num_classes)
    c = model.conv1
    model.conv1 = nn.Conv2d(in_channels, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False)
    return model


def build_model(name, in_channels, num_classes, img_size):
    if name == "lenet":
        return LeNet(in_channels, num_classes, img_size)
    if name == "mobilenetv3":
        return MobileNetV3(in_channels, num_classes)
    if name == "resnet18":
        return ResNet18(in_channels, num_classes)
    raise ValueError(name)


def model_layers(model):
    return [f"{name}.weight" for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
