"""CIFAR 용 MobileNetV1 (ReLU). depthwise 3x3 은 13개.

블록: dw(3x3, groups=C) -> bn1 -> ReLU -> pw(1x1) -> bn2 -> ReLU
folded=True 이면 BN 이 없고 conv 가 bias 를 가진다 (fold_bn.py 가 만든 가중치용).
"""
import torch.nn as nn
import torch.nn.functional as F

CFG = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]


class Block(nn.Module):
    def __init__(self, cin, cout, stride, folded):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=folded)
        self.bn1 = nn.Identity() if folded else nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, 1, 0, bias=folded)
        self.bn2 = nn.Identity() if folded else nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, folded=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=folded)
        self.bn1 = nn.Identity() if folded else nn.BatchNorm2d(32)
        blocks, cin = [], 32
        for c in CFG:
            cout, stride = (c, 1) if isinstance(c, int) else c
            blocks.append(Block(cin, cout, stride, folded))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
