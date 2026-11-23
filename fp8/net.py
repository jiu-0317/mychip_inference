"""BN 을 합친 MobileNetV1 에서 depthwise 13개만 칩 연산으로 바꾼 모델. 나머지는 FP32."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dwconv import dw_chip


def uniform_cfg(model, w, i, o):
    """모든 레이어·채널에 같은 포맷. w/i/o: 0=E4M3, 1=E5M2"""
    cfg = []
    for b in model.blocks:
        C = b.dw.weight.shape[0]
        cfg.append({"W": torch.full((C,), w, dtype=torch.long),
                    "I": torch.full((C,), i, dtype=torch.long),
                    "O": torch.full((C,), o, dtype=torch.long)})
    return cfg


class FP8Net(nn.Module):
    def __init__(self, folded_model, cfg, arith="chip"):
        super().__init__()
        self.m = folded_model
        self.cfg = cfg
        self.arith = arith

    def forward(self, x):
        m = self.m
        x = F.relu(m.conv1(x))
        for b, c in zip(m.blocks, self.cfg):
            d = x.device
            s = dw_chip(x, b.dw.weight, b.dw.stride[0], c["W"].to(d), c["I"].to(d), c["O"].to(d), self.arith)
            x = F.relu(s + b.dw.bias.view(1, -1, 1, 1))       # bias, ReLU 는 칩 밖
            x = F.relu(b.pw(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return m.fc(x)
