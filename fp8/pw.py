"""pointwise(1x1) 를 FP8 로 양자화하는 실험용 코드. 칩 연산이 아니라 '양자화 + 정확한 덧셈'이다.

  입력  x  : 입력 채널별 포맷으로 양자화
  weight w : 출력 채널별 포맷으로 양자화 (출력 채널 하나 = weight Cin 개 묶음)
  합       : FP32 로 정확히 계산한 뒤 출력 채널별 포맷으로 양자화
bias 덧셈과 ReLU 는 밖에서 한다. depthwise / 첫 conv / FC 는 FP32 그대로.

포맷 선택 (단위: layer = 레이어당 W/I/O 하나씩, channel = 채널마다)
  B  오차 최소 : FP32 값 기준, 두 포맷으로 양자화해 제곱오차가 작은 쪽
  C  출력 기준 : 앞 레이어부터 순서대로, 양자화된 경로의 값으로 결정
       layer   : (W,I,O) 8조합 중 레이어 출력(bias, ReLU 뒤)이 FP32 와 가장 가까운 조합
       channel : I 는 입력 채널별 양자화 오차가 작은 쪽, 그다음 출력 채널마다 (W,O) 4조합 중
                 채널 출력이 FP32 와 가장 가까운 조합
"""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import quant as q
from .choose import _stats, _add

E4, E5 = q.E4M3, q.E5M2


def pw_fp8(x, w, mode_w, mode_i, mode_o):
    """x: (N,Cin,H,W), w: (Cout,Cin,1,1). 반환: bias 더하기 전 합(양자화됨)."""
    Cout, Cin = w.shape[0], w.shape[1]
    mi = mode_i.view(1, Cin, 1, 1)
    mw = mode_w.view(Cout, 1)
    mo = mode_o.view(1, Cout, 1, 1)
    xq = q.decode_sel(q.encode_sel(x, mi), mi)
    wq = q.decode_sel(q.encode_sel(w.flatten(1), mw), mw).view(Cout, Cin, 1, 1)
    s = F.conv2d(xq, wq)
    return q.decode_sel(q.encode_sel(s, mo), mo)


def uniform_cfg(model, w, i, o):
    cfg = []
    for b in model.blocks:
        Cout, Cin = b.pw.weight.shape[0], b.pw.weight.shape[1]
        cfg.append({"W": torch.full((Cout,), w, dtype=torch.long),
                    "I": torch.full((Cin,), i, dtype=torch.long),
                    "O": torch.full((Cout,), o, dtype=torch.long)})
    return cfg


class PWNet(nn.Module):
    """pointwise 13개만 FP8, 나머지는 FP32."""
    def __init__(self, folded_model, cfg):
        super().__init__()
        self.m, self.cfg = folded_model, cfg

    def forward(self, x):
        m = self.m
        x = F.relu(m.conv1(x))
        for b, c in zip(m.blocks, self.cfg):
            d = x.device
            x = F.relu(b.dw(x))
            s = pw_fp8(x, b.pw.weight, c["W"].to(d), c["I"].to(d), c["O"].to(d))
            x = F.relu(s + b.pw.bias.view(1, -1, 1, 1))
        return m.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def collect_stats(model, calib_batches):
    L = len(model.blocks)
    I = [dict() for _ in range(L)]
    O = [dict() for _ in range(L)]

    def make_hook(i):
        def hook(mod, inp):
            x = inp[0]
            _add(I[i], _stats(x, (0, 2, 3)))
            _add(O[i], _stats(F.conv2d(x, mod.weight), (0, 2, 3)))
        return hook

    hooks = [b.pw.register_forward_pre_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        for x in calib_batches:
            model(x)
    for h in hooks:
        h.remove()
    W = [_stats(b.pw.weight.detach().flatten(1), (1,)) for b in model.blocks]
    return W, I, O


def choose_b(stats, level):
    cfg = []
    for w, i, o in zip(*stats):
        c = {}
        for k, s in (("W", w), ("I", i), ("O", o)):
            if level == "layer":
                pick = int(s["err1"].sum() < s["err0"].sum())
                c[k] = torch.full((s["err0"].numel(),), pick, dtype=torch.long)
            else:
                c[k] = (s["err1"] < s["err0"]).long().cpu()
        cfg.append(c)
    return cfg


@torch.no_grad()
def choose_c(model, calib_batches, level):
    dev = calib_batches[0].device
    ref = [F.relu(model.conv1(x)) for x in calib_batches]
    cur = [r.clone() for r in ref]
    cfg = []

    def full(n, v):
        return torch.full((n,), v, dtype=torch.long, device=dev)

    for b in model.blocks:
        Cout, Cin = b.pw.weight.shape[0], b.pw.weight.shape[1]
        w = b.pw.weight
        bias = b.pw.bias.view(1, -1, 1, 1)
        ref = [F.relu(b.dw(r)) for r in ref]
        cur = [F.relu(b.dw(x)) for x in cur]
        target = [F.relu(b.pw(r)) for r in ref]
        if level == "layer":
            combos = list(itertools.product((E4, E5), repeat=3))
            err = torch.zeros(len(combos), dtype=torch.float64, device=dev)
            for ci, (cw, cin, co) in enumerate(combos):
                for x, t in zip(cur, target):
                    y = F.relu(pw_fp8(x, w, full(Cout, cw), full(Cin, cin), full(Cout, co)) + bias)
                    err[ci] += ((y - t).double() ** 2).sum()
            cw, cin, co = combos[int(err.argmin())]
            c = {"W": full(Cout, cw), "I": full(Cin, cin), "O": full(Cout, co)}
        else:
            si = {}
            for x in cur:
                _add(si, _stats(x, (0, 2, 3)))
            mi = (si["err1"] < si["err0"]).long()
            combos = list(itertools.product((E4, E5), repeat=2))
            err = torch.zeros(len(combos), Cout, dtype=torch.float64, device=dev)
            for ci, (cw, co) in enumerate(combos):
                for x, t in zip(cur, target):
                    y = F.relu(pw_fp8(x, w, full(Cout, cw), mi, full(Cout, co)) + bias)
                    err[ci] += ((y - t).double() ** 2).sum((0, 2, 3))
            comb = torch.tensor(combos, dtype=torch.long, device=dev)[err.argmin(0)]
            c = {"W": comb[:, 0].contiguous(), "I": mi, "O": comb[:, 1].contiguous()}
        cur = [F.relu(pw_fp8(x, w, c["W"], c["I"], c["O"]) + bias) for x in cur]
        ref = target
        cfg.append({k: v.cpu() for k, v in c.items()})
    return cfg
