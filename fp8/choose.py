"""채널 단위 포맷 선택 방법 3가지. 각 함수는 cfg(레이어별 {"W","I","O"}: (C,) 텐서, 0=E4M3 1=E5M2)를 돌려준다.

A  범위 규칙  : 0 이 아닌 값 중 E4M3 범위(2^-6 ~ 448) 밖 비율이 TAU 를 넘으면 E5M2
B  오차 최소  : 두 포맷으로 양자화해 제곱오차가 작은 쪽 (동률이면 E4M3)
C  칩 인지 탐색: 채널마다 (W,I,O) 8조합을 칩 모델로 계산해 FP32 채널 출력(bias, ReLU 뒤)과의
                 제곱오차가 가장 작은 조합. 앞 레이어부터 순서대로 정하고, 뒤 레이어는 이미
                 양자화된 앞 레이어 출력을 입력으로 쓴다.
A, B 의 입력/합 통계는 FP32 모델의 값으로 구한다. W, I, O 는 서로 독립으로 고른다.
"""
import itertools
import torch
import torch.nn.functional as F
from . import quant as q
from .dwconv import dw_chip

TAU = 0.10
COMBOS = list(itertools.product((q.E4M3, q.E5M2), repeat=3))     # (W, I, O), 첫 번째가 전부 E4M3


def _stats(v, dims):
    v = v.double()
    a = v.abs()
    nz = v != 0
    out = {"nz": nz.double().sum(dims),
           "oor": (nz & ((a < q.MIN_NORMAL[q.E4M3]) | (a > q.MAX_NORMAL[q.E4M3]))).double().sum(dims)}
    for m in (q.E4M3, q.E5M2):
        qv = q.decode(q.encode(v.float(), m), m).double()
        out["err%d" % m] = ((v - qv) ** 2).sum(dims)
    return out


def _add(acc, new):
    for k, v in new.items():
        acc[k] = acc.get(k, 0) + v


def collect_fp32_stats(model, calib_batches):
    """레이어별 (W, I, O) 채널 통계. FP32 순전파 한 번."""
    L = len(model.blocks)
    I = [dict() for _ in range(L)]
    O = [dict() for _ in range(L)]

    def make_hook(i):
        def hook(mod, inp):
            x = inp[0]
            _add(I[i], _stats(x, (0, 2, 3)))
            _add(O[i], _stats(F.conv2d(x, mod.weight, None, mod.stride, mod.padding, 1, mod.groups), (0, 2, 3)))
        return hook

    hooks = [b.dw.register_forward_pre_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        for x in calib_batches:
            model(x)
    for h in hooks:
        h.remove()
    W = [_stats(b.dw.weight.detach().flatten(1), (1,)) for b in model.blocks]
    return W, I, O


def _rule_a(s):
    return (s["oor"] > TAU * s["nz"]).long().cpu()


def _rule_b(s):
    return (s["err1"] < s["err0"]).long().cpu()


def choose_a(stats):
    W, I, O = stats
    return [{"W": _rule_a(w), "I": _rule_a(i), "O": _rule_a(o)} for w, i, o in zip(W, I, O)]


def choose_b(stats):
    W, I, O = stats
    return [{"W": _rule_b(w), "I": _rule_b(i), "O": _rule_b(o)} for w, i, o in zip(W, I, O)]


@torch.no_grad()
def choose_c(model, calib_batches):
    dev = calib_batches[0].device
    ref = [F.relu(model.conv1(x)) for x in calib_batches]        # FP32 경로
    cur = [r.clone() for r in ref]                               # 양자화 경로
    cfg = []
    for b in model.blocks:
        C = b.dw.weight.shape[0]
        bias = b.dw.bias.view(1, -1, 1, 1)
        stride = b.dw.stride[0]
        target = [F.relu(b.dw(r)) for r in ref]
        err = torch.zeros(len(COMBOS), C, dtype=torch.float64, device=dev)
        for ci, (w, i, o) in enumerate(COMBOS):
            mw, mi, mo = (torch.full((C,), m, dtype=torch.long, device=dev) for m in (w, i, o))
            for x, t in zip(cur, target):
                y = F.relu(dw_chip(x, b.dw.weight, stride, mw, mi, mo) + bias)
                err[ci] += ((y - t).double() ** 2).sum((0, 2, 3))
        best = err.argmin(0)                                     # 동률이면 앞 조합(E4M3 쪽)
        comb = torch.tensor(COMBOS, dtype=torch.long, device=dev)[best]      # (C,3)
        c = {"W": comb[:, 0].contiguous(), "I": comb[:, 1].contiguous(), "O": comb[:, 2].contiguous()}
        cur = [F.relu(b.pw(F.relu(dw_chip(x, b.dw.weight, stride, c["W"], c["I"], c["O"]) + bias))) for x in cur]
        ref = [F.relu(b.pw(t)) for t in target]
        cfg.append({k: v.cpu() for k, v in c.items()})
    return cfg


def live_mask(model):
    """weight 9개가 두 포맷 모두에서 0 이 되는(꺼진) 채널을 뺀 마스크. 레이어별 (C,) bool."""
    out = []
    for b in model.blocks:
        w = b.dw.weight.detach().flatten(1)
        out.append((q.encode(w, q.E5M2) & 0x7F).ne(0).any(1).cpu())
    return out
