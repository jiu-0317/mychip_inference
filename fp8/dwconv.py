"""depthwise 3x3 한 레이어를 칩 연산으로 수행.

x (FP32) --입력 포맷으로 양자화--> 8비트 코드 --+
w (FP32, BN 합침) --weight 포맷으로 양자화------+--> 칩 MAC(9원소) --> 출력 포맷 8비트 --> FP32 복원
bias 덧셈과 ReLU 는 이 함수 밖(칩 밖)에서 한다.

커널 인덱스 k = 3*행 + 열, 누산 순서 0 -> 8, 패딩 위치 입력은 0.
"""
import torch
import torch.nn.functional as F
from . import chip_torch as ct
from . import quant as q


def dw_chip(x, w, stride, mode_w, mode_i, mode_o, arith="chip", return_codes=False):
    """x: (N,C,H,W) float, w: (C,1,3,3) float, mode_*: (C,) 정수 텐서 (0=E4M3, 1=E5M2).
    arith="chip" : 칩 비트 정확 연산
    arith="exact": 양자화는 같고 MAC 만 FP32 정확 연산 (칩 연산 오차 분리용)
    반환: bias 더하기 전 합 (N,C,Ho,Wo) float. return_codes=True 면 (합, 입력코드, weight코드, 출력코드)."""
    N, C, H, W = x.shape
    mi4, mw2 = mode_i.view(1, C, 1, 1), mode_w.view(C, 1)
    xc = q.encode_sel(x, mi4)                              # (N,C,H,W) 8비트 코드
    wc = q.encode_sel(w.flatten(1), mw2)                   # (C,9)
    if arith == "exact":
        s = F.conv2d(q.decode_sel(xc, mi4), q.decode_sel(wc, mw2).view(C, 1, 3, 3), None, stride, 1, 1, C)
        mo4 = mode_o.view(1, C, 1, 1)
        oc = q.encode_sel(s, mo4)
        y = q.decode_sel(oc, mo4)
    else:
        xd = ct.ir_decode(xc, mi4)                         # 9비트 E5M3
        wd = ct.ir_decode(wc, mw2)                         # (C,9)
        cols = F.unfold(xd.float(), 3, padding=1, stride=stride)   # (N, C*9, L), 채널 c 의 k 번째 = c*9+k
        L = cols.shape[-1]
        cols = cols.view(N, C, 9, L).to(torch.int32)
        acc = torch.zeros((N, C, L), dtype=torch.int32, device=x.device)
        for k in range(9):
            acc = ct.acc_add(acc, ct.fpu(wd[:, k].view(1, C, 1), cols[:, :, k, :]) << 4)
        mo3 = mode_o.view(1, C, 1)
        oc = ct.acc_out(acc, mo3, relu=False)
        Ho = (H + 2 - 3) // stride + 1
        y = q.decode_sel(oc, mo3).view(N, C, Ho, -1)
        oc = oc.view(N, C, Ho, -1)
    if return_codes:
        return y, xc, wc, oc
    return y
