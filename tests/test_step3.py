"""dw_chip 검증: 위치별로 스칼라 기준(chip_ref)과 비교. 실행: python tests/test_step3.py
커널 순서(k=3*행+열), 패딩, stride, 채널별 포맷이 맞는지 확인한다."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import torch.nn.functional as F
from fp8 import chip_ref as ref, quant as q
from fp8.dwconv import dw_chip

torch.manual_seed(0)
bad = tot = 0
worst = 0.0
for stride in (1, 2):
    for trial in range(20):
        N, C, H = 2, 6, 8
        scale = 10.0 ** torch.empty(1, C, 1, 1).uniform_(-3, 1)
        x = F.relu(torch.randn(N, C, H, H)) * scale
        w = torch.randn(C, 1, 3, 3) * 10.0 ** torch.empty(C, 1, 1, 1).uniform_(-3, 0.5)
        mw, mi, mo = (torch.randint(0, 2, (C,)) for _ in range(3))
        y, xc, wc, oc = dw_chip(x, w, stride, mw, mi, mo, return_codes=True)
        xp = F.pad(xc, (1, 1, 1, 1))                          # 패딩 = 코드 0
        Ho = oc.shape[2]
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Ho):
                        acc = 0
                        for k in range(9):
                            r, cc = divmod(k, 3)
                            xi = int(xp[n, c, i * stride + r, j * stride + cc])
                            p = ref.fpu(ref.ir_decode(int(wc[c, k]), int(mw[c])), ref.ir_decode(xi, int(mi[c])))
                            acc = ref.acc_add(acc, p << 4)
                        e = ref.acc_r(acc & 0xFFF, int(mo[c])) | (((acc >> 12) & 1) << 7)
                        bad += e != int(oc[n, c, i, j]); tot += 1
        # 참고: FP32 conv 와의 상대 차이 (양자화 오차 수준이어야 함)
        s = F.conv2d(x, w, None, stride, 1, 1, C)
        worst = max(worst, ((y - s).norm() / s.norm()).item())
print("positions %d  mismatch %d   (relative diff vs FP32 conv, worst %.3f)" % (tot, bad, worst))
print("STEP 3 dw_chip check:", "PASS" if bad == 0 else "FAIL")
sys.exit(0 if bad == 0 else 1)
