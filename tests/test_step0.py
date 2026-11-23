"""단계 0: 비트 정확 모델 검증.  실행:  python tests/test_step0.py  (inference 폴더에서)

T1  알려진 값: W=I=1.0(E4M3 0x38) x9 -> 0x51 (9.0)   [smoke_test T5 와 동일]
T2  원본 모델(chip_ref, ReLU 있음) vs torch 모델(relu=True): 입력 mode 통일
T3  원소별 mode 혼합 + ReLU 제거: 스칼라 기준 vs torch 모델
T4  실물 칩용 2회 실행(weight 부호 반전) 복원값 == ReLU 제거 모델
T5  양자화기: 표현값 왕복, 무작위 값의 최근접/동률/포화 규칙
"""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from fp8 import chip_ref as ref, chip_torch as ct, quant as q

N_RANDOM = 1_000_000      # T3, T4 무작위 케이스 수
N_SCALAR = 200_000        # T2 케이스 수
torch.manual_seed(0); random.seed(0)


def ref_mac(W, I, mw, mi, mo, relu):
    """chip_ref 함수만으로 만든 기준. 원소별 mode, ReLU on/off."""
    acc = 0
    for k in range(9):
        p = ref.fpu(ref.ir_decode(W[k], mw[k]), ref.ir_decode(I[k], mi[k]))
        acc = ref.acc_add(acc, p << 4)
    if relu:
        return ref.acc_r(acc, mo)
    # ReLU 제거: 부호만 떼고 변환한 뒤 부호 비트를 다시 붙인다 (ACC_R 은 부호와 무관하게 크기를 변환)
    return ref.acc_r(acc & 0xFFF, mo) | (((acc >> 12) & 1) << 7)


def gen(n, wide):
    """wide=True: 0~255 전 범위 / False: 결과가 0·포화로 쏠리지 않게 지수를 중간 대역으로"""
    if wide:
        W = torch.randint(0, 256, (n, 9)); I = torch.randint(0, 256, (n, 9))
        mw = torch.randint(0, 2, (n, 9)); mi = torch.randint(0, 2, (n, 9))
    else:
        mw = torch.randint(0, 2, (n, 9)); mi = torch.randint(0, 2, (n, 9))
        def mk(mode):
            s = torch.randint(0, 2, (n, 9))
            e4 = torch.randint(4, 11, (n, 9)); m3 = torch.randint(0, 8, (n, 9))
            e5 = torch.randint(12, 19, (n, 9)); m2 = torch.randint(0, 4, (n, 9))
            c = torch.where(mode == 0, (e4 << 3) | m3, (e5 << 2) | m2)
            z = torch.rand(n, 9) < 0.1
            return torch.where(z, torch.zeros_like(c), c) | (s << 7)
        W, I = mk(mw), mk(mi)
    mo = torch.randint(0, 2, (n,))
    return W, I, mw, mi, mo


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    return ok


def main():
    allok = True

    # T1
    one = torch.full((1, 9), 0x38)
    r = ct.mac(one, one, 0, 0, 0, relu=True).item()
    allok &= check("T1 known value 9.0", r == 0x51, "got 0x%02X" % r)

    # T2: 원본 ref.mac (mode 통일, ReLU 있음)
    bad = 0; n2 = N_SCALAR
    for wide in (True, False):
        W, I, _, _, mo = gen(n2 // 2, wide)
        mi_ = torch.randint(0, 2, (n2 // 2,))
        out = ct.mac(W, I, mi_[:, None], mi_[:, None], mo, relu=True)
        Wl, Il, ml, ol, rl = W.tolist(), I.tolist(), mi_.tolist(), mo.tolist(), out.tolist()
        for j in range(len(Wl)):
            if ref.mac(Wl[j], Il[j], ml[j], ol[j]) != rl[j]:
                bad += 1
    allok &= check("T2 original model (ReLU on), %d cases" % n2, bad == 0, "mismatch=%d" % bad)

    # T3: 원소별 mode 혼합, ReLU 제거
    bad = 0; n3 = N_RANDOM; nonzero = 0; neg = 0
    for wide in (True, False):
        W, I, mw, mi, mo = gen(n3 // 2, wide)
        out = ct.mac(W, I, mw, mi, mo, relu=False)
        Wl, Il, wl, il, ol, rl = W.tolist(), I.tolist(), mw.tolist(), mi.tolist(), mo.tolist(), out.tolist()
        for j in range(len(Wl)):
            e = ref_mac(Wl[j], Il[j], wl[j], il[j], ol[j], relu=False)
            bad += e != rl[j]
            nonzero += (e & 0x7F) != 0
            neg += (e >> 7) & 1
    allok &= check("T3 mixed modes, ReLU off, %d cases" % n3, bad == 0,
                   "mismatch=%d (nonzero=%d, negative=%d)" % (bad, nonzero, neg))

    # T4: 2회 실행 복원 (대량, torch)
    bad = 0; tot = 0
    for wide in (True, False):
        W, I, mw, mi, mo = gen(N_RANDOM // 2, wide)
        signed = ct.mac(W, I, mw, mi, mo, relu=False)
        r1 = ct.mac(W, I, mw, mi, mo, relu=True)
        r2 = ct.mac(W ^ 0x80, I, mw, mi, mo, relu=True)
        rec = torch.where(r1 != 0, r1, torch.where(r2 != 0, r2 | 0x80, torch.zeros_like(r1)))
        canon = torch.where((signed & 0x7F) == 0, torch.zeros_like(signed), signed)   # -0 -> +0
        # NaN 결과(입력에 E5M2 exp=31 이 있을 때만)는 부호가 없으므로 제외
        e5nan = (((W >> 2) & 0x1F) == 31) & (mw == 1) | (((I >> 2) & 0x1F) == 31) & (mi == 1)
        valid = ~e5nan.any(dim=1)
        bad += ((rec != canon) & valid).sum().item(); tot += valid.sum().item()
    allok &= check("T4 two-pass sign-flip == ReLU-off, %d cases" % tot, bad == 0, "mismatch=%d" % bad)

    # T5: 양자화기
    ok5 = True
    for mode in (q.E4M3, q.E5M2):
        vals, codes = q._TAB[mode]
        for sgn in (1.0, -1.0):
            x = (vals * sgn).to(torch.float32)
            c = q.encode(x, mode)
            ok5 &= bool((q.decode(c, mode).double() == x.double()).all())
        # 무작위: 전수 최근접과 비교
        x = torch.exp(torch.empty(200_000).uniform_(-14, 14)).double() * torch.where(torch.rand(200_000) < 0.5, -1.0, 1.0)
        x = x.to(torch.float32)
        got = q.decode(q.encode(x, mode), mode).double()
        d = (x.double().abs()[:, None] - vals[None, :]).abs()
        best = d.min(dim=1).values
        ok5 &= bool(((got.abs() - x.double().abs()).abs() <= best + 0).all())
        ok5 &= bool((torch.sign(got) * torch.sign(x.double()) >= 0).all())
        # 동률: 중간값은 짝수 코드로
        mids = ((vals[:-1] + vals[1:]) / 2).to(torch.float32)
        exact = mids.double() == (vals[:-1] + vals[1:]) / 2
        cm = q.encode(mids, mode)[exact]
        ok5 &= bool(((cm & 1) == 0).all())
        ok5 &= q.encode(torch.tensor([1e30]), mode).item() == codes[-1].item()
        ok5 &= q.encode(torch.tensor([-1e30]), mode).item() == (codes[-1].item() | 0x80)
        ok5 &= q.encode(torch.tensor([0.0]), mode).item() == 0
    ok5 &= q.MAX_NORMAL[q.E4M3] == 448.0 and q.MAX_NORMAL[q.E5M2] == 57344.0
    allok &= check("T5 quantizer rules", ok5)

    print("\nSTEP 0:", "ALL PASS" if allok else "FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
