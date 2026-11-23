"""MyChip_FP_MAC 데이터패스의 비트 정확 모델 (PyTorch, 텐서 단위).

기준: fp8/chip_ref.py (= mpw_test/smoke_gen/model.py, RTL/GLS 와 일치 검증됨).
RTL 과의 차이는 ACC_R 의 ReLU 를 끌 수 있다는 것 하나뿐이다 (relu=False 가 FPGA 용 RTL).

모든 값은 정수 텐서(int32)로 다룬다.
  8비트 코드  : SPI 로 주고받는 E4M3 / E5M2 바이트
  9비트 E5M3 : IR 디코딩 후 레지스터 파일에 저장되는 값
  13비트 E5M7: 누산기 값
mode: 0 = E4M3, 1 = E5M2 (원소마다 다른 값을 줄 수 있다)
누산 순서: 마지막 차원의 인덱스 0 -> 8 (칩의 addr 0 -> 8)
"""
import torch


def ir_decode(b, mode):
    """8비트 코드 + mode -> 9비트 E5M3 (IR.v)"""
    b = b.to(torch.int32)
    mode = torch.as_tensor(mode, dtype=torch.int32, device=b.device)
    s = (b >> 7) & 1
    e4 = (b >> 3) & 0xF
    d4 = torch.where(e4 == 0, s << 8, (s << 8) | ((e4 + 8) << 3) | (b & 7))
    e5 = (b >> 2) & 0x1F
    d5 = torch.where(e5 == 0, s << 8, (b << 1) & 0x1FF)
    return torch.where(mode == 0, d4, d5)


def fpu(w, i):
    """9비트 E5M3 x 9비트 E5M3 -> 9비트 E5M3 (FPU.v)"""
    so = ((w >> 8) ^ (i >> 8)) & 1
    ew, ei = (w >> 3) & 0x1F, (i >> 3) & 0x1F
    mw, mi = w & 7, i & 7
    is_zero = ((ew == 0) & (mw == 0)) | ((ei == 0) & (mi == 0))
    is_ni = (ew == 31) | (ei == 31)
    mp = (8 | mw) * (8 | mi)                      # 8비트
    top = (mp >> 7) & 1
    en = ew + ei - 15 + top
    mn = torch.where(top == 1, mp, (mp << 1) & 0xFF)
    cand = (mn >> 4) & 7
    guard = (mn >> 3) & 1
    rnd = (mn >> 2) & 1
    sticky = ((mn & 3) != 0).to(torch.int32)
    ru = guard & (rnd | sticky | (cand & 1))
    mr = cand + ru
    r_ovf = (mr >> 3) & 1
    mf = torch.where(r_ovf == 1, torch.zeros_like(mr), mr & 7)
    ef = en + r_ovf
    sgn = so << 8
    out = sgn | ((ef & 0x1F) << 3) | mf
    out = torch.where(ef <= 0, sgn, out)
    out = torch.where(ef > 30, sgn | (30 << 3) | 7, out)
    out = torch.where(is_zero, sgn, out)
    out = torch.where(is_ni, sgn | (31 << 3) | 1, out)
    return out


def acc_add(a, b):
    """13비트 E5M7 + 13비트 E5M7 -> 13비트 E5M7 (ACC_adder.v, 자리 맞춤은 버림)"""
    sa, sb = (a >> 12) & 1, (b >> 12) & 1
    ea, eb = (a >> 7) & 0x1F, (b >> 7) & 0x1F
    ma, mb = a & 0x7F, b & 0x7F
    a_ni, b_ni = ea == 31, eb == 31
    a_z = (ea == 0) & (ma == 0)
    b_z = (eb == 0) & (mb == 0)
    fa, fb = 0x80 | ma, 0x80 | mb
    a_larger = (ea > eb) | ((ea == eb) & (ma >= mb))
    el = torch.where(a_larger, ea, eb)
    ed = torch.where(a_larger, ea - eb, eb - ea)
    sl = torch.where(a_larger, sa, sb)
    ss = torch.where(a_larger, sb, sa)
    ml = torch.where(a_larger, fa, fb)
    msm = torch.where(a_larger, fb, fa)
    shifted = torch.where(ed >= 8, torch.zeros_like(msm), msm >> ed.clamp(max=8))
    same = sl == ss
    opb = torch.where(same, shifted, (~shifted) & 0x1FF)
    raw = (ml + opb + (~same).to(torch.int32)) & 0x1FF
    carry = (raw >> 8) & 1
    raw8 = raw & 0xFF
    sh = torch.zeros_like(raw)
    for k in range(8):                            # 가장 높은 1의 위치 k -> shift = 7-k
        sh = torch.where(raw8 >= (1 << k), torch.full_like(sh, 7 - k), sh)
    en = torch.where(carry == 1, el + 1, el - sh)
    mn = torch.where(carry == 1, (raw >> 1) & 0x7F, (raw8 << sh) & 0x7F)
    none = (carry == 0) & (raw8 == 0)
    en = torch.where(none, torch.zeros_like(en), en)
    mn = torch.where(none, torch.zeros_like(mn), mn)
    sgn = sl << 12
    out = sgn | ((en & 0x1F) << 7) | mn
    out = torch.where(en <= 0, sgn, out)
    out = torch.where(en >= 31, sgn | (30 << 7) | 0x7F, out)
    out = torch.where(b_z, a, out)
    out = torch.where(a_z, b, out)
    out = torch.where(a_z & b_z, torch.zeros_like(out), out)
    out = torch.where(a_ni | b_ni, torch.full_like(out, (31 << 7) | 1), out)
    return out


def acc_out(d, mode, relu=False):
    """13비트 E5M7 -> 8비트 코드 (ACC_R.v). relu=True 가 테이프아웃된 칩."""
    mode = torch.as_tensor(mode, dtype=torch.int32, device=d.device)
    s = (d >> 12) & 1
    er = (d >> 7) & 0x1F
    m = d & 0x7F
    is_nan = (er == 31) & (m != 0)
    is_zero = (er == 0) & (m == 0)
    sgn = s << 7
    # E4M3
    t0 = (m >> 4) & 7
    ru0 = ((m >> 3) & 1) & (((m >> 2) & 1) | ((m & 3) != 0).to(torch.int32) | ((m >> 4) & 1))
    mr0 = t0 + ru0
    e0 = er - 8 + ((mr0 >> 3) & 1)
    o0 = sgn | ((e0 & 0xF) << 3) | (mr0 & 7)
    o0 = torch.where(is_zero | (e0 <= 0), sgn, o0)
    o0 = torch.where(e0 > 15, sgn | (15 << 3) | 6, o0)
    o0 = torch.where(is_nan, sgn | 0x7F, o0)
    # E5M2
    t1 = (m >> 5) & 3
    ru1 = ((m >> 4) & 1) & (((m >> 3) & 1) | ((m & 7) != 0).to(torch.int32) | ((m >> 5) & 1))
    mr1 = t1 + ru1
    e1 = er + ((mr1 >> 2) & 1)
    o1 = sgn | ((e1 & 0x1F) << 2) | (mr1 & 3)
    o1 = torch.where(is_zero, sgn, o1)
    o1 = torch.where(e1 >= 31, sgn | (30 << 2) | 3, o1)
    o1 = torch.where(is_nan, sgn | (31 << 2) | 1, o1)
    out = torch.where(mode == 0, o0, o1)
    if relu:
        out = torch.where((s == 1) & ~is_nan, torch.zeros_like(out), out)
    return out


def mac_decoded(wd, xd, mode_out, relu=False):
    """wd, xd: (..., 9) 9비트 E5M3. 반환: (...) 8비트 코드."""
    acc = torch.zeros(wd.shape[:-1], dtype=torch.int32, device=wd.device)
    for k in range(9):
        acc = acc_add(acc, fpu(wd[..., k], xd[..., k]) << 4)
    return acc_out(acc, mode_out, relu)


def mac(w, x, mode_w, mode_x, mode_out, relu=False):
    """w, x: (..., 9) 8비트 코드. mode_*: 스칼라 또는 브로드캐스트 가능한 텐서."""
    return mac_decoded(ir_decode(w, mode_w), ir_decode(x, mode_x), mode_out, relu)
