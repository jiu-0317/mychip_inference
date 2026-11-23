"""FP32 <-> FP8 코드 변환 (스케일 없음).

양자화 규칙 (GPU / FPGA / 실물 칩 공통):
  - 가장 가까운 표현값으로 반올림, 정확히 중간이면 코드가 짝수인 쪽
  - 최대 정규값(E4M3 448, E5M2 57344)을 넘으면 최대 정규값으로 포화
  - 최소 정규값(E4M3 2^-6, E5M2 2^-14)보다 작으면 0 과 최소 정규값 중 가까운 쪽
    (정확히 중간이면 0). subnormal 코드는 만들지 않는다 (칩이 0 으로 flush 하므로).
mode: 0 = E4M3, 1 = E5M2
"""
import torch

E4M3, E5M2 = 0, 1
NAMES = {E4M3: "E4M3", E5M2: "E5M2"}


def _positive_table(mode):
    """(값 오름차순, 대응 7비트 코드). 0 포함, NaN/Inf 코드 제외."""
    vals, codes = [0.0], [0]
    if mode == E4M3:
        for e in range(1, 16):
            for m in range(8):
                if e == 15 and m == 7:           # 출력에서 NaN 표시로 쓰는 코드
                    continue
                vals.append((1 + m / 8) * 2.0 ** (e - 7))
                codes.append((e << 3) | m)
    else:
        for e in range(1, 31):                   # e=31 은 NaN/Inf
            for m in range(4):
                vals.append((1 + m / 4) * 2.0 ** (e - 15))
                codes.append((e << 2) | m)
    return (torch.tensor(vals, dtype=torch.float64),
            torch.tensor(codes, dtype=torch.int32))


def _decode_table(mode):
    """8비트 코드 -> 칩이 해석하는 실수값 (IR.v 기준)."""
    t = torch.zeros(256, dtype=torch.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        if mode == E4M3:
            e, m = (c >> 3) & 0xF, c & 7
            v = 0.0 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
            if e == 15 and m == 7:
                v = float("nan")
        else:
            e, m = (c >> 2) & 0x1F, c & 3
            v = 0.0 if e == 0 else (1 + m / 4) * 2.0 ** (e - 15)
            if e == 31:
                v = float("nan")
        t[c] = s * v
    return t


_TAB = {m: _positive_table(m) for m in (E4M3, E5M2)}
_DEC = {m: _decode_table(m) for m in (E4M3, E5M2)}
MAX_NORMAL = {m: float(_TAB[m][0][-1]) for m in (E4M3, E5M2)}
MIN_NORMAL = {m: float(_TAB[m][0][1]) for m in (E4M3, E5M2)}


def encode(x, mode):
    """실수 텐서 -> 8비트 코드(int32). mode 는 정수 하나."""
    vals, codes = _TAB[mode]
    vals, codes = vals.to(x.device), codes.to(x.device)
    ax = x.abs().to(torch.float64)
    mids = (vals[:-1] + vals[1:]) / 2
    lo = torch.searchsorted(mids, ax, right=False)   # 중간값이면 아래쪽
    hi = torch.searchsorted(mids, ax, right=True)    # 중간값이면 위쪽
    idx = torch.where((lo != hi) & ((codes[lo] & 1) == 1), hi, lo)
    code = codes[idx]
    sign = (torch.signbit(x) & (code != 0)).to(torch.int32)
    return code | (sign << 7)


def decode(code, mode):
    """8비트 코드 -> float32. mode 는 정수 하나."""
    return _DEC[mode].to(code.device)[code.long()].to(torch.float32)


def encode_sel(x, mode):
    """mode 가 x 와 브로드캐스트되는 텐서일 때."""
    mode = torch.as_tensor(mode, device=x.device).expand_as(x)
    return torch.where(mode == E4M3, encode(x, E4M3), encode(x, E5M2))


def decode_sel(code, mode):
    mode = torch.as_tensor(mode, device=code.device).expand_as(code)
    return torch.where(mode == E4M3, decode(code, E4M3), decode(code, E5M2))
