"""단계 2: 분포 분석. FP32 순전파만 사용 (칩 모델 없음).
실행:  python analyze.py
결과:  화면 표 + results/step2_stats.pt (채널별 통계)

depthwise 레이어마다 채널별로 세 가지 값을 본다.
  W: BN 을 합친 weight 9개      I: 레이어 입력 (ReLU 뒤, 0 이상)      O: bias 더하기 전 합
각각 E4M3 / E5M2 로 양자화했을 때의 오차와 0 소실, 포화를 센다.
"""
import os
import torch
import torch.nn.functional as F
from models.mobilenetv1 import MobileNetV1
from cifar import calib_loader
from fp8 import quant as q

MODES = (q.E4M3, q.E5M2)


def tensor_stats(v, dims):
    """v 를 dims 축으로 합산한 채널별 통계 dict (float64)."""
    v = v.double()
    out = {"n": torch.tensor(float(v.numel() // v.shape[1 if 0 in dims else 0])),
           "nz": (v != 0).double().sum(dims), "energy": (v * v).sum(dims)}
    for m in MODES:
        qv = q.decode(q.encode(v.float(), m), m).double()
        out["err%d" % m] = ((v - qv) ** 2).sum(dims)
        out["flush%d" % m] = ((v != 0) & (qv == 0)).double().sum(dims)
        out["sat%d" % m] = (v.abs() > q.MAX_NORMAL[m]).double().sum(dims)
    return out


def add(acc, new):
    for k, v in new.items():
        acc[k] = acc.get(k, 0) + v.cpu()


def pct(a, b):
    return 100.0 * float(a) / max(float(b), 1e-300)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    model = MobileNetV1(folded=True); model.load_state_dict(ck["state"]); model.to(device).eval()
    L = len(model.blocks)
    I_acc = [dict() for _ in range(L)]
    O_acc = [dict() for _ in range(L)]

    def make_hook(i):
        def hook(mod, inp):
            x = inp[0]
            add(I_acc[i], tensor_stats(x, (0, 2, 3)))
            s = F.conv2d(x, mod.weight, None, mod.stride, mod.padding, 1, mod.groups)
            add(O_acc[i], tensor_stats(s, (0, 2, 3)))
        return hook

    hooks = [b.dw.register_forward_pre_hook(make_hook(i)) for i, b in enumerate(model.blocks)]
    with torch.no_grad():
        for x, _ in calib_loader():
            model(x.to(device))
    for h in hooks:
        h.remove()

    W_acc = []
    for b in model.blocks:
        w = b.dw.weight.detach().flatten(1)            # (C, 9)
        W_acc.append({k: v.cpu() for k, v in tensor_stats(w, (1,)).items()})

    print("FP32(folded) test acc: %.2f%%   calibration: train 1000 images" % ck.get("test_acc", float("nan")))
    print("\n[W] 합쳐진 weight.  dead = 9개 모두 두 포맷에서 0 / E4-only-dead = E4M3 에서만 9개 모두 0")
    print("     relerr = 양자화 오차 에너지 / 값 에너지 (살아있는 채널 합산),  E5 better = E5M2 오차가 더 작은 채널 비율")
    print("  layer    C  dead  E4-only-dead | relerr E4M3   E5M2 | E5 better | weights lost E4M3  E5M2")
    summary = []
    for i, s in enumerate(W_acc):
        C = s["energy"].numel()
        dead5 = s["flush1"] == s["nz"]                 # E5M2 에서 전부 0 (=E4M3 에서도 전부 0)
        dead4 = (s["flush0"] == s["nz"]) & ~dead5
        live = ~dead5
        e = s["energy"][live].sum()
        print("  dw%02d  %4d  %4d  %12d | %10.2e %8.2e | %8.1f%% | %15.1f%% %5.1f%%" % (
            i + 1, C, dead5.sum(), dead4.sum(), s["err0"][live].sum() / e, s["err1"][live].sum() / e,
            pct((s["err1"] < s["err0"])[live].sum(), live.sum()),
            pct(s["flush0"][live].sum(), s["nz"][live].sum()), pct(s["flush1"][live].sum(), s["nz"][live].sum())))
        summary.append(live)

    for name, accs in (("I", I_acc), ("O", O_acc)):
        print("\n[%s] %s  (weight 가 dead 인 채널 제외)" % (name, "레이어 입력 (ReLU 뒤)" if name == "I" else "bias 더하기 전 합"))
        print("  layer  nonzero | relerr E4M3   E5M2 | E5 better | lost E4M3   E5M2 | saturated E4M3")
        for i, s in enumerate(accs):
            live = summary[i]
            e = s["energy"][live].sum()
            tot = s["n"] * live.sum()
            nz = s["nz"][live].sum()
            print("  dw%02d  %6.1f%% | %10.2e %8.2e | %8.1f%% | %8.2f%% %5.2f%% | %13.4f%% |" % (
                i + 1, pct(nz, tot), s["err0"][live].sum() / e, s["err1"][live].sum() / e,
                pct((s["err1"] < s["err0"])[live].sum(), live.sum()),
                pct(s["flush0"][live].sum(), nz), pct(s["flush1"][live].sum(), nz),
                pct(s["sat0"][live].sum(), nz)))

    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    out = os.path.join(here, "results", "step2_stats.pt")
    torch.save({"W": W_acc, "I": I_acc, "O": O_acc}, out)
    print("\nsaved:", out)
