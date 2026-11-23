"""pointwise / 첫 conv / FC 의 분포 분석 (단계 2 를 depthwise 밖으로 확장). FP32 순전파만 사용.
실행:  python analyze_pw.py
결과:  화면 표 + results/pw_stats.pt

레이어마다 세 가지 값을 본다.
  W: BN 을 합친 weight (출력 채널 하나 = weight Cin 개 묶음)
  I: 레이어 입력 (입력 채널별)          O: bias 더하기 전 합 (출력 채널별)
"""
import os
import torch
import torch.nn.functional as F
from models.mobilenetv1 import MobileNetV1
from cifar import calib_loader
from analyze import tensor_stats, add, pct


def amax(v):
    return v.detach().abs().max().item()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    model = MobileNetV1(folded=True); model.load_state_dict(ck["state"]); model.to(device).eval()

    layers = [("conv1", model.conv1)] + [("pw%02d" % (i + 1), b.pw) for i, b in enumerate(model.blocks)] + [("fc", model.fc)]
    I_acc = {n: dict() for n, _ in layers}
    O_acc = {n: dict() for n, _ in layers}
    I_max = {n: 0.0 for n, _ in layers}
    O_max = {n: 0.0 for n, _ in layers}

    def make_hook(name):
        def hook(mod, inp):
            x = inp[0]
            if isinstance(mod, torch.nn.Linear):
                s = F.linear(x, mod.weight)
                dims = (0,)
            else:
                s = F.conv2d(x, mod.weight, None, mod.stride, mod.padding, 1, mod.groups)
                dims = (0, 2, 3)
            add(I_acc[name], tensor_stats(x, dims))
            add(O_acc[name], tensor_stats(s, dims))
            I_max[name] = max(I_max[name], amax(x)); O_max[name] = max(O_max[name], amax(s))
        return hook

    hooks = [m.register_forward_pre_hook(make_hook(n)) for n, m in layers]
    with torch.no_grad():
        for x, _ in calib_loader():
            model(x.to(device))
    for h in hooks:
        h.remove()

    print("FP32(folded) test acc: %.2f%%   calibration: train 1000 images" % ck.get("test_acc", float("nan")))
    print("E4M3 표현 범위 0.0156 ~ 448,  E5M2 표현 범위 0.000061 ~ 57344")
    print("\n[W] weight (출력 채널별 묶음).  dead = 묶음 전체가 두 포맷 모두에서 0 / E4-only-dead = E4M3 에서만 전체 0")
    print("  layer   Cout  fan-in  dead  E4-only-dead | relerr E4M3   E5M2 | E5 better | lost E4M3   E5M2 | median|w|  max|w|")
    W_acc, live_all = {}, {}
    for name, m in layers:
        w = m.weight.detach().flatten(1)
        s = {k: v.cpu() for k, v in tensor_stats(w, (1,)).items()}
        W_acc[name] = s
        dead5 = s["flush1"] == s["nz"]
        dead4 = (s["flush0"] == s["nz"]) & ~dead5
        live = ~dead5
        live_all[name] = live
        e = s["energy"][live].sum()
        print("  %-6s %5d  %6d  %4d  %12d | %10.2e %8.2e | %8.1f%% | %8.1f%% %5.1f%% | %9.2e %7.2f" % (
            name, w.shape[0], w.shape[1], dead5.sum(), dead4.sum(),
            s["err0"][live].sum() / e, s["err1"][live].sum() / e,
            pct((s["err1"] < s["err0"])[live].sum(), live.sum()),
            pct(s["flush0"][live].sum(), s["nz"][live].sum()), pct(s["flush1"][live].sum(), s["nz"][live].sum()),
            w.abs().median().item(), amax(w)))

    for tag, accs, mx, title in (("I", I_acc, I_max, "레이어 입력 (입력 채널별, 전체 채널)"),
                                 ("O", O_acc, O_max, "bias 더하기 전 합 (출력 채널별, weight 가 dead 인 채널 제외)")):
        print("\n[%s] %s" % (tag, title))
        print("  layer  nonzero | relerr E4M3   E5M2 | E5 better | lost E4M3   E5M2 | saturated E4M3 |  max|v|")
        for name, _ in layers:
            s = accs[name]
            C = s["energy"].numel()
            live = live_all[name] if tag == "O" else torch.ones(C, dtype=torch.bool)
            e = s["energy"][live].sum()
            nz = s["nz"][live].sum()
            print("  %-6s %6.1f%% | %10.2e %8.2e | %8.1f%% | %8.2f%% %5.2f%% | %13.4f%% | %7.2f" % (
                name, pct(nz, s["n"] * live.sum()), s["err0"][live].sum() / e, s["err1"][live].sum() / e,
                pct((s["err1"] < s["err0"])[live].sum(), live.sum()),
                pct(s["flush0"][live].sum(), nz), pct(s["flush1"][live].sum(), nz),
                pct(s["sat0"][live].sum(), nz), mx[name]))

    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    out = os.path.join(here, "results", "pw_stats.pt")
    torch.save({"W": W_acc, "I": I_acc, "O": O_acc, "I_max": I_max, "O_max": O_max}, out)
    print("\nsaved:", out)
