"""pointwise 혼합 정밀도 실험: pointwise 13개만 FP8(양자화 + 정확한 덧셈), 나머지는 FP32.
실행:  python run_pw.py
결과:  화면 표 + results/pw_mixed.json, results/pw_cfg.pt
"""
import os, json, time, itertools
import torch
from models.mobilenetv1 import MobileNetV1
from cifar import loaders, evaluate, calib_loader
from fp8 import quant as q, pw

SEEDS = (0, 1, 2)
METHODS = (("layer   B min error", "B", "layer"), ("layer   C output", "C", "layer"),
           ("channel B min error", "B", "channel"), ("channel C output", "C", "channel"))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    model = MobileNetV1(folded=True); model.load_state_dict(ck["state"]); model.to(device).eval()
    _, test_loader = loaders(workers=2)
    acc32, pred32, _ = evaluate(model, test_loader, device)

    def run(cfg):
        acc, pred, _ = evaluate(pw.PWNet(model, cfg), test_loader, device)
        return acc, (pred == pred32).float().mean().item() * 100

    rows, cfgs = [], {}
    print("FP32 (folded): %.2f%%\n" % acc32)
    print("[1] 동일 포맷 8조합 (pointwise 전체)")
    print("  W     I     O     | acc     drop     | agree with FP32")
    for w, i, o in itertools.product((q.E4M3, q.E5M2), repeat=3):
        acc, agree = run(pw.uniform_cfg(model, w, i, o))
        print("  %s  %s  %s  | %6.2f%% %+6.2f%%p | %6.2f%%" % (q.NAMES[w], q.NAMES[i], q.NAMES[o], acc, acc - acc32, agree))
        rows.append({"method": "uniform %s/%s/%s" % (q.NAMES[w], q.NAMES[i], q.NAMES[o]), "seed": None, "acc": acc, "agree": agree})

    print("\n[2] 혼합 (캘리브레이션 시드 3개)")
    print("  method               seed | acc     drop     | agree with FP32")
    for seed in SEEDS:
        t0 = time.time()
        calib = [x.to(device) for x, _ in calib_loader(seed=seed)]
        stats = pw.collect_stats(model, calib)
        for name, kind, level in METHODS:
            cfg = pw.choose_b(stats, level) if kind == "B" else pw.choose_c(model, calib, level)
            acc, agree = run(cfg)
            print("  %-20s %4d | %6.2f%% %+6.2f%%p | %6.2f%%" % (name, seed, acc, acc - acc32, agree))
            rows.append({"method": name, "seed": seed, "acc": acc, "agree": agree})
            cfgs["%s/seed%d" % (name, seed)] = cfg
        print("  (seed %d: %.0fs)" % (seed, time.time() - t0))

    print("\n  mean +- std over seeds")
    for name, _, _ in METHODS:
        a = torch.tensor([r["acc"] for r in rows if r["method"] == name])
        g = torch.tensor([r["agree"] for r in rows if r["method"] == name])
        print("  %-20s | acc %6.2f +- %.2f  (%+.2f%%p) | agree %6.2f +- %.2f" % (
            name, a.mean(), a.std(), a.mean() - acc32, g.mean(), g.std()))

    print("\n[3] 선택 결과 (seed 0).  layer: 레이어의 W/I/O 포맷 (4=E4M3, 5=E5M2)")
    print("    channel: E5M2 를 고른 채널 비율 %  W / I / O  (W, O 는 weight 가 꺼진 출력 채널 제외)")
    print("  layer | layer B  layer C |     channel B     |     channel C")
    for l, b in enumerate(model.blocks):
        wq = b.pw.weight.detach().flatten(1)
        live = (q.encode(wq, q.E5M2) & 0x7F).ne(0).any(1).cpu()
        cells = []
        for name in (METHODS[0][0], METHODS[1][0]):
            c = cfgs["%s/seed0" % name][l]
            cells.append("/".join("5" if int(c[t][0]) else "4" for t in "WIO"))
        for name in (METHODS[2][0], METHODS[3][0]):
            c = cfgs["%s/seed0" % name][l]
            cells.append("%5.1f %5.1f %5.1f" % (100.0 * c["W"][live].float().mean().item(),
                                                 100.0 * c["I"].float().mean().item(),
                                                 100.0 * c["O"][live].float().mean().item()))
        print("  pw%02d  |  %s    %s  | %s | %s" % (l + 1, cells[0], cells[1], cells[2], cells[3]))

    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "pw_mixed.json"), "w") as f:
        json.dump({"fp32_acc": acc32, "rows": rows}, f, indent=1)
    torch.save(cfgs, os.path.join(here, "results", "pw_cfg.pt"))
    print("\nsaved: results/pw_mixed.json, results/pw_cfg.pt")
