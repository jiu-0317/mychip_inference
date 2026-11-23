"""단계 4 (실험 2): 채널 단위 포맷 선택 방법 A / B / C 평가. 캘리브레이션 시드 3개.
실행:  python run_mixed.py
결과:  화면 표 + results/step4_mixed.json, results/step4_cfg.pt (선택된 포맷)
"""
import os, json, time
import torch
from models.mobilenetv1 import MobileNetV1
from cifar import loaders, evaluate, calib_loader
from fp8.net import FP8Net, uniform_cfg
from fp8 import quant as q, choose

SEEDS = (0, 1, 2)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    model = MobileNetV1(folded=True); model.load_state_dict(ck["state"]); model.to(device).eval()
    _, test_loader = loaders(workers=2)
    acc32, pred32, _ = evaluate(model, test_loader, device)
    live = choose.live_mask(model)

    def run(cfg):
        acc, pred, _ = evaluate(FP8Net(model, cfg), test_loader, device)
        return acc, (pred == pred32).float().mean().item() * 100

    print("FP32 (folded): %.2f%%\n" % acc32)
    print("  method           seed | acc     drop     | agree with FP32")
    rows, cfgs = [], {}
    for name, (w, i, o) in (("all E4M3", (0, 0, 0)), ("all E5M2", (1, 1, 1))):
        acc, agree = run(uniform_cfg(model, w, i, o))
        print("  %-16s    - | %6.2f%% %+6.2f%%p | %6.2f%%" % (name, acc, acc - acc32, agree))
        rows.append({"method": name, "seed": None, "acc": acc, "agree": agree})

    for seed in SEEDS:
        t0 = time.time()
        calib = [x.to(device) for x, _ in calib_loader(seed=seed)]
        stats = choose.collect_fp32_stats(model, calib)
        todo = (("A range rule", choose.choose_a(stats)),
                ("B min error", choose.choose_b(stats)),
                ("C chip-aware", choose.choose_c(model, calib)))
        for name, cfg in todo:
            acc, agree = run(cfg)
            print("  %-16s %4d | %6.2f%% %+6.2f%%p | %6.2f%%" % (name, seed, acc, acc - acc32, agree))
            rows.append({"method": name, "seed": seed, "acc": acc, "agree": agree})
            cfgs["%s/seed%d" % (name[0], seed)] = cfg
        print("  (seed %d: %.0fs)" % (seed, time.time() - t0))

    print("\n  mean +- std over seeds")
    for name in ("A range rule", "B min error", "C chip-aware"):
        a = torch.tensor([r["acc"] for r in rows if r["method"] == name])
        g = torch.tensor([r["agree"] for r in rows if r["method"] == name])
        print("  %-16s | acc %6.2f +- %.2f  (%+.2f%%p) | agree %6.2f +- %.2f" % (
            name, a.mean(), a.std(), a.mean() - acc32, g.mean(), g.std()))

    print("\n  E5M2 를 고른 채널 비율 % (seed 0, 꺼진 채널 제외)   W / I / O")
    print("  layer  live |        A         |        B         |        C")
    for l in range(len(live)):
        m = live[l]
        cells = []
        for k in "ABC":
            c = cfgs["%s/seed0" % k][l]
            cells.append("%5.1f %5.1f %5.1f" % tuple(100.0 * c[t][m].float().mean().item() if m.any() else 0.0 for t in "WIO"))
        print("  dw%02d  %4d | %s | %s | %s" % (l + 1, int(m.sum()), cells[0], cells[1], cells[2]))

    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    with open(os.path.join(here, "results", "step4_mixed.json"), "w") as f:
        json.dump({"fp32_acc": acc32, "rows": rows}, f, indent=1)
    torch.save(cfgs, os.path.join(here, "results", "step4_cfg.pt"))
    print("\nsaved: results/step4_mixed.json, results/step4_cfg.pt")
