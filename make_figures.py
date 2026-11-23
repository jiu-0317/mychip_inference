"""발표용 그림 4개 생성. 실행:  python make_figures.py   (필요: pip install matplotlib)
결과:  figures/fig1_distribution.png      단계 2  값 분포와 포맷 표현 범위 (dw03, dw12 입력)
       figures/fig1b_distribution_linear.png  같은 내용을 일반(선형) 눈금으로
       figures/fig2_uniform.png           단계 3  동일 포맷 8조합의 정확도 하락
       figures/fig3_methods.png           단계 4  방법별 FP32 와 달라진 예측 수
       figures/fig4_selection.png         단계 4  방법 C 가 E5M2 를 고른 채널 비율
그림에 쓴 값은 모두 화면에 출력한다 (results/ 의 파일과 대조용).
fig2~4 는 results/ 의 저장된 결과만 사용하고, fig1 만 FP32 순전파를 한 번 돌린다.
"""
import os, json, math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.mobilenetv1 import MobileNetV1
from cifar import calib_loader
from fp8 import quant as q

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED, GRID, SURF = "#9a998f", "#1f1f1d", "#6b6a63", "#e4e3dd", "#fcfcfb"
N_TEST = 10000

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  saved:", path)


def load_model(device):
    ck = torch.load(os.path.join(HERE, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    m = MobileNetV1(folded=True); m.load_state_dict(ck["state"])
    return m.to(device).eval()


def live_masks(model):
    """weight 9개가 두 포맷 모두에서 0 이 되는(꺼진) 채널을 뺀 마스크 (analyze.py, run_mixed.py 와 같은 정의)"""
    return [(q.encode(b.dw.weight.detach().flatten(1), q.E5M2) & 0x7F).ne(0).any(1).cpu() for b in model.blocks]


# ---------------------------------------------------------------- fig 1
def fig1(model, live, device, layers=(3, 12)):
    print("\n[fig1] depthwise 입력 분포 (캘리브레이션 1000장, 꺼진 채널 제외, 0 이 아닌 값)")
    lo, hi = -24, 17
    edges = torch.arange(lo, hi + 1, dtype=torch.float64)
    hist = {l: torch.zeros(len(edges) - 1, dtype=torch.float64) for l in layers}
    cnt = {l: {"nz": 0, "lost4": 0, "lost5": 0, "sat4": 0} for l in layers}

    def make_hook(l):
        def hook(mod, inp):
            x = inp[0][:, live[l - 1].to(inp[0].device)]
            v = x[x != 0]
            cnt[l]["nz"] += v.numel()
            cnt[l]["lost4"] += int((q.encode(v, q.E4M3) == 0).sum())
            cnt[l]["lost5"] += int((q.encode(v, q.E5M2) == 0).sum())
            cnt[l]["sat4"] += int((v > q.MAX_NORMAL[q.E4M3]).sum())
            lg = torch.log2(v.double()).clamp(lo, hi - 1e-9).cpu()
            hist[l] += torch.histc(lg, bins=len(edges) - 1, min=lo, max=hi)
        return hook

    hooks = [model.blocks[l - 1].dw.register_forward_pre_hook(make_hook(l)) for l in layers]
    with torch.no_grad():
        for x, _ in calib_loader():
            model(x.to(device))
    for h in hooks:
        h.remove()

    stats = torch.load(os.path.join(HERE, "results", "step2_stats.pt"), map_location="cpu")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    e4 = (math.log2(q.MIN_NORMAL[q.E4M3]), math.log2(q.MAX_NORMAL[q.E4M3]))
    e5 = (math.log2(q.MIN_NORMAL[q.E5M2]), math.log2(q.MAX_NORMAL[q.E5M2]))
    ymax = max(float((hist[l] / hist[l].sum() * 100).max()) for l in layers) * 1.38
    for ax, l in zip(axes, layers):
        c = cnt[l]
        share = hist[l] / hist[l].sum() * 100
        p4, p5 = 100.0 * c["lost4"] / c["nz"], 100.0 * c["lost5"] / c["nz"]
        s = stats["I"][l - 1]; m = live[l - 1]
        ref4 = 100.0 * float(s["flush0"][m].sum() / s["nz"][m].sum())
        ref5 = 100.0 * float(s["flush1"][m].sum() / s["nz"][m].sum())
        print("  dw%02d  nonzero values %d | rounds to 0: E4M3 %.2f%% (step2 %.2f%%)  E5M2 %.2f%% (step2 %.2f%%) | above 448: %d"
              % (l, c["nz"], p4, ref4, p5, ref5, c["sat4"]))
        ax.axvspan(e5[0], e5[1], color=ORANGE, alpha=0.10, lw=0)
        ax.axvspan(e4[0], e4[1], color=BLUE, alpha=0.16, lw=0)
        ax.bar(edges[:-1].numpy() + 0.5, share.numpy(), width=0.86, color=INK, alpha=0.78, lw=0)
        ax.set_ylim(0, ymax)
        ax.text((e4[0] + e4[1]) / 2, ymax * 0.955, "E4M3 range", ha="center", va="top", color=INK, fontsize=11)
        ax.text(e5[0] + 0.3, ymax * 0.86, "E5M2 range", ha="left", va="top", color=INK, fontsize=11)
        ax.set_title("dw%02d input  |  rounds to 0:  E4M3 %.1f%%,  E5M2 %.1f%%" % (l, p4, p5))
        ax.set_xlim(lo, hi)
        ax.set_xticks(range(-24, 17, 4))
        ax.set_xlabel("log2 |value|")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of nonzero values (%)")
    fig.tight_layout()
    save(fig, "fig1_distribution.png")


# ---------------------------------------------------------------- fig 1b (선형 눈금)
def fig1_linear(model, live, device, spec=((3, 1.0, 50), (12, 0.06, 60))):
    """spec: (레이어, 가로축 최댓값, 구간 수). 레이어마다 값 크기가 달라 가로축 범위가 다르다."""
    print("\n[fig1b] depthwise 입력 분포, 선형 눈금 (캘리브레이션 1000장, 꺼진 채널 제외, 0 이 아닌 값)")
    hist = {l: torch.zeros(nb, dtype=torch.float64) for l, _, nb in spec}
    cnt = {l: {"nz": 0, "lost4": 0, "over": 0} for l, _, _ in spec}
    rng = {l: (xmax, nb) for l, xmax, nb in spec}

    def make_hook(l):
        def hook(mod, inp):
            x = inp[0][:, live[l - 1].to(inp[0].device)]
            v = x[x != 0]
            xmax, nb = rng[l]
            cnt[l]["nz"] += v.numel()
            cnt[l]["lost4"] += int((q.encode(v, q.E4M3) == 0).sum())
            cnt[l]["over"] += int((v >= xmax).sum())
            hist[l] += torch.histc(v[v < xmax].double().cpu(), bins=nb, min=0, max=xmax)
        return hook

    hooks = [model.blocks[l - 1].dw.register_forward_pre_hook(make_hook(l)) for l, _, _ in spec]
    with torch.no_grad():
        for x, _ in calib_loader():
            model(x.to(device))
    for h in hooks:
        h.remove()

    thr = q.MIN_NORMAL[q.E4M3] / 2            # 이보다 작으면 E4M3 에서 0 으로 반올림 (2^-7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (l, xmax, nb) in zip(axes, spec):
        c = cnt[l]
        share = hist[l] / c["nz"] * 100
        w = xmax / nb
        p4 = 100.0 * c["lost4"] / c["nz"]
        print("  dw%02d  nonzero %d | rounds to 0 in E4M3 (< %.5f): %.2f%% | beyond axis (>= %g): %.2f%% | bin width %g"
              % (l, c["nz"], thr, p4, xmax, 100.0 * c["over"] / c["nz"], w))
        centers = (torch.arange(nb, dtype=torch.float64) + 0.5) * w
        ax.axvspan(0, thr, color=ORANGE, alpha=0.22, lw=0)
        ax.bar(centers.numpy(), share.numpy(), width=w * 0.88, color=INK, alpha=0.78, lw=0)
        ax.axvline(thr, color=ORANGE, lw=1.5)
        top = float(share.max()) * 1.25
        ax.set_ylim(0, top)
        ax.text(thr + xmax * 0.015, top * 0.95, "below %.4f: rounds to 0 in E4M3" % thr, ha="left", va="top", fontsize=11)
        ax.set_title("dw%02d input  |  rounds to 0 in E4M3: %.1f%%" % (l, p4))
        ax.set_xlim(0, xmax)
        ax.set_xlabel("value   (%.1f%% of values are beyond this axis)" % (100.0 * c["over"] / c["nz"]))
        ax.set_ylabel("share of nonzero values (%)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig1b_distribution_linear.png")


# ---------------------------------------------------------------- fig 2
def fig2():
    print("\n[fig2] 동일 포맷 8조합 (results/step3_uniform.json)")
    d = json.load(open(os.path.join(HERE, "results", "step3_uniform.json")))
    fp32 = d["fp32_acc"]
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        drop = r["acc"] - fp32
        col = BLUE if r["W"] == "E4M3" else ORANGE
        ax.barh(y, drop, height=0.62, color=col, lw=0)
        ax.text(drop - 0.03, y, "%+.2f" % drop, ha="right", va="center", fontsize=11, color=INK)
        print("  W %s  I %s  O %s | acc %.2f%%  drop %+.2f%%p" % (r["W"], r["I"], r["O"], r["acc"], drop))
    ax.set_yticks(ys)
    ax.set_yticklabels(["%s / %s / %s" % (r["W"], r["I"], r["O"]) for r in rows], color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(-0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(-0.5, len(rows) - 0.42, "0.5%p", ha="center", va="bottom", color=MUTED, fontsize=10)
    ax.set_xlim(-2.45, 0.05)
    ax.set_xlabel("accuracy change vs FP32 (%%p)   [FP32 = %.2f%%]" % fp32)
    ax.set_ylabel("weight / input / output format")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["weight = E4M3", "weight = E5M2"], loc="upper left", frameon=False)
    ax.set_title("Uniform format, depthwise on chip arithmetic")
    fig.tight_layout()
    save(fig, "fig2_uniform.png")


# ---------------------------------------------------------------- fig 3
def fig3():
    print("\n[fig3] 방법별 FP32 와 달라진 예측 수 (results/step4_mixed.json, 테스트 %d장)" % N_TEST)
    d = json.load(open(os.path.join(HERE, "results", "step4_mixed.json")))
    names = [("all E4M3", "all E4M3"), ("A range rule", "A\nrange rule"),
             ("B min error", "B\nmin error"), ("C chip-aware", "C\nchip-aware")]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    top = 0
    for i, (key, label) in enumerate(names):
        rs = [r for r in d["rows"] if r["method"] == key]
        changed = torch.tensor([round((100.0 - r["agree"]) / 100.0 * N_TEST) for r in rs], dtype=torch.float64)
        acc = torch.tensor([r["acc"] for r in rs], dtype=torch.float64)
        mean = float(changed.mean())
        std = float(changed.std()) if len(rs) > 1 else 0.0
        print("  %-13s changed %s  mean %.1f  std %.1f | acc mean %.2f%%" % (
            key, [int(c) for c in changed], mean, std, float(acc.mean())))
        ax.bar(i, mean, width=0.58, color=GRAY if i == 0 else BLUE, lw=0)
        if std > 0:
            ax.errorbar(i, mean, yerr=std, color=INK, lw=1.2, capsize=4)
        ax.text(i, mean + std + 6, "%d" % round(mean), ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(i, 12, "acc %.2f%%" % float(acc.mean()), ha="center", va="bottom", fontsize=10.5, color="white")
        top = max(top, mean + std)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names], color=INK)
    ax.set_ylim(0, top * 1.16)
    ax.set_ylabel("predictions different from FP32\n(out of %d test images)" % N_TEST)
    ax.grid(axis="x", visible=False)
    ax.set_title("Per-channel format selection (lower = closer to FP32)")
    fig.tight_layout()
    save(fig, "fig3_methods.png")


# ---------------------------------------------------------------- fig 4
def fig4(live):
    print("\n[fig4] 방법 C 가 E5M2 를 고른 채널 비율 %, seed 0, 꺼진 채널 제외 (results/step4_cfg.pt)")
    cfg = torch.load(os.path.join(HERE, "results", "step4_cfg.pt"), map_location="cpu")["C/seed0"]
    xs = list(range(1, len(cfg) + 1))
    series = (("W", "weight", BLUE, "o"), ("I", "input", ORANGE, "s"), ("O", "output", AQUA, "^"))
    vals = {k: [100.0 * c[k][m].float().mean().item() for c, m in zip(cfg, live)] for k, _, _, _ in series}
    print("  layer  live |     W      I      O")
    for i in range(len(cfg)):
        print("  dw%02d  %4d | %5.1f  %5.1f  %5.1f" % (i + 1, int(live[i].sum()), vals["W"][i], vals["I"][i], vals["O"][i]))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, name, col, mk in series:
        ax.plot(xs, vals[k], color=col, lw=2, marker=mk, ms=7, mec=SURF, mew=1.5, label=name, clip_on=False, zorder=3)
        ax.text(xs[-1] + 0.25, vals[k][-1], name, color=INK, va="center", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(["dw%02d" % x for x in xs], rotation=45, ha="right")
    ax.set_xlim(0.5, len(xs) + 1.5)
    ax.set_ylim(0, 70)
    ax.set_ylabel("channels assigned E5M2 (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Method C: where E5M2 is chosen (rest is E4M3)")
    fig.tight_layout()
    save(fig, "fig4_selection.png")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    live = live_masks(model)
    fig1(model, live, device)
    fig1_linear(model, live, device)
    fig2()
    fig3()
    fig4(live)
