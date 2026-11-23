"""단계 3 (실험 1): weight / 입력 / 출력을 각각 한 포맷으로 통일한 8조합 평가.
실행:  python run_uniform.py
결과:  화면 표 + results/step3_uniform.json
"""
import os, json, time, itertools
import torch
from models.mobilenetv1 import MobileNetV1
from cifar import loaders, evaluate
from fp8.net import FP8Net, uniform_cfg
from fp8 import quant as q

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_folded.pt"), map_location="cpu")
    model = MobileNetV1(folded=True); model.load_state_dict(ck["state"]); model.to(device).eval()
    _, test_loader = loaders(workers=2)
    acc32, pred32, _ = evaluate(model, test_loader, device)
    print("FP32 (folded): %.2f%%\n" % acc32)
    print("  W     I     O     | acc     drop    | agree with FP32 | time")
    rows = []
    for w, i, o in itertools.product((q.E4M3, q.E5M2), repeat=3):
        t0 = time.time()
        net = FP8Net(model, uniform_cfg(model, w, i, o))
        acc, pred, _ = evaluate(net, test_loader, device)
        agree = (pred == pred32).float().mean().item() * 100
        print("  %s  %s  %s  | %6.2f%% %+6.2f%%p | %14.2f%% | %.0fs" % (
            q.NAMES[w], q.NAMES[i], q.NAMES[o], acc, acc - acc32, agree, time.time() - t0))
        rows.append({"W": q.NAMES[w], "I": q.NAMES[i], "O": q.NAMES[o], "acc": acc, "agree": agree})
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    out = os.path.join(here, "results", "step3_uniform.json")
    with open(out, "w") as f:
        json.dump({"fp32_acc": acc32, "rows": rows}, f, indent=1)
    print("\nsaved:", out)
