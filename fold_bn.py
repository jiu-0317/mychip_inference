"""단계 1-2: BN 을 conv weight 에 합치고(fold), 합치기 전후가 같은지 확인.
실행:  python fold_bn.py
결과:  ckpt/mobilenetv1_folded.pt
  w' = w * gamma / sqrt(var + eps),   b' = beta - mean * gamma / sqrt(var + eps)
"""
import os, sys
import torch
from models.mobilenetv1 import MobileNetV1
from cifar import loaders, evaluate


def fold(conv, bn):
    """(합쳐진 weight, bias). float64 로 계산."""
    scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
    w = conv.weight.double() * scale.view(-1, 1, 1, 1)
    b = bn.bias.double() - bn.running_mean.double() * scale
    return w, b


def fold_model(model, double=False):
    """double=True 면 float64 모델을 돌려준다 (검증용). 기본은 float32."""
    folded = MobileNetV1(num_classes=model.fc.out_features, folded=True)
    if double:
        folded = folded.double()
    with torch.no_grad():
        pairs = [(model.conv1, model.bn1, folded.conv1)]
        for src, dst in zip(model.blocks, folded.blocks):
            pairs += [(src.dw, src.bn1, dst.dw), (src.pw, src.bn2, dst.pw)]
        for conv, bn, tgt in pairs:
            w, b = fold(conv, bn)
            tgt.weight.copy_(w); tgt.bias.copy_(b)
        folded.fc.load_state_dict(model.fc.state_dict())
    return folded


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(here, "ckpt", "mobilenetv1_fp32.pt"), map_location="cpu")
    model = MobileNetV1(); model.load_state_dict(ck["state"]); model.eval()
    folded = fold_model(model).eval()
    _, test_loader = loaders(workers=2)
    acc0, p0, l0 = evaluate(model.to(device), test_loader, device)
    acc1, p1, l1 = evaluate(folded.to(device), test_loader, device)
    agree = (p0 == p1).float().mean().item() * 100
    maxdiff = (l0 - l1).abs().max().item()
    print("before fold : %.2f%%" % acc0)
    print("after  fold : %.2f%%" % acc1)
    print("prediction agreement: %.2f%%   max logit diff: %.2e" % (agree, maxdiff))
    print("\ndepthwise folded weight |w| range per layer (E4M3 표현 범위: 0.0156 ~ 448)")
    for i, b in enumerate(folded.blocks):
        a = b.dw.weight.detach().abs().flatten().cpu()
        print("  dw%02d  C=%4d  min %.2e  median %.2e  max %.2e   <2^-6: %5.1f%%"
              % (i + 1, b.dw.weight.shape[0], a.min(), a.median(), a.max(), (a < 2 ** -6).float().mean() * 100))
    # float64 로 다시 비교: 차이가 반올림 오차라면 여기서 사라진다
    m64 = model.double()
    f64 = fold_model(m64, double=True).to(device).eval()
    d64 = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).double()
            d64 = max(d64, (m64(x) - f64(x)).abs().max().item())
    print("float64 max logit diff: %.2e  (1e-6 이하면 fold 수식은 정확, FP32 차이는 반올림 오차)" % d64)
    ok = d64 < 1e-6
    out = os.path.join(here, "ckpt", "mobilenetv1_folded.pt")
    torch.save({"state": folded.cpu().state_dict(), "test_acc": acc1}, out)
    print("\nSTEP 1 fold check:", "PASS" if ok else
          "FAIL (float64 에서도 차이가 남음 - fold 수식 확인 필요)")
    print("saved:", out)
