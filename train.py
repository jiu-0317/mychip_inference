"""단계 1-1: FP32 MobileNetV1 을 CIFAR-10 으로 학습.
실행:  python train.py            (기본 200 epoch)
결과:  ckpt/mobilenetv1_fp32.pt   (마지막 epoch 가중치)
"""
import argparse, os, time
import torch
import torch.nn as nn
from models.mobilenetv1 import MobileNetV1
from cifar import loaders, evaluate

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    train_loader, test_loader = loaders(args.batch, args.workers)
    model = MobileNetV1().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(train_loader), pct_start=0.15)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "ckpt"), exist_ok=True)
    out = os.path.join(here, "ckpt", "mobilenetv1_fp32.pt")
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if ep % 5 == 0 or ep == args.epochs:
            acc, _, _ = evaluate(model, test_loader, device)
            print("epoch %3d  loss %.4f  test acc %.2f%%  (%.0fs)" % (ep, tot / n, acc, time.time() - t0))
        else:
            print("epoch %3d  loss %.4f  (%.0fs)" % (ep, tot / n, time.time() - t0))
        torch.save({"state": model.state_dict(), "epoch": ep}, out)
    acc, _, _ = evaluate(model, test_loader, device)
    torch.save({"state": model.state_dict(), "epoch": args.epochs, "test_acc": acc}, out)
    print("\nFP32 test accuracy: %.2f%%   saved: %s" % (acc, out))
