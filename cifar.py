"""CIFAR-10 로더. 데이터는 inference/data 에 내려받는다."""
import os
import torch
import torchvision
import torchvision.transforms as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def loaders(batch_size=128, workers=4):
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    te = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train = torchvision.datasets.CIFAR10(ROOT, train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10(ROOT, train=False, download=True, transform=te)
    return (torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=workers,
                                        pin_memory=True, drop_last=True, persistent_workers=workers > 0),
            torch.utils.data.DataLoader(test, 500, shuffle=False, num_workers=workers, pin_memory=True))


@torch.no_grad()
def evaluate(model, loader, device):
    """(정확도 %, 예측 텐서, 로짓 텐서)"""
    model.eval()
    preds, logits, labels = [], [], []
    for x, y in loader:
        o = model(x.to(device)).float().cpu()
        logits.append(o); preds.append(o.argmax(1)); labels.append(y)
    preds, logits, labels = torch.cat(preds), torch.cat(logits), torch.cat(labels)
    return (preds == labels).float().mean().item() * 100, preds, logits


def calib_loader(n_per_class=100, seed=0, batch_size=250):
    """캘리브레이션용: 학습 세트에서 클래스당 n_per_class 장 (증강 없음, 시드 고정)."""
    te = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train = torchvision.datasets.CIFAR10(ROOT, train=True, download=True, transform=te)
    g = torch.Generator().manual_seed(seed)
    targets = torch.tensor(train.targets)
    idx = []
    for c in range(10):
        ids = (targets == c).nonzero().flatten()
        idx += ids[torch.randperm(len(ids), generator=g)[:n_per_class]].tolist()
    return torch.utils.data.DataLoader(torch.utils.data.Subset(train, sorted(idx)), batch_size, shuffle=False)
