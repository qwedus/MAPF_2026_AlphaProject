"""Train the NavHintCNN on a navigation-hint dataset (goal_dim 2 or 4).

Self-contained (does not use dataset.py's 2-D validation). Mirrors train.py:
80/20 split, Adam, best-val checkpoint. Saves goal_mean/std, hint_mode, label.

Usage:
  py scripts/train_navhint.py --npz real_v03_both.npz --hint-mode both \\
     --out cnn_navhint.pt --epochs 55
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spec
from src.model_navhint import NavHintCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hint-mode", default="both")
    ap.add_argument("--label", default=None)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(PROJECT_ROOT / args.npz)
    grids = torch.from_numpy(data[spec.KEY_GRID]).float()
    goals = torch.from_numpy(data[spec.KEY_GOAL]).float()
    acts = torch.from_numpy(data[spec.KEY_ACT]).long()
    goal_dim = goals.shape[1]

    gmean = goals.mean(0, keepdim=True)
    gstd = goals.std(0, keepdim=True).clamp_min(1e-6)
    goals_n = (goals - gmean) / gstd

    ds = TensorDataset(grids, goals_n, acts)
    n_val = max(1, int(len(ds) * 0.2))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False)

    model = NavHintCNN(goal_dim=goal_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    print(f"[navhint] goal_dim={goal_dim} device={device} train={len(tr)} val={len(va)}")

    def run(loader, train):
        model.train(train)
        tl_, tc_, tn_ = 0.0, 0, 0
        for g, d, y in loader:
            g, d, y = g.to(device), d.to(device), y.to(device)
            with torch.set_grad_enabled(train):
                logits = model((g, d))
                loss = crit(logits, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tl_ += loss.item() * y.size(0); tc_ += (logits.argmax(1) == y).sum().item(); tn_ += y.size(0)
        return tl_ / tn_, tc_ / tn_

    best_acc, best_state, best_ep = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(1, args.epochs + 1):
        _, tra = run(tl, True)
        _, vaa = run(vl, False)
        if vaa > best_acc:
            best_acc, best_state, best_ep = vaa, copy.deepcopy(model.state_dict()), ep
    print(f"best val acc={best_acc:.3f} (ep{best_ep})")

    torch.save({"model_state": best_state, "mode": "cnn", "goal_mean": gmean,
                "goal_std": gstd, "hint_mode": args.hint_mode,
                "label": args.label or Path(args.out).stem,
                "best_epoch": best_ep, "best_val_acc": best_acc},
               PROJECT_ROOT / args.out)
    print(f"저장: {args.out}  (best-val, ep{best_ep}, acc={best_acc:.3f})")


if __name__ == "__main__":
    main()
