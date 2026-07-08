"""Generate report figures for the IL v0.3 work log.

Replicates train.py's training setup exactly (seed, split, model, optimizer) but
records per-epoch history and evaluates on a held-out npz, then renders PNGs into
docs/img/ for embedding in the work-log markdown.

Figures:
  fig_epoch_curves.png    - val accuracy per epoch, MLP vs CNN (overfit story)
  fig_confusion_heldout.png - held-out confusion matrices, MLP & CNN
  fig_dataset_stats.png   - dataset sizes + per-action distributions
  fig_data_expansion.png  - CNN held-out acc: baseline vs expanded (if given)

Labels are English on purpose (matplotlib has no Korean font here).

Usage:
  py scripts/plot_report_figures.py \\
     --baseline real_v03_full.npz --heldout real_v03.npz [--expanded combined.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spec
from dataset import MAPFILDataset
from model_mlp import ActionMLP
from model_cnn import ActionCNN
from train import collate_passthrough, run_epoch

ACTION_LABELS = ["Up", "Down", "Left", "Right", "Wait"]
IMG_DIR = PROJECT_ROOT / "docs" / "img"


def train_with_history(npz_path, mode, epochs, bs=64, lr=1e-3):
    """Same setup as train.py, but return (model, history, goal_mean, goal_std)."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MAPFILDataset(npz_path, mode=mode)
    n_val = max(1, int(len(ds) * 0.2))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    collate = collate_passthrough if mode == "cnn" else None
    tl = DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=bs, shuffle=False, collate_fn=collate)

    model = (ActionMLP() if mode == "mlp" else ActionCNN()).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    for _ in range(1, epochs + 1):
        trl, tra = run_epoch(model, tl, mode, device, optim)
        val, vaa = run_epoch(model, vl, mode, device, None)
        hist["train_acc"].append(tra); hist["val_acc"].append(vaa)
        hist["train_loss"].append(trl); hist["val_loss"].append(val)
    return model, hist, ds.goal_mean, ds.goal_std, device


@torch.no_grad()
def heldout_confusion(model, mode, npz_path, goal_mean, goal_std, device):
    """Confusion matrix (rows=true, cols=pred) on a held-out npz using train stats."""
    model.eval()
    data = np.load(npz_path)
    grids = torch.from_numpy(data[spec.KEY_GRID]).float()
    goals = torch.from_numpy(data[spec.KEY_GOAL]).float()
    acts = torch.from_numpy(data[spec.KEY_ACT]).long()
    goals_n = (goals - goal_mean) / goal_std

    if mode == "mlp":
        x = torch.cat([grids.reshape(len(grids), -1), goals_n], dim=1).to(device)
        logits = model(x)
    else:
        logits = model((grids.to(device), goals_n.to(device)))
    pred = logits.argmax(1).cpu()

    k = spec.NUM_ACTIONS
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(acts.tolist(), pred.tolist()):
        cm[t, p] += 1
    acc = float((pred == acts).float().mean())
    return cm, acc


def plot_epoch_curves(histories, epochs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, h in histories.items():
        xs = range(1, epochs + 1)
        axes[0].plot(xs, h["val_acc"], label=f"{mode.upper()} val", linewidth=2)
        axes[0].plot(xs, h["train_acc"], "--", label=f"{mode.upper()} train", alpha=0.5)
        axes[1].plot(xs, h["val_loss"], label=f"{mode.upper()} val", linewidth=2)
    axes[0].set_title("Accuracy per epoch (train dashed / val solid)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Validation loss per epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("MLP overfits early; CNN keeps improving", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_confusions(cms, out):
    fig, axes = plt.subplots(1, len(cms), figsize=(5.4 * len(cms), 4.6))
    if len(cms) == 1:
        axes = [axes]
    for ax, (label, cm, acc) in zip(axes, cms):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(ACTION_LABELS))); ax.set_yticks(range(len(ACTION_LABELS)))
        ax.set_xticklabels(ACTION_LABELS); ax.set_yticklabels(ACTION_LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{label}  (held-out acc={acc:.3f})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Held-out confusion matrices (real_v03)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_dataset_stats(datasets, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [d[0] for d in datasets]
    sizes = [d[1] for d in datasets]
    axes[0].bar(names, sizes, color="#4C78A8")
    axes[0].set_title("Dataset size (samples)")
    for i, v in enumerate(sizes):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    axes[0].tick_params(axis="x", rotation=15)

    x = np.arange(len(ACTION_LABELS)); w = 0.8 / len(datasets)
    for k, (name, _size, dist) in enumerate(datasets):
        frac = np.array(dist) / max(sum(dist), 1)
        axes[1].bar(x + k * w, frac, w, label=name)
    axes[1].set_xticks(x + w * (len(datasets) - 1) / 2)
    axes[1].set_xticklabels(ACTION_LABELS)
    axes[1].set_title("Action distribution (fraction)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_expansion(pairs, out):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    names = [p[0] for p in pairs]; accs = [p[1] for p in pairs]
    bars = ax.bar(names, accs, color=["#9ecae1", "#3182bd"])
    ax.set_ylim(0, 1); ax.set_ylabel("held-out accuracy (real_v03)")
    ax.set_title("CNN: effect of more/diverse data")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)


def action_dist(npz_path):
    d = np.load(npz_path)
    return int(len(d[spec.KEY_ACT])), np.bincount(d[spec.KEY_ACT], minlength=spec.NUM_ACTIONS).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="real_v03_full.npz")
    ap.add_argument("--heldout", default="real_v03.npz")
    ap.add_argument("--expanded", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) epoch curves + held-out confusion on baseline
    histories, cms = {}, []
    for mode in ("mlp", "cnn"):
        model, hist, gm, gs, dev = train_with_history(args.baseline, mode, args.epochs)
        histories[mode] = hist
        cm, acc = heldout_confusion(model, mode, args.heldout, gm, gs, dev)
        cms.append((mode.upper(), cm, acc))
        print(f"[{mode}] final val_acc={hist['val_acc'][-1]:.3f}  held-out acc={acc:.3f}")

    plot_epoch_curves(histories, args.epochs, IMG_DIR / "fig_epoch_curves.png")
    plot_confusions(cms, IMG_DIR / "fig_confusion_heldout.png")

    # 2) dataset stats
    datasets = []
    for name, path in [("il_smoke(303)", args.heldout), ("full_sweep(1632)", args.baseline)]:
        if Path(path).is_file():
            n, dist = action_dist(path); datasets.append((name, n, dist))
    if args.expanded and Path(args.expanded).is_file():
        n, dist = action_dist(args.expanded); datasets.append((f"diverse({n})", n, dist))
    plot_dataset_stats(datasets, IMG_DIR / "fig_dataset_stats.png")

    # 3) expansion effect (CNN baseline vs expanded)
    if args.expanded and Path(args.expanded).is_file():
        base_acc = cms[1][2]  # CNN baseline held-out
        model, hist, gm, gs, dev = train_with_history(args.expanded, "cnn", args.epochs)
        _, exp_acc = heldout_confusion(model, "cnn", args.heldout, gm, gs, dev)
        print(f"[cnn expanded] held-out acc={exp_acc:.3f}")
        plot_expansion(
            [(f"baseline\n(1632)", base_acc), (f"+diverse\n({action_dist(args.expanded)[0]})", exp_acc)],
            IMG_DIR / "fig_data_expansion.png",
        )
        # persist expanded CNN as the new best checkpoint
        torch.save({"model_state": model.state_dict(), "mode": "cnn",
                    "goal_mean": gm, "goal_std": gs}, PROJECT_ROOT / "cnn_diverse.pt")

    print(f"figures -> {IMG_DIR}")


if __name__ == "__main__":
    main()
