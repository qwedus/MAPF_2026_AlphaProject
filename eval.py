# -*- coding: utf-8 -*-
"""
학습된 체크포인트를 npz로 평가.

    python eval.py --ckpt mlp.pt --npz dummy_v02.npz
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import spec
from dataset import MAPFILDataset
from infer import MAPFPredictor


def confusion_matrix(y_true, y_pred, n=spec.NUM_ACTIONS):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_report(cm):
    names = [spec.ACTION_NAMES[i] for i in range(spec.NUM_ACTIONS)]
    header = f"{'':6s}" + "".join(f"{n:>6s}" for n in names)
    print("\n[Confusion Matrix]  행=정답, 열=예측")
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:6s}" + "".join(f"{v:6d}" for v in row))

    print("\n[Per-action 지표]")
    print(f"{'action':8s}  {'precision':>10s}  {'recall':>8s}  {'F1':>8s}  {'support':>8s}")
    for i in range(spec.NUM_ACTIONS):
        tp  = cm[i, i]
        sup = cm[i].sum()
        p   = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r   = tp / sup if sup else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"{spec.ACTION_NAMES[i]:8s}  {p:10.3f}  {r:8.3f}  {f1:8.3f}  {sup:8d}")

    acc = cm.diagonal().sum() / cm.sum()
    print(f"\n전체 accuracy: {acc:.4f}  (무작위 기준 0.2000)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz",  required=True)
    args = ap.parse_args()

    pred = MAPFPredictor(args.ckpt)
    ds   = MAPFILDataset(args.npz, mode=pred.mode)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_true, all_pred = [], []
    for batch in loader:
        if pred.mode == "mlp":
            x, y = batch
            grids = x[:, :spec.GRID_FLAT].reshape(-1, spec.GRID_C, spec.GRID_H, spec.GRID_W).numpy()
            goals = x[:, spec.GRID_FLAT:].numpy()
            # dataset이 이미 정규화했으므로 역정규화 후 predictor에게 넘긴다
            goals_raw = goals * ds.goal_std.numpy() + ds.goal_mean.numpy()
        else:
            (grid_t, goal_t), y = batch
            grids = grid_t.numpy()
            goals_raw = (goal_t * ds.goal_std + ds.goal_mean).numpy()

        preds = pred.predict_batch(grids, goals_raw)
        all_true.extend(y.numpy())
        all_pred.extend(preds)

    cm = confusion_matrix(all_true, all_pred)
    print_report(cm)


if __name__ == "__main__":
    main()
