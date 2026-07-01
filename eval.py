# -*- coding: utf-8 -*-
"""
학습된 체크포인트를 npz로 평가.

단일 체크포인트 (기존과 동일, confusion matrix 상세 리포트):
    python eval.py --ckpt mlp.pt --npz real_v02.npz

여러 체크포인트 비교 (아키텍처×학습방식 4칸 매트릭스):
    python eval.py --npz real_v02.npz \
        --ckpt mlp.pt=MLP-BC --ckpt cnn.pt=CNN-BC \
        --ckpt mlp_dagger.pt=MLP-DAgger --ckpt cnn_dagger.pt=CNN-DAgger

라벨(=이후)을 생략하면 체크포인트의 mode와 파일명으로 자동 추정한다
(파일명에 "dagger"가 들어있으면 DAgger, 아니면 BC).
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


def per_action_metrics(cm):
    """action별 (precision, recall, f1, support) 딕셔너리."""
    out = {}
    for i in range(spec.NUM_ACTIONS):
        tp  = cm[i, i]
        sup = cm[i].sum()
        p   = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r   = tp / sup if sup else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) else 0.0
        out[i] = {"precision": p, "recall": r, "f1": f1, "support": int(sup)}
    return out


def print_report(cm, label=""):
    names = [spec.ACTION_NAMES[i] for i in range(spec.NUM_ACTIONS)]
    header = f"{'':6s}" + "".join(f"{n:>6s}" for n in names)
    title = f"[{label}] " if label else ""
    print(f"\n{title}[Confusion Matrix]  행=정답, 열=예측")
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:6s}" + "".join(f"{v:6d}" for v in row))

    metrics = per_action_metrics(cm)
    print("\n[Per-action 지표]")
    print(f"{'action':8s}  {'precision':>10s}  {'recall':>8s}  {'F1':>8s}  {'support':>8s}")
    for i in range(spec.NUM_ACTIONS):
        m = metrics[i]
        print(f"{spec.ACTION_NAMES[i]:8s}  {m['precision']:10.3f}  {m['recall']:8.3f}  "
              f"{m['f1']:8.3f}  {m['support']:8d}")

    acc = cm.diagonal().sum() / cm.sum()
    print(f"\n전체 accuracy: {acc:.4f}  (무작위 기준 0.2000)")


def evaluate_checkpoint(ckpt_path: str, npz_path: str):
    """체크포인트 하나를 npz로 평가해 confusion matrix를 반환."""
    pred = MAPFPredictor(ckpt_path)
    ds   = MAPFILDataset(npz_path, mode=pred.mode)
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

    return confusion_matrix(all_true, all_pred), pred.mode


def _auto_label(path: str, mode: str) -> str:
    method = "DAgger" if "dagger" in path.lower() else "BC"
    return f"{mode.upper()}-{method}"


def print_comparison(rows):
    """rows: [(label, cm), ...] → 아키텍처×학습방식 요약표."""
    print("\n=== 비교 요약 ===")
    header = f"{'label':16s}  {'accuracy':>8s}  {'macro-F1':>8s}"
    header += "".join(f"  {spec.ACTION_NAMES[i]+'-F1':>8s}" for i in range(spec.NUM_ACTIONS))
    print(header)
    for label, cm in rows:
        metrics = per_action_metrics(cm)
        acc = cm.diagonal().sum() / cm.sum()
        macro_f1 = sum(m["f1"] for m in metrics.values()) / spec.NUM_ACTIONS
        line = f"{label:16s}  {acc:8.4f}  {macro_f1:8.4f}"
        line += "".join(f"  {metrics[i]['f1']:8.3f}" for i in range(spec.NUM_ACTIONS))
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, action="append",
                     help="체크포인트 경로. 여러 번 지정 가능: --ckpt path 또는 --ckpt path=Label")
    ap.add_argument("--npz",  required=True, help="모든 체크포인트를 평가할 공통 npz (같은 held-out set 권장)")
    args = ap.parse_args()

    rows = []
    for raw in args.ckpt:
        path, label = raw.split("=", 1) if "=" in raw else (raw, None)
        cm, mode = evaluate_checkpoint(path, args.npz)
        if label is None:
            label = _auto_label(path, mode)
        print_report(cm, label=label)
        rows.append((label, cm))

    if len(rows) > 1:
        print_comparison(rows)


if __name__ == "__main__":
    main()
