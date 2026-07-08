"""
BC 학습 루프. MLP/CNN 공통.
실행: python train.py --mode mlp   (또는 --mode cnn)
GT 없이 더미로 돌려서 loss 하강 / acc 상승을 확인하는 게 1차 목표.
"""
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import spec
from dataset import MAPFILDataset
from model_mlp import ActionMLP
from model_cnn import ActionCNN


def collate_passthrough(batch):
    """CNN 모드: ((grid,goal), action) 배치를 텐서로 묶는다."""
    grids = torch.stack([b[0][0] for b in batch])
    goals = torch.stack([b[0][1] for b in batch])
    acts  = torch.stack([b[1] for b in batch])
    return (grids, goals), acts


def run_epoch(model, loader, mode, device, optim=None):
    train = optim is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    tot_loss = tot_correct = tot_n = 0
    for x, y in loader:
        if mode == "mlp":
            x = x.to(device)
        else:
            x = (x[0].to(device), x[1].to(device))
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = crit(logits, y)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot_n += bs
    return tot_loss / tot_n, tot_correct / tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--npz", default="dummy_v02.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = MAPFILDataset(args.npz, mode=args.mode)
    n_val = max(1, int(len(ds) * 0.2))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    collate = collate_passthrough if args.mode == "cnn" else None
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, collate_fn=collate)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, collate_fn=collate)

    model = (ActionMLP() if args.mode == "mlp" else ActionCNN()).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[{args.mode}] device={device}  params={n_param:,}  "
          f"train={len(tr)} val={len(va)}")

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for ep in range(1, args.epochs + 1):
        trl, tra = run_epoch(model, tl, args.mode, device, optim)
        val, vaa = run_epoch(model, vl, args.mode, device, None)
        flag = ""
        if vaa > best_acc:   # 지금까지 최고 val acc → 스냅샷 보관 (과적합 전 시점)
            best_acc = vaa
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            flag = "  *best"
        print(f"ep{ep:02d}  train_loss={trl:.3f} acc={tra:.3f}  "
              f"| val_loss={val:.3f} acc={vaa:.3f}{flag}")

    # 무작위 베이스라인(5클래스 → 0.2) 대비 확인
    print(f"\n무작위 기준 acc=0.200 / 최종 val acc={vaa:.3f} "
          f"/ best val acc={best_acc:.3f} (ep{best_epoch})")
    torch.save({
        "model_state": best_state,   # 마지막이 아니라 best-val 시점 가중치
        "mode": args.mode,
        "goal_mean": ds.goal_mean,   # (1,2) — infer/eval에서 동일 정규화
        "goal_std":  ds.goal_std,    # (1,2)
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
    }, f"{args.mode}.pt")
    print(f"저장: {args.mode}.pt  (best-val, ep{best_epoch}, acc={best_acc:.3f})")


if __name__ == "__main__":
    main()
