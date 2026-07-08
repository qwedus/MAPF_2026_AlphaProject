"""Full DAgger training on top of the BC-pretrained CNN.

Starts from cnn_diverse.pt (best BC model), rolls out the current policy in
MAPFStepSimulator, relabels visited states with the CBS expert, aggregates, and
retrains — repeated for several iterations.

Scenarios are drawn ONLY from the CBS-solvable pool (scenarios_diverse entries
that succeeded in the CBS batch), so get_expert_actions never hits a long
timeout. Each episode is wrapped in try/except so a rare failure is skipped
rather than crashing the whole run.

Held-out action-accuracy (real_v03, no CBS needed) is logged every iteration to
track progress cheaply. Final model -> cnn_dagger.pt.

Usage:
  py scripts/train_dagger.py --iters 15 --scen-per-iter 40 --max-steps 64
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spec
from model_cnn import ActionCNN
from dagger import DAggerTrainer
from simulator import MAPFStepSimulator
from src.scenario_loader import load_map_generator_scenario


# CBS re-solves fast on these; dense/maze and large crowds can make an
# intermediate rollout state very slow, which stalls the whole DAgger run.
EASY_TYPES = {"empty", "sparse", "rooms"}


def _scen_meta(source_path):
    """Parse 'scen_<type>_s<size>_n<agents>_v<k>' from the scenario filename."""
    stem = Path(source_path).stem.replace("scenario_", "").replace("scen_", "")
    parts = stem.split("_")
    mtype = parts[0]
    size = int(next((p[1:] for p in parts if p.startswith("s") and p[1:].isdigit()), 0))
    n = int(next((p[1:] for p in parts if p.startswith("n") and p[1:].isdigit()), 0))
    return mtype, size, n


def load_solvable_scenarios(paths_globs, max_size=99, max_agents=3):
    """Return list of (grid, starts_dict, goals_dict) for CBS-solved scenarios.

    Keeps only empty/sparse/rooms (fast CBS) within the size/agent caps so per-step
    expert re-solves stay tractable. paths_globs may be one glob or a list.
    Large maps with FEW agents (bigmap) and small maps with MANY agents (dense) are
    both CBS-cheap per step, so the 'extended' pool broadens coverage without
    blowing up the re-solve cost.
    """
    if isinstance(paths_globs, str):
        paths_globs = [paths_globs]
    scenarios = []
    for paths_glob in paths_globs:
        for status_path in glob.glob(paths_glob):
            with open(status_path) as f:
                status = json.load(f)
            if not status.get("success"):
                continue
            mtype, size, n = _scen_meta(status["source_path"])
            if mtype not in EASY_TYPES or size > max_size or n > max_agents:
                continue
            try:
                grid, starts, goals = load_map_generator_scenario(status["source_path"])
            except Exception:
                continue
            grid = np.asarray(grid)
            starts_d = {i: tuple(starts[i]) for i in range(len(starts))}
            goals_d = {i: tuple(goals[i]) for i in range(len(goals))}
            scenarios.append((grid, starts_d, goals_d))
    return scenarios


@torch.no_grad()
def heldout_acc(model, gmean, gstd, npz_path, device):
    model.eval()
    d = np.load(npz_path)
    grids = torch.from_numpy(d[spec.KEY_GRID]).float()
    goals = (torch.from_numpy(d[spec.KEY_GOAL]).float() - gmean) / gstd  # CPU stats
    acts = torch.from_numpy(d[spec.KEY_ACT]).long()
    pred = model((grids.to(device), goals.to(device))).argmax(1).cpu()
    return float((pred == acts).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", default="cnn_diverse.pt")
    ap.add_argument("--out-ckpt", default="cnn_dagger.pt")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--scen-per-iter", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--heldout", default="real_v03.npz")
    ap.add_argument("--cbs-timeout", type=int, default=8,
                    help="per-step CBS timeout cap; short so a hard state fails fast")
    ap.add_argument("--extended", action="store_true",
                    help="broaden pool: diverse+dense+bigmap, up to --max-agents (large-map "
                         "few-agent + small-map many-agent), for large/congested coverage")
    ap.add_argument("--max-agents", type=int, default=3, help="agent cap for the pool")
    ap.add_argument("--save-iters", action="store_true",
                    help="also snapshot each iteration to <out>_iterNN.pt. DAgger's gain "
                         "(deadlock recovery) is NOT captured by held-out acc, so pick the "
                         "best iter by success/deadlock eval afterward, not by acc.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.init_ckpt, map_location=device)
    model = ActionCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    # keep normalization stats on CPU: DAggerTrainer.train() subtracts them from
    # CPU buffers, and _predict moves them to device per-call as needed.
    gmean, gstd = ckpt["goal_mean"].cpu(), ckpt["goal_std"].cpu()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=args.max_steps,
                            cbs_timeout_sec=args.cbs_timeout)
    trainer = DAggerTrainer(model, opt, sim, mode="cnn", device=device,
                            goal_mean=gmean, goal_std=gstd)

    if args.extended:
        globs = ["outputs/paths/diverse/*/status.json",
                 "outputs/paths/dense/*/status.json",
                 "outputs/paths/bigmap/*/status.json"]
    else:
        globs = ["outputs/paths/diverse/*/status.json"]
    pool = load_solvable_scenarios(globs, max_agents=args.max_agents)
    base_acc = heldout_acc(model, gmean, gstd, args.heldout, device)
    print(f"solvable scenarios={len(pool)}  device={device}")
    print(f"[init] BC held-out acc={base_acc:.3f}  (target: DAgger should keep/raise this)")
    print(f"config: iters={args.iters} scen/iter={args.scen_per_iter} "
          f"max_steps={args.max_steps} train_epochs={args.train_epochs}\n")

    t_start = time.time()
    for it in range(1, args.iters + 1):
        batch = random.sample(pool, min(args.scen_per_iter, len(pool)))
        t0 = time.time()
        collected = skipped = 0
        for grid, starts, goals in batch:
            try:
                collected += trainer.collect(grid, starts, goals, max_steps=args.max_steps)
            except Exception:
                skipped += 1
        model.train()
        trainer.train(epochs=args.train_epochs)  # prints per-epoch loss/acc
        ho = heldout_acc(model, gmean, gstd, args.heldout, device)
        dt = time.time() - t0
        print(f"[iter {it:02d}/{args.iters}] collected={collected} skipped={skipped} "
              f"buffer={trainer.n_samples} held-out acc={ho:.3f}  ({dt:.0f}s)")
        payload = {"model_state": model.state_dict(), "mode": "cnn",
                   "goal_mean": gmean, "goal_std": gstd, "iter": it, "heldout_acc": ho}
        torch.save(payload, args.out_ckpt)
        if args.save_iters:  # keep every iter so the best can be chosen by success/deadlock eval
            stem = Path(args.out_ckpt)
            torch.save(payload, stem.with_name(f"{stem.stem}_iter{it:02d}{stem.suffix}"))

    total = time.time() - t_start
    final_acc = heldout_acc(model, gmean, gstd, args.heldout, device)
    print(f"\n완료: {args.out_ckpt}  held-out {base_acc:.3f} -> {final_acc:.3f}  "
          f"(총 {total/60:.1f}분, 누적 샘플 {trainer.n_samples})")


if __name__ == "__main__":
    main()
