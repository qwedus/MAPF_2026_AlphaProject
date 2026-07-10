"""DAgger on top of the navigation-hint (navhint) BC model.

Reuses the original DAggerTrainer and MAPFStepSimulator unchanged. The only new
piece is a thin wrapper that overwrites each obs's goal_dir with the navhint
feature (make_goal_dir) before DAgger sees it — so both the stored samples and
the policy's inputs use the new representation. Expert relabeling is delegated to
the real simulator (it reads its own positions, not the obs goal_dir).

Usage:
  py scripts/train_dagger_navhint.py --init-ckpt cnn_navhint.pt \\
     --out-ckpt cnn_dagger_navhint.pt --extended --max-agents 7 --iters 12
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dagger import DAggerTrainer
from simulator import MAPFStepSimulator
from src.nav_hint import bfs_dist, make_goal_dir
from src.model_navhint import NavHintCNN
from scripts.train_dagger import load_solvable_scenarios


class NavHintSim:
    """Wraps a MAPFStepSimulator so every obs carries the navhint goal_dir.
    Does not modify the wrapped simulator."""

    def __init__(self, real, hint_mode="both"):
        self.real = real
        self.hint_mode = hint_mode

    def reset(self, map_grid, starts, goals):
        obs = self.real.reset(map_grid, starts, goals)
        self._goals = {int(a): tuple(g) for a, g in goals.items()}
        grid = np.asarray(map_grid)
        self._dist = {a: bfs_dist(grid, self._goals[a]) for a in self._goals}
        return self._wrap(obs)

    def step(self, actions):
        obs, done, info = self.real.step(actions)
        return self._wrap(obs), done, info

    def get_expert_actions(self, obs):
        return self.real.get_expert_actions(obs)

    def _wrap(self, obs):
        for aid in obs:
            pos = self.real._positions[aid]
            obs[aid]["goal_dir"] = np.asarray(
                make_goal_dir(self._dist[aid], pos, self._goals[aid], self.hint_mode),
                dtype=np.float32)
        return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", default="cnn_navhint.pt")
    ap.add_argument("--out-ckpt", default="cnn_dagger_navhint.pt")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--scen-per-iter", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--cbs-timeout", type=int, default=8)
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--max-agents", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(PROJECT_ROOT / args.init_ckpt, map_location=device)
    goal_dim = int(ckpt["goal_mean"].shape[-1])
    hint_mode = ckpt.get("hint_mode", "both")
    model = NavHintCNN(goal_dim=goal_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    gmean, gstd = ckpt["goal_mean"].cpu(), ckpt["goal_std"].cpu()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    real = MAPFStepSimulator(cbs_solver_root=None, max_steps=args.max_steps,
                             cbs_timeout_sec=args.cbs_timeout)
    sim = NavHintSim(real, hint_mode=hint_mode)
    trainer = DAggerTrainer(model, opt, sim, mode="cnn", device=device,
                            goal_mean=gmean, goal_std=gstd)

    globs = (["outputs/paths/diverse/*/status.json", "outputs/paths/dense/*/status.json",
              "outputs/paths/bigmap/*/status.json"] if args.extended
             else ["outputs/paths/diverse/*/status.json"])
    pool = load_solvable_scenarios(globs, max_agents=args.max_agents)
    print(f"navhint DAgger  goal_dim={goal_dim} hint={hint_mode}  pool={len(pool)}  device={device}")
    print(f"config: iters={args.iters} scen/iter={args.scen_per_iter} max_steps={args.max_steps}\n")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        batch = random.sample(pool, min(args.scen_per_iter, len(pool)))
        collected = skipped = 0
        ti = time.time()
        for grid, starts, goals in batch:
            try:
                collected += trainer.collect(grid, starts, goals, max_steps=args.max_steps)
            except Exception:
                skipped += 1
        model.train()
        trainer.train(epochs=args.train_epochs)
        print(f"[iter {it:02d}/{args.iters}] collected={collected} skipped={skipped} "
              f"buffer={trainer.n_samples}  ({time.time()-ti:.0f}s)")
        payload = {"model_state": model.state_dict(), "mode": "cnn",
                   "goal_mean": gmean, "goal_std": gstd, "hint_mode": hint_mode,
                   "label": f"dagger-navhint({hint_mode})", "iter": it}
        torch.save(payload, PROJECT_ROOT / args.out_ckpt)

    print(f"\n완료: {args.out_ckpt}  (총 {(time.time()-t0)/60:.1f}분, 누적 {trainer.n_samples})")


if __name__ == "__main__":
    main()
