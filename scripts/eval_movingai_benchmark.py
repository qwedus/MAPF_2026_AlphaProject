"""Evaluate a trained BC policy on the standard MovingAI MAPF benchmark.

Uses the MovingAI (Sturtevant) benchmark maps/scenarios that MAPF papers use,
rather than our synthetic il_smoke maps. Two metrics per instance:

  1) action-accuracy : along the CBS-optimal trajectory, how often does the
     policy pick the same action as the CBS expert (needs CBS to solve = only
     feasible on small maps / few agents with our atb033 adapter).
  2) success-rate    : roll out the policy in MAPFStepSimulator and check whether
     all agents reach their goals collision-free within max_steps. This is the
     metric MAPF papers report; it needs the step simulator (now available).

.map format : header (type/height/width/map) + rows of '.'(free) '@'/'T'(wall).
.scen format: tab cols  bucket map w h start_x start_y goal_x goal_y opt_len
              (x = col, y = row).

Usage:
  py scripts/eval_movingai_benchmark.py --bench-dir <scratch>/mapf_bench \\
     --map empty-8-8 --agents 2 4 6 --instances 20 --ckpt cnn_diverse.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spec
from model_cnn import ActionCNN
from simulator import MAPFStepSimulator


def parse_map(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    h = int(next(l for l in lines if l.startswith("height")).split()[1])
    w = int(next(l for l in lines if l.startswith("width")).split()[1])
    start = lines.index("map") + 1
    grid = np.zeros((h, w), dtype=int)
    for r, row in enumerate(lines[start:start + h]):
        for c, ch in enumerate(row[:w]):
            grid[r, c] = 0 if ch == "." else 1
    return grid


def parse_scen(path: Path):
    """Return list of ((start_row,start_col),(goal_row,goal_col)) in file order."""
    agents = []
    for line in path.read_text().splitlines():
        if line.startswith("version") or not line.strip():
            continue
        p = line.split("\t")
        sx, sy, gx, gy = int(p[4]), int(p[5]), int(p[6]), int(p[7])
        agents.append(((sy, sx), (gy, gx)))  # (row,col)
    return agents


def load_policy(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ActionCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["goal_mean"], ckpt["goal_std"]


@torch.no_grad()
def policy_actions(model, gmean, gstd, obs):
    acts = {}
    for aid, st in obs.items():
        g = torch.from_numpy(st["grid"]).float().unsqueeze(0)
        d = (torch.from_numpy(st["goal_dir"]).float().unsqueeze(0) - gmean) / gstd
        acts[aid] = int(model((g, d)).argmax(1))
    return acts


def eval_instance(model, gmean, gstd, grid, starts, goals, max_steps):
    """Return (action_matches, action_total, success:bool) for one instance."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)

    # --- metric 1: action-accuracy along CBS-optimal trajectory ---
    matches = total = 0
    try:
        obs = sim.reset(grid, starts, goals)
        for _ in range(max_steps):
            expert = sim.get_expert_actions(obs)          # CBS re-solve
            pred = policy_actions(model, gmean, gstd, obs)
            for aid in obs:
                total += 1
                matches += int(pred[aid] == expert[aid])
            obs, done, info = sim.step(expert)            # walk expert path
            if done:
                break
    except Exception:
        matches = total = 0  # CBS failed/timeout -> no action-acc for this instance

    # --- metric 2: success-rate by rolling out the POLICY ---
    obs = sim.reset(grid, starts, goals)
    success = False
    for _ in range(max_steps):
        pred = policy_actions(model, gmean, gstd, obs)
        obs, done, info = sim.step(pred)
        if done:
            success = bool(info["all_at_goal"])
            break
    return matches, total, success


def dedup_agents(agents, n):
    """Take first n agents with unique start and unique goal cells."""
    picked, seen_s, seen_g = [], set(), set()
    for s, g in agents:
        if s in seen_s or g in seen_g or s == g:
            continue
        seen_s.add(s); seen_g.add(g); picked.append((s, g))
        if len(picked) == n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--map", default="empty-8-8")
    ap.add_argument("--agents", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--ckpt", type=Path, default=PROJECT_ROOT / "cnn_diverse.pt")
    ap.add_argument("--max-steps", type=int, default=64)
    args = ap.parse_args()

    map_path = next(args.bench_dir.rglob(f"{args.map}.map"))
    scen_paths = sorted(args.bench_dir.rglob(f"{args.map}-random-*.scen"))[: args.instances]
    grid = parse_map(map_path)
    model, gmean, gstd = load_policy(args.ckpt)
    print(f"map={args.map} {grid.shape}  walls={int(grid.sum())}  "
          f"scen files={len(scen_paths)}  ckpt={args.ckpt.name}")

    print(f"\n{'agents':>6} {'instances':>9} {'action-acc':>11} {'success-rate':>13}")
    results = []
    for n in args.agents:
        m = t = succ = cnt = 0
        for sp in scen_paths:
            picked = dedup_agents(parse_scen(sp), n)
            if len(picked) < n:
                continue
            starts = {i: picked[i][0] for i in range(n)}
            goals = {i: picked[i][1] for i in range(n)}
            mi, ti, si = eval_instance(model, gmean, gstd, grid, starts, goals, args.max_steps)
            m += mi; t += ti; succ += int(si); cnt += 1
        acc = m / t if t else float("nan")
        sr = succ / cnt if cnt else float("nan")
        results.append((n, cnt, acc, sr))
        print(f"{n:>6} {cnt:>9} {acc:>11.3f} {sr:>13.3f}")

    return results


if __name__ == "__main__":
    main()
