"""Deadlock-focused comparison of BC vs DAgger in the congestion regime.

The DAgger hypothesis was: it should help specifically at *deadlocks* — states
where a BC policy drives agents into a stuck configuration. Whole-instance
success rate averages this out, so here we instrument the rollout to separate
"stuck (deadlock)" failures from "still progressing but timed out", and give
partial credit (fraction of agents that reached goal).

Per rollout we track total remaining Manhattan distance (sum of |goal_dir|_1 over
agents, read straight from the obs). Classification of a failed episode:
  - deadlock: remaining distance did NOT improve over the last STALL_WINDOW steps
              (agents frozen / oscillating) -> the "deadlock point" we expected.
  - timeout : still making progress at the end (would likely finish with more steps).

Metrics per (model, #agents): success rate, mean fraction of agents at goal,
deadlock rate, mean final remaining distance. No CBS, so it's fast.

Usage:
  py scripts/eval_deadlock.py --bench-dir <scratch>/mapf_bench --map empty-8-8 \\
     --agents 4 5 6 7 8 --instances 25 --ckpts cnn_diverse.pt cnn_dagger.pt
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

from simulator import MAPFStepSimulator
from scripts.eval_movingai_benchmark import parse_map, parse_scen, dedup_agents
from scripts.eval_cbs_vs_il import load_model, predict

STALL_WINDOW = 12  # steps of no net progress -> deadlock


def remaining(obs):
    return sum(float(abs(st["goal_dir"]).sum()) for st in obs.values())


def at_goal_frac(obs):
    return np.mean([1.0 if abs(st["goal_dir"]).sum() == 0 else 0.0 for st in obs.values()])


def rollout(m, grid, sd, gd, max_steps):
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, sd, gd)
    hist = [remaining(obs)]
    success = False
    for _ in range(max_steps):
        obs, done, info = sim.step(predict(m, obs))
        hist.append(remaining(obs))
        if done:
            success = bool(info["all_at_goal"]); break
    reached = at_goal_frac(obs)
    # deadlock: failed AND no net progress across the last window
    deadlock = False
    if not success and len(hist) > STALL_WINDOW:
        recent = hist[-STALL_WINDOW:]
        deadlock = (min(recent) >= recent[0] - 1e-9)  # never improved on the window start
    return success, reached, deadlock, hist[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--map", default="empty-8-8")
    ap.add_argument("--agents", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--instances", type=int, default=25)
    ap.add_argument("--ckpts", nargs="+", default=["cnn_diverse.pt", "cnn_dagger.pt"])
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()

    models = [load_model(PROJECT_ROOT / c) for c in args.ckpts]
    grid = parse_map(next(args.bench_dir.rglob(f"{args.map}.map")))
    scen_paths = sorted(args.bench_dir.rglob(f"{args.map}-random-*.scen"))[: args.instances]
    print(f"map={args.map}  models={[m['name'] for m in models]}  inst<={len(scen_paths)}")
    print(f"(succ=success, reached=mean %agents at goal, DL=deadlock rate among all, "
          f"rem=mean final remaining dist)\n")

    results = []
    for n in args.agents:
        insts = [dedup_agents(parse_scen(sp), n) for sp in scen_paths]
        insts = [p for p in insts if len(p) == n]
        print(f"### {args.map}  agents={n}  inst={len(insts)}")
        for m in models:
            succ = reached = dl = rem = 0.0
            for picked in insts:
                sd = {i: picked[i][0] for i in range(n)}
                gd = {i: picked[i][1] for i in range(n)}
                s, r, d, fr = rollout(m, grid, sd, gd, args.max_steps)
                succ += s; reached += r; dl += d; rem += fr
            k = len(insts)
            print(f"    {m['name']:16} succ={succ/k:.2f}  reached={reached/k:.2f}  "
                  f"DL={dl/k:.2f}  rem={rem/k:5.1f}")
            results.append({"map": args.map, "n": n, "model": m["name"],
                            "succ": succ/k, "reached": reached/k, "deadlock": dl/k, "rem": rem/k})
        print()

    import json
    out = PROJECT_ROOT / "outputs" / "deadlock.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")
    return results


if __name__ == "__main__":
    main()
