"""Prototype eval: straight-line goal_dir vs BFS-flow goal_dir.

Same architecture, same ~47k training data — the only difference is the goal_dir
feature. At rollout, the flow model's obs goal_dir is overwritten with the BFS
flow step (next move toward goal around walls), computed from the static map and
the agent's current position. Reports success / reached (%agents at goal) /
rem (mean remaining BFS distance to goal) so we can see navigation progress even
when success is still 0 (the maze/room regime).

Usage:
  py scripts/eval_flow.py --bench-dir <bench> \\
     --maps empty-8-8 random-32-32-10 room-32-32-4 maze-32-32-2 --agents 4 8 \\
     --straight-ckpt cnn_bigdense.pt --flow-ckpt cnn_flow.pt --instances 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import MAPFStepSimulator
from src.nav_hint import bfs_dist, make_goal_dir
from src.model_navhint import load_navhint
from scripts.eval_movingai_benchmark import parse_map, parse_scen, dedup_agents
from scripts.eval_cbs_vs_il import load_model, predict


def rollout(model, grid, sd, gd, max_steps, hint_mode=None):
    """hint_mode None = use the simulator's straight goal_dir (baseline). Otherwise
    overwrite each agent's goal_dir with make_goal_dir(...) in that mode."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, sd, gd)
    dist = {aid: bfs_dist(grid, gd[aid]) for aid in gd}  # for hint and metric
    success = False
    for _ in range(max_steps):
        if hint_mode is not None:
            for aid in obs:
                obs[aid]["goal_dir"] = np.asarray(
                    make_goal_dir(dist[aid], sim._positions[aid], gd[aid], hint_mode),
                    dtype=np.float32)
        obs, done, info = sim.step(predict(model, obs))
        if done:
            success = bool(info["all_at_goal"]); break
    pos = sim._positions
    reached = float(np.mean([1.0 if tuple(pos[a]) == tuple(gd[a]) else 0.0 for a in gd]))
    # remaining true path distance (BFS); -1 (unreachable cell) clamped to a big number
    rem = 0.0
    for a in gd:
        d = dist[a][pos[a][0], pos[a][1]]
        rem += float(d) if d >= 0 else float(dist[a].max() + 1)
    return success, reached, rem / len(gd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--maps", nargs="+", default=["empty-8-8", "random-32-32-10",
                                                  "room-32-32-4", "maze-32-32-2"])
    ap.add_argument("--agents", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--straight-ckpt", default="cnn_bigdense.pt")
    ap.add_argument("--flow-ckpt", default="cnn_navhint.pt")
    ap.add_argument("--max-steps", type=int, default=160)
    args = ap.parse_args()

    straight = load_model(PROJECT_ROOT / args.straight_ckpt)   # v0.3 2-D goal_dir
    flow = load_navhint(PROJECT_ROOT / args.flow_ckpt)          # navhint (2 or 4-D)
    hint_mode = flow["hint_mode"]
    print(f"straight={args.straight_ckpt}  flow={args.flow_ckpt} (hint={hint_mode}, dim={flow['goal_dim']})")
    print("(succ / reached=%agents at goal / rem=mean remaining BFS dist, lower=better)\n")

    for mapname in args.maps:
        grid = parse_map(next(args.bench_dir.rglob(f"{mapname}.map")))
        scen_paths = sorted(args.bench_dir.rglob(f"{mapname}-random-*.scen"))[: args.instances]
        for n in args.agents:
            insts = [dedup_agents(parse_scen(sp), n) for sp in scen_paths]
            insts = [p for p in insts if len(p) == n]
            if not insts:
                continue
            agg = {"straight": [0.0, 0.0, 0.0], "flow": [0.0, 0.0, 0.0]}
            for picked in insts:
                sd = {i: picked[i][0] for i in range(n)}
                gd = {i: picked[i][1] for i in range(n)}
                for name, model, hm in [("straight", straight, None), ("flow", flow, hint_mode)]:
                    s, r, rm = rollout(model, grid, sd, gd, args.max_steps, hm)
                    agg[name][0] += s; agg[name][1] += r; agg[name][2] += rm
            k = len(insts)
            print(f"### {mapname}  agents={n}  inst={k}")
            for name in ("straight", "flow"):
                s, r, rm = agg[name]
                print(f"    {name:9} succ={s/k:.2f}  reached={r/k:.2f}  rem={rm/k:6.1f}")
            print()


if __name__ == "__main__":
    main()
