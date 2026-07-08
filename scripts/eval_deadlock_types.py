"""Classify WHY stuck agents are stuck — coordination vs navigation.

eval_deadlock tells us HOW MANY failures are deadlocks. This tells us WHAT KIND.
At the end of a failed rollout, for each agent not at its goal we look at the
cell in its straight-line goal direction (the move it "wants" to make) and read
that cell from its own 5x5 observation:

  robot  : blocked by another robot          -> COORDINATION (needs yielding)
  goal   : blocked by another agent's goal    -> COORDINATION (target-cell contention)
  wall   : blocked by a wall/obstacle         -> NAVIGATION  (straight line hits wall; must detour)
  open   : intended cell is free but stuck     -> NAVIGATION  (oscillating / lost)

Story: data scaling / DAgger should shrink the COORDINATION share; the wall/open
(NAVIGATION) share is the goal_dir-can't-express-detours limit that data alone
does not fix.

Usage:
  py scripts/eval_deadlock_types.py --bench-dir <bench> \\
     --maps empty-8-8 random-32-32-20 maze-32-32-2 --agents 6 \\
     --ckpts cnn.pt cnn_bigdense.pt --instances 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import MAPFStepSimulator
from scripts.eval_movingai_benchmark import parse_map, parse_scen, dedup_agents
from scripts.eval_cbs_vs_il import load_model, predict

CENTER = 2  # 5x5 window center = the agent itself


def intended_cell(goal_dir):
    """Dominant straight-line step (row,col delta) toward goal; (0,0) if at goal."""
    dr, dc = float(goal_dir[0]), float(goal_dir[1])
    if dr == 0 and dc == 0:
        return 0, 0
    if abs(dr) >= abs(dc):
        return (1 if dr > 0 else -1), 0
    return 0, (1 if dc > 0 else -1)


NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def classify_stuck(obs):
    """For each agent not at goal, classify what blocks it.

    First look at the intended (straight-line) cell; if that is open, fall back to
    "is any other robot in my 4-neighborhood?" — an open straight-line cell can
    still be a coordination jam (robots jostling from the side), which the single
    intended cell misses. Buckets: wall/open = NAVIGATION, robot/robot_adj/goal =
    COORDINATION."""
    counts = {"robot": 0, "robot_adj": 0, "goal": 0, "wall": 0, "open": 0}
    for st in obs.values():
        gd = st["goal_dir"]
        if abs(gd[0]) + abs(gd[1]) == 0:
            continue  # at goal, not stuck
        dr, dc = intended_cell(gd)
        grid = st["grid"]
        if grid[0, CENTER + dr, CENTER + dc] == 1:
            counts["wall"] += 1
        elif grid[1, CENTER + dr, CENTER + dc] == 1:
            counts["robot"] += 1
        elif grid[2, CENTER + dr, CENTER + dc] == 1:
            counts["goal"] += 1
        elif any(grid[1, CENTER + a, CENTER + b] == 1 for a, b in NEIGHBORS):
            counts["robot_adj"] += 1  # straight-line open but a robot is adjacent
        else:
            counts["open"] += 1
    return counts


def rollout_stuck_types(m, grid, sd, gd, max_steps, stall_window=12):
    """Roll out; if it ends in a deadlock, return the blocker classification."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, sd, gd)
    hist = [sum(float(abs(s["goal_dir"]).sum()) for s in obs.values())]
    success = False
    for _ in range(max_steps):
        obs, done, info = sim.step(predict(m, obs))
        hist.append(sum(float(abs(s["goal_dir"]).sum()) for s in obs.values()))
        if done:
            success = bool(info["all_at_goal"]); break
    if success:
        return None  # solved, no deadlock
    deadlock = len(hist) > stall_window and min(hist[-stall_window:]) >= hist[-stall_window] - 1e-9
    if not deadlock:
        return None  # failed but still progressing (slow), not a deadlock
    return classify_stuck(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--maps", nargs="+", default=["empty-8-8", "random-32-32-20", "maze-32-32-2"])
    ap.add_argument("--agents", type=int, nargs="+", default=[6])
    ap.add_argument("--instances", type=int, default=25)
    ap.add_argument("--ckpts", nargs="+", default=["cnn.pt", "cnn_bigdense.pt"])
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--out", default="deadlock_types.json")
    args = ap.parse_args()

    models = [load_model(PROJECT_ROOT / c) for c in args.ckpts]
    print(f"maps={args.maps} agents={args.agents} models={[m['name'] for m in models]}")
    print("(of STUCK agents in deadlocked episodes: NAV=facing a wall in goal dir "
          "(detour/representation limit); COORD=robot-jam or oscillation (data/DAgger can help))\n")

    results = []
    for mapname in args.maps:
        grid = parse_map(next(args.bench_dir.rglob(f"{mapname}.map")))
        scen_paths = sorted(args.bench_dir.rglob(f"{mapname}-random-*.scen"))[: args.instances]
        for n in args.agents:
            insts = [dedup_agents(parse_scen(sp), n) for sp in scen_paths]
            insts = [p for p in insts if len(p) == n]
            if not insts:
                continue
            print(f"### {mapname}  agents={n}  inst={len(insts)}")
            for m in models:
                tot = {"robot": 0, "robot_adj": 0, "goal": 0, "wall": 0, "open": 0}
                dl_episodes = 0
                for picked in insts:
                    sd = {i: picked[i][0] for i in range(n)}
                    gd = {i: picked[i][1] for i in range(n)}
                    c = rollout_stuck_types(m, grid, sd, gd, args.max_steps)
                    if c is not None:
                        dl_episodes += 1
                        for k in tot:
                            tot[k] += c[k]
                s = sum(tot.values())
                if s == 0:
                    print(f"    {m['name']:18} (no deadlocks)")
                    continue
                # NAV = facing a wall in the goal direction (straight-line-into-wall,
                # the representation limit). COORD = everything else: a robot blocks
                # (robot/robot_adj) or pure oscillation with no static blocker (open).
                nav = tot["wall"] / s
                coord = 1.0 - nav
                robot_share = (tot["robot"] + tot["robot_adj"]) / s
                print(f"    {m['name']:18} DLep={dl_episodes:2d} stuck={s:3d}  "
                      f"NAV/wall={nav:.2f}   COORD={coord:.2f} "
                      f"(robot-jam {robot_share:.2f}, oscillation {tot['open']/s:.2f})")
                results.append({"map": mapname, "n": n, "model": m["name"],
                                "dl_episodes": dl_episodes, "stuck_agents": s,
                                "coord": coord, "nav": nav, **{k: tot[k] / s for k in tot}})
            print()

    out = PROJECT_ROOT / "outputs" / args.out
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
