"""Build a v0.3-shape dataset where goal_dir is the BFS FLOW direction instead
of the straight-line vector — a prototype of the navigation-hint representation.

Reuses already-saved CBS outputs (no re-solving): for each successful scenario we
reload its map + goals + expert paths, then re-extract (states_grid, goal_dir,
action) exactly like dataset_exporter, but with goal_dir = flow_step (next step
toward goal around walls). states_grid and actions are byte-identical to the
straight dataset, so a straight-vs-flow comparison isolates the goal_dir change.

Does NOT modify any original file: it imports the existing grid/action helpers
and only swaps in the flow goal_dir.

Usage:
  py scripts/build_flow_dataset.py --pools diverse bigmap dense --out real_v03_flow.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spec
from src.dataset_exporter import (
    _action_id, _extract_local_grid, _location_at_time, _normalize_grid_map,
)
from src.nav_hint import bfs_dist, make_goal_dir
from src.scenario_loader import load_map_generator_scenario


def extract_flow(grid_map, goals, paths, mode="flow"):
    """Same iteration as dataset_exporter.extract_state_action_pairs, but goal_dir
    is a navigation-hint feature (see nav_hint.make_goal_dir). states_grid and
    actions are identical to the straight dataset. goals: list[(r,c)]."""
    grid = _normalize_grid_map(grid_map)
    h, w = grid.shape
    norm_paths = {int(a): [tuple(p) for p in path] for a, path in paths.items()}
    goals = {int(a): tuple(g) for a, g in enumerate(goals)}

    states, gdirs, acts = [], [], []
    for aid, path in sorted(norm_paths.items()):
        if len(path) < 2:
            continue
        dist = bfs_dist(grid, goals[aid])  # BFS once per agent (goal fixed)
        for t, (cur, nxt) in enumerate(zip(path, path[1:])):
            others_pos = [_location_at_time(op, t) for oa, op in sorted(norm_paths.items()) if oa != aid]
            others_goal = [goals[oa] for oa in sorted(norm_paths) if oa != aid]
            states.append(_extract_local_grid(grid, cur, others_pos, others_goal))
            gdirs.append(make_goal_dir(dist, cur, goals[aid], mode))
            acts.append(_action_id(cur, nxt))
    return states, gdirs, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["diverse", "bigmap", "dense"],
                    help="subdirs under outputs/paths/ to pull successful scenarios from")
    ap.add_argument("--out", default="real_v03_flow.npz")
    ap.add_argument("--mode", choices=["flow", "flowdist", "both"], default="flow",
                    help="goal_dir feature: flow(2) / flow*dist(2) / straight+flow(4)")
    args = ap.parse_args()

    all_s, all_g, all_a = [], [], []
    n_scen = n_fail = 0
    for pool in args.pools:
        for status_path in sorted(glob.glob(str(PROJECT_ROOT / "outputs/paths" / pool / "*/status.json"))):
            status = json.load(open(status_path, encoding="utf-8"))
            if not status.get("success"):
                continue
            try:
                grid, _starts, goals = load_map_generator_scenario(status["source_path"])
                paths = json.load(open(Path(status_path).with_name("paths.json"), encoding="utf-8"))["paths"]
                s, g, a = extract_flow(grid, goals, paths, args.mode)
            except Exception as exc:
                n_fail += 1
                continue
            if s:
                all_s.extend(s); all_g.extend(g); all_a.extend(a); n_scen += 1

    states = np.stack(all_s).astype(np.float32)
    gdirs = np.asarray(all_g, dtype=np.float32)
    acts = np.asarray(all_a, dtype=np.int64)
    out = PROJECT_ROOT / args.out
    np.savez(out, **{spec.KEY_GRID: states, spec.KEY_GOAL: gdirs, spec.KEY_ACT: acts})
    print(f"scenarios={n_scen} (skipped {n_fail})  samples={len(acts)}  "
          f"goal_dim={gdirs.shape[1]}  unique goal vecs={len(np.unique(gdirs, axis=0))}  -> {out}")


if __name__ == "__main__":
    main()
