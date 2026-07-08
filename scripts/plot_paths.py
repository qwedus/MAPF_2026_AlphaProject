"""Render agent paths on a grid: CBS expert paths, and (optionally) an IL
policy's rollout on the SAME instance for side-by-side deadlock case studies.

Two figures:
  --mode expert   : CBS expert paths on a few map-generator scenarios
                    (shows "what the training labels look like"). -> fig_cbs_paths.png
  --mode case     : pick a MovingAI instance, run CBS + each --ckpts model,
                    draw CBS (solid) vs model rollout (dashed, X where stuck).
                    -> fig_case_study.png

CBS gives {agent_id: [(row,col), ...]}. IL rollout positions are read from the
simulator's internal state each step. Start = circle, goal = star, deadlock
(agent not at goal at the end) = big X.

Usage:
  py scripts/plot_paths.py --mode expert \\
     --scenarios scenarios_diverse/scenario_empty_s8_n3_v0.json ...
  py scripts/plot_paths.py --mode case --bench-dir <bench> \\
     --map empty-8-8 --agents 6 --scen-idx 0 --ckpts cnn.pt cnn_bigdense.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator import MAPFStepSimulator
from src.cbs_adapter import CBSAdapter, CBSAdapterConfig
from src.scenario_loader import load_map_generator_scenario

IMG_DIR = PROJECT_ROOT / "docs" / "img"


def agent_colors(n):
    cmap = cm.get_cmap("tab10" if n <= 10 else "tab20", max(n, 1))
    return [cmap(i) for i in range(n)]


import os
import tempfile

def run_cbs(grid, starts, goals, timeout=15):
    """Return {aid: [(r,c), ...]} or None on failure/timeout.

    Uses a unique work_dir so it never collides with a concurrent CBS caller
    (e.g. a background DAgger run sharing the default outputs/paths dir)."""
    work = Path(tempfile.mkdtemp(prefix=f"cbs_plot_{os.getpid()}_"))
    adapter = CBSAdapter(CBSAdapterConfig(solver_root=None, timeout_sec=timeout, work_dir=work))
    try:
        return adapter.plan(list(starts), list(goals), np.asarray(grid))
    except Exception as exc:
        print(f"  CBS failed: {exc}")
        return None


def rollout_positions(ckpt, grid, starts_d, goals_d, max_steps=160):
    """Roll out an IL policy, recording each agent's position per step.

    Returns (paths {aid: [(r,c)...]}, success bool, stuck set of aids not at goal).
    """
    from scripts.eval_cbs_vs_il import load_model, predict
    m = load_model(PROJECT_ROOT / ckpt)
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(np.asarray(grid), starts_d, goals_d)
    paths = {aid: [pos] for aid, pos in sim._positions.items()}
    success = False
    for _ in range(max_steps):
        obs, done, info = sim.step(predict(m, obs))
        for aid, pos in sim._positions.items():
            paths[aid].append(pos)
        if done:
            success = bool(info["all_at_goal"])
            break
    stuck = {aid for aid in goals_d if tuple(sim._positions[aid]) != tuple(goals_d[aid])}
    return paths, success, stuck, m["name"]


def draw_grid(ax, grid):
    grid = np.asarray(grid)
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1.6, origin="upper", interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, grid.shape[1] - 0.5); ax.set_ylim(grid.shape[0] - 0.5, -0.5)


def draw_paths(ax, paths, colors, linestyle="-", lw=2.2, alpha=0.9, jitter=0.0):
    """paths: {aid: [(row,col), ...]}. Draws a polyline through cell centers."""
    for aid in sorted(paths):
        pts = paths[aid]
        ys = [p[0] + jitter for p in pts]
        xs = [p[1] + jitter for p in pts]
        ax.plot(xs, ys, linestyle=linestyle, color=colors[aid % len(colors)],
                lw=lw, alpha=alpha, solid_capstyle="round", zorder=3)


def draw_endpoints(ax, starts_d, goals_d, colors, stuck=None):
    stuck = stuck or set()
    for aid in sorted(starts_d):
        c = colors[aid % len(colors)]
        sr, sc = starts_d[aid]; gr, gc = goals_d[aid]
        ax.scatter([sc], [sr], color=c, s=70, marker="o", edgecolors="black",
                   linewidths=0.6, zorder=4)
        ax.scatter([gc], [gr], color=c, s=150, marker="*", edgecolors="black",
                   linewidths=0.6, zorder=4)
        if aid in stuck:
            ax.scatter([sc], [sr], color=c, s=0)  # keep color cycle
    # mark stuck agents' final positions handled by caller via X


def plot_expert(scenarios, out):
    n = len(scenarios)
    cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    for k, scen in enumerate(scenarios):
        ax = axes[k // cols][k % cols]
        grid, starts, goals = load_map_generator_scenario(scen)
        starts_d = {i: tuple(starts[i]) for i in range(len(starts))}
        goals_d = {i: tuple(goals[i]) for i in range(len(goals))}
        colors = agent_colors(len(starts))
        draw_grid(ax, grid)
        cbs = run_cbs(grid, starts, goals, timeout=30)
        if cbs is not None:
            draw_paths(ax, cbs, colors, linestyle="-", lw=2.4)
            mk = max((len(p) for p in cbs.values()), default=1) - 1
            title = f"{Path(scen).stem.replace('scenario_','')}\nCBS makespan={mk}"
        else:
            title = f"{Path(scen).stem.replace('scenario_','')}\nCBS timeout"
        draw_endpoints(ax, starts_d, goals_d, colors)
        ax.set_title(title, fontsize=9)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("CBS expert paths (o=start, *=goal) — the supervision IL learns from", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def plot_case(grid, starts_d, goals_d, ckpts, out, max_steps=160):
    colors = agent_colors(len(starts_d))
    starts = [starts_d[i] for i in sorted(starts_d)]
    goals = [goals_d[i] for i in sorted(goals_d)]
    cbs = run_cbs(grid, starts, goals, timeout=20)

    panels = [("CBS expert", cbs, None)]
    for ck in ckpts:
        paths, success, stuck, name = rollout_positions(ck, grid, starts_d, goals_d, max_steps)
        tag = "solved" if success else f"stuck:{sorted(stuck)}"
        panels.append((f"{name}  ({tag})", paths, stuck))

    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(4.3 * ncol, 4.6), squeeze=False)
    for ax, (title, paths, stuck) in zip(axes[0], panels):
        draw_grid(ax, grid)
        if paths is not None:
            solid = stuck is None
            draw_paths(ax, paths, colors, linestyle="-" if solid else "--",
                       lw=2.4 if solid else 1.8, alpha=0.9 if solid else 0.85)
            if stuck:  # mark where each stuck agent ended up
                for aid in stuck:
                    er, ec = paths[aid][-1]
                    ax.scatter([ec], [er], color=colors[aid % len(colors)], s=180,
                               marker="X", edgecolors="black", linewidths=0.8, zorder=5)
        draw_endpoints(ax, starts_d, goals_d, colors)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Deadlock case study: CBS solution vs IL rollout "
                 "(o=start *=goal X=stuck; dashed=IL path)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["expert", "case"], required=True)
    ap.add_argument("--scenarios", nargs="+", help="map-gen scenario JSONs (expert mode)")
    ap.add_argument("--bench-dir", type=Path, help="MovingAI bench dir (case mode)")
    ap.add_argument("--map", default="empty-8-8")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--scen-idx", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", default=["cnn.pt", "cnn_bigdense.pt"])
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "expert":
        out = Path(args.out) if args.out else IMG_DIR / "fig_cbs_paths.png"
        plot_expert(args.scenarios, out)
    else:
        from scripts.eval_movingai_benchmark import parse_map, parse_scen, dedup_agents
        grid = parse_map(next(args.bench_dir.rglob(f"{args.map}.map")))
        scen = sorted(args.bench_dir.rglob(f"{args.map}-random-*.scen"))[args.scen_idx]
        picked = dedup_agents(parse_scen(scen), args.agents)
        n = min(args.agents, len(picked))
        starts_d = {i: picked[i][0] for i in range(n)}
        goals_d = {i: picked[i][1] for i in range(n)}
        out = Path(args.out) if args.out else IMG_DIR / "fig_case_study.png"
        plot_case(grid, starts_d, goals_d, args.ckpts, out, args.max_steps)


if __name__ == "__main__":
    main()
