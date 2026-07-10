"""Head-to-head CBS vs (all) IL models on the same MovingAI instances.

Motivates the hybrid thesis by solving each instance BOTH ways and recording
three axes, swept over map density and #agents.

  CBS  : run the solver once with a time budget.
         -> solved? / wall-time / makespan (optimal)   [exact, but slow / times out when hard]
  IL_k : roll out each trained policy in MAPFStepSimulator.
         -> success? / wall-time / makespan (steps)     [fast, scales, degrades gracefully]

All IL checkpoints (MLP / CNN variants) are evaluated on the *same* instance so
CBS and every model are directly comparable. No dataset creation, no training.

Usage:
  py scripts/eval_cbs_vs_il.py --bench-dir <scratch>/mapf_bench \\
     --configs empty-8-8:2,4,8,16 random-32-32-10:2,4,8 random-32-32-20:2,4,8 \\
               room-32-32-4:2,4,8 \\
     --ckpts mlp.pt cnn.pt cnn_diverse.pt cnn_dagger.pt \\
     --instances 10 --cbs-timeout 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_cnn import ActionCNN
from model_mlp import ActionMLP
from simulator import MAPFStepSimulator
from src.cbs_adapter import CBSAdapter, CBSAdapterConfig
from scripts.eval_movingai_benchmark import parse_map, parse_scen, dedup_agents

LABELS = {  # ckpt filename -> short label
    "mlp.pt": "MLP-BC", "mlp_final.pt": "MLP(47k)", "cnn.pt": "CNN(1.6k)",
    "cnn_diverse.pt": "CNN(13.7k)", "cnn_big.pt": "CNN(27.7k)",
    "cnn_bigdense.pt": "CNN(47k)", "cnn_dagger.pt": "CNN-DAgger",
    "cnn_dagger_ext.pt": "CNN-DAgger-ext", "cnn_dagger_final.pt": "DAgger(47k)",
}


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    mode = ck.get("mode", "cnn")
    model = (ActionMLP() if mode == "mlp" else ActionCNN())
    model.load_state_dict(ck["model_state"]); model.eval()
    return {"name": LABELS.get(Path(path).name, Path(path).name), "mode": mode,
            "model": model, "gmean": ck["goal_mean"].cpu(), "gstd": ck["goal_std"].cpu()}


@torch.no_grad()
def predict(m, obs):
    acts = {}
    for aid, st in obs.items():
        g = torch.from_numpy(st["grid"]).float().unsqueeze(0)
        d = (torch.from_numpy(st["goal_dir"]).float().unsqueeze(0) - m["gmean"]) / m["gstd"]
        if m["mode"] == "mlp":
            logits = m["model"](torch.cat([g.reshape(1, -1), d], dim=1))
        else:
            logits = m["model"]((g, d))
        acts[aid] = int(logits.argmax(1))
    return acts


def run_cbs(grid, starts, goals, timeout):
    adapter = CBSAdapter(CBSAdapterConfig(solver_root=None, timeout_sec=timeout))
    t0 = time.time()
    try:
        paths = adapter.plan(starts, goals, grid)
        return True, time.time() - t0, max((len(p) for p in paths.values()), default=1) - 1
    except Exception:
        return False, time.time() - t0, None


def run_il(m, grid, starts_d, goals_d, max_steps):
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts_d, goals_d)
    t0 = time.time()
    success, steps = False, max_steps
    for _ in range(max_steps):
        obs, done, info = sim.step(predict(m, obs))
        if done:
            success = bool(info["all_at_goal"]); steps = info["t"]; break
    return success, time.time() - t0, (steps if success else None)


class Agg:
    def __init__(self): self.n = self.ok = 0; self.t = 0.0; self.mk = 0.0; self.mkn = 0
    def add(self, ok, t, mk):
        self.n += 1; self.ok += int(ok); self.t += t
        if ok and mk is not None: self.mk += mk; self.mkn += 1
    def row(self):
        return (self.ok / self.n if self.n else float("nan"),
                self.t / self.n if self.n else float("nan"),
                self.mk / self.mkn if self.mkn else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="+", default=["cnn_diverse.pt"])
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--cbs-timeout", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=160)
    args = ap.parse_args()

    models = [load_model(PROJECT_ROOT / c) for c in args.ckpts]
    print(f"models: CBS(timeout {args.cbs_timeout}s) + " + ", ".join(m["name"] for m in models))
    print("(succ=success rate, t=mean wall-time sec, mk=mean makespan on successes)\n")

    rows = []
    for cfg in args.configs:
        mapname, agent_str = cfg.split(":")
        grid = parse_map(next(args.bench_dir.rglob(f"{mapname}.map")))
        dens = float(grid.mean())
        scen_paths = sorted(args.bench_dir.rglob(f"{mapname}-random-*.scen"))[: args.instances]
        for n in [int(x) for x in agent_str.split(",")]:
            cbs = Agg(); ils = {m["name"]: Agg() for m in models}
            for sp in scen_paths:
                picked = dedup_agents(parse_scen(sp), n)
                if len(picked) < n:
                    continue
                starts = [picked[i][0] for i in range(n)]
                goals = [picked[i][1] for i in range(n)]
                sd = {i: picked[i][0] for i in range(n)}
                gd = {i: picked[i][1] for i in range(n)}
                cbs.add(*run_cbs(grid, starts, goals, args.cbs_timeout))
                for m in models:
                    ils[m["name"]].add(*run_il(m, grid, sd, gd, args.max_steps))
            if not cbs.n:
                continue
            print(f"### {mapname}  (density {dens*100:.0f}%)  agents={n}  inst={cbs.n}")
            sr, t, mk = cbs.row()
            print(f"    {'CBS':16} succ={sr:.2f}  t={t:7.2f}s  mk={mk:6.1f}")
            rec = {"map": mapname, "density": dens, "n": n, "inst": cbs.n,
                   "CBS": {"sr": sr, "t": t, "mk": mk}}
            for m in models:
                sr, t, mk = ils[m["name"]].row()
                print(f"    {m['name']:16} succ={sr:.2f}  t={t:7.3f}s  mk={mk:6.1f}")
                rec[m["name"]] = {"sr": sr, "t": t, "mk": mk}
            rows.append(rec); print()

    out = PROJECT_ROOT / "outputs" / "cbs_vs_il.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out}")
    return rows


if __name__ == "__main__":
    main()
