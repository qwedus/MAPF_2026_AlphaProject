"""Generate high-agent-density scenarios to fix the congestion/deadlock regime.

Diagnosis (work log 2026-07): large-map failures came from goal_dir OOD (fixed in
2-14), but high-congestion failures (n8+) are a *crowded FOV / deadlock* problem
— the model rarely saw many other-robots in its 5x5 window during training
(agents capped at 5). So here we add many-agent scenarios on small/medium maps
where CBS still solves fast, to populate the crowded-FOV distribution.

empty/sparse only (CBS-cheap even with 6-8 agents on <=15 maps).

Run CBS with:
  py scripts/run_cbs_batch.py --scenarios-dir scenarios_dense \\
     --output-dir outputs/paths/dense --timeout-sec 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map_generator import generate_map, save_grid, save_scenario, to_scenario

# (map_type, sizes, agent_counts, seeds_per_combo)
PLAN = [
    ("empty",  [8, 11, 15], [6, 7, 8], 12),
    ("sparse", [8, 11, 15], [6, 7, 8], 12),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "scenarios_dense")
    ap.add_argument("--base-seed", type=int, default=9000)
    args = ap.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.base_seed
    made = 0
    by_type: dict[str, int] = {}
    failed = []
    for mtype, sizes, agents, nseeds in PLAN:
        for size in sizes:
            for n in agents:
                for v in range(nseeds):
                    tag = f"{mtype}_s{size}_n{n}_v{v}"
                    try:
                        grid, s_rc, g_rc = generate_map(mtype, size=size, num_agents=n, seed=seed)
                    except RuntimeError:
                        failed.append(tag); seed += 1; continue
                    save_grid(grid, out_dir / f"map_{tag}.npy")
                    scen = to_scenario(grid, s_rc, g_rc,
                                       scenario_id=f"scen_dns_{tag}", map_id=tag,
                                       map_file=f"{out_dir.name}/map_{tag}.npy")
                    save_scenario(scen, out_dir / f"scenario_{tag}.json")
                    made += 1
                    by_type[mtype] = by_type.get(mtype, 0) + 1
                    seed += 1

    print(f"생성 완료: {made}개 -> {out_dir}")
    for t, c in sorted(by_type.items()):
        print(f"  {t:7s}: {c}")
    if failed:
        print(f"⚠ 스킵 {len(failed)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
