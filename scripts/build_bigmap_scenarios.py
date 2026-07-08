"""Generate scenarios that fill the two distribution gaps hurting large-map IL.

Diagnosis (see work log 2026-07): the policy only sees a 5x5 FOV, so its only
global cue is goal_dir. Training maps were <=15x15, so goal_dir components never
exceeded ~14 (92% <=7), while 32x32 benchmarks need up to 31 -> the goal input is
far out-of-distribution. Also training capped agents at 5, so the FOV rarely got
crowded like the high-congestion test regime.

So we add:
  1) LARGE maps, FEW agents  -> large goal_dir values (CBS solves these fast)
  2) MEDIUM maps, MORE agents -> crowded FOV (other-robot channel density)

Only empty/sparse (fast CBS). Scenario ids are '..._v<k>' so they never collide
with existing sets. Run CBS with:
  py scripts/run_cbs_batch.py --scenarios-dir scenarios_bigmap \\
     --output-dir outputs/paths/bigmap --timeout-sec 30
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
    # gap 1: large maps, few agents -> extend goal_dir range
    ("empty",  [16, 20, 24, 32], [2, 3], 8),
    ("sparse", [16, 20, 24, 32], [2, 3], 8),
    # gap 2: medium maps, more agents -> crowded FOV
    ("empty",  [11, 15], [5, 6, 7], 8),
    ("sparse", [11, 15], [5, 6, 7], 6),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "scenarios_bigmap")
    ap.add_argument("--base-seed", type=int, default=5000)
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
                                       scenario_id=f"scen_big_{tag}", map_id=tag,
                                       map_file=f"{out_dir.name}/map_{tag}.npy")
                    save_scenario(scen, out_dir / f"scenario_{tag}.json")
                    made += 1
                    by_type[f"{mtype}"] = by_type.get(mtype, 0) + 1
                    seed += 1

    print(f"생성 완료: {made}개 -> {out_dir}")
    for t, c in sorted(by_type.items()):
        print(f"  {t:7s}: {c}")
    if failed:
        print(f"⚠ 스킵 {len(failed)}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
