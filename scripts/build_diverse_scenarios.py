"""Generate a larger, more diverse scenario set for IL data.

The default ``map_generator.build_dataset`` makes exactly ONE scenario per
(map_type, size, agent_count) combo. That gives 45 scenarios and ~1.6k samples,
and the left/right confusion in the trained policy suggests the model just wants
more varied layouts.

This script keeps the same generator but draws MANY seeds per combo, so each
combo contributes several different wall layouts / start-goal placements. Combo
choice is curated from the CBS failure log: maze and large-crowd (n5) combos time
out in CBS, so they get few (or no) seeds, while the CBS-friendly combos
(empty / sparse / rooms, few agents) get the bulk of the budget.

Scenario ids are ``scen_<type>_s<size>_n<agents>_v<seed_idx>`` so they never
collide with the existing ``scenarios_full`` ids.

Run CBS on the result with:
    py scripts/run_cbs_batch.py --scenarios-dir scenarios_diverse \\
        --output-dir outputs/paths/diverse --timeout-sec 45
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map_generator import generate_map, save_grid, save_scenario, to_scenario

# (map_type, sizes, agent_counts, seeds_per_combo)
# Budget is skewed toward CBS-solvable combos; maze/large-n5 kept minimal
# because they time out in CBS regardless of how many seeds we draw.
PLAN: list[tuple[str, list[int], list[int], int]] = [
    ("empty",  [8, 11, 15], [2, 3, 5], 10),
    ("sparse", [8, 11, 15], [2, 3, 5], 10),
    ("dense",  [8, 11, 15], [2, 3],    10),
    ("dense",  [8, 11],     [5],        5),   # small dense crowds only
    ("rooms",  [8, 11, 15], [2, 3],    10),
    ("rooms",  [8, 11],     [5],        5),
    ("maze",   [8, 11],     [2],        6),   # maze is CBS-expensive: easy configs only
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "scenarios_diverse")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Seed offset; kept away from scenarios_full's 0-based seeds.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.base_seed
    made = 0
    build_failed: list[str] = []
    by_type: dict[str, int] = {}

    for map_type, sizes, agent_counts, n_seeds in PLAN:
        for size in sizes:
            for n in agent_counts:
                for v in range(n_seeds):
                    tag = f"{map_type}_s{size}_n{n}_v{v}"
                    try:
                        grid, s_rc, g_rc = generate_map(
                            map_type, size=size, num_agents=n, seed=seed
                        )
                    except RuntimeError:
                        build_failed.append(tag)  # not enough free space; skip
                        seed += 1
                        continue

                    map_file = f"{out_dir.name}/map_{tag}.npy"
                    save_grid(grid, out_dir / f"map_{tag}.npy")
                    scen = to_scenario(
                        grid, s_rc, g_rc,
                        scenario_id=f"scen_{tag}",
                        map_id=tag,
                        map_file=map_file,
                    )
                    save_scenario(scen, out_dir / f"scenario_{tag}.json")
                    made += 1
                    by_type[map_type] = by_type.get(map_type, 0) + 1
                    seed += 1

    print(f"생성 완료: {made}개 시나리오 -> {out_dir}")
    for t, c in sorted(by_type.items()):
        print(f"  {t:7s}: {c}")
    if build_failed:
        print(f"⚠ 공간 부족으로 생성 스킵 {len(build_failed)}개: {build_failed[:8]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
