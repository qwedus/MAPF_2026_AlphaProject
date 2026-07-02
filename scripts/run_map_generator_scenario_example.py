"""Run CBS on one map-generator scenario JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig, save_internal_paths_json
from src.scenario_loader import load_map_generator_scenario


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CBS on a map-generator scenario JSON.",
    )
    parser.add_argument(
        "scenario_json",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "scenarios/scenario_ex.json",
        help="Path to a map-generator scenario JSON. Defaults to scenarios/scenario_ex.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/paths",
        help="Directory where CBS artifacts and internal paths JSON are written.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="CBS timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenario_path = args.scenario_json.resolve()
    scenario_id = _load_scenario_id(scenario_path)

    grid_map, starts, goals = load_map_generator_scenario(scenario_path)
    work_dir = args.output_dir.resolve() / scenario_id
    adapter = CBSAdapter(
        CBSAdapterConfig(
            work_dir=work_dir,
            timeout_sec=args.timeout_sec,
            python_executable=sys.executable,
        )
    )
    paths = adapter.plan(starts=starts, goals=goals, grid=grid_map)

    internal_paths_json_path = args.output_dir.resolve() / f"{scenario_id}_internal_paths.json"
    save_internal_paths_json(paths, internal_paths_json_path)

    print(f"scenario JSON: {scenario_path}")
    print(f"scenario_id: {scenario_id}")
    print(f"grid_map shape: {grid_map.shape}")
    print(f"starts row_col: {starts}")
    print(f"goals row_col: {goals}")
    if adapter.last_run is not None:
        print(f"input.yaml: {adapter.last_run.input_yaml_path}")
        print(f"output.yaml: {adapter.last_run.output_yaml_path}")
    print(f"internal paths: {paths}")
    print(f"internal paths JSON: {internal_paths_json_path}")
    print("collision validation: deferred to simulator module")
    return 0


def _load_scenario_id(scenario_path: Path) -> str:
    with scenario_path.open("r", encoding="utf-8") as handle:
        scenario = json.load(handle)
    if isinstance(scenario, dict) and scenario.get("scenario_id"):
        return str(scenario["scenario_id"])
    return scenario_path.stem


if __name__ == "__main__":
    raise SystemExit(main())
