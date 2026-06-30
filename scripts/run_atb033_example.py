"""Run a fixed example through the subprocess-based atb033 CBS adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig, save_internal_paths_json


def main() -> int:
    grid_map = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    starts = [(0, 0), (4, 4)]
    goals = [(4, 0), (0, 4)]

    work_dir = PROJECT_ROOT / "outputs/paths/atb033_example"
    adapter = CBSAdapter(
        CBSAdapterConfig(
            work_dir=work_dir,
            python_executable=sys.executable,
        )
    )
    paths = adapter.plan(starts=starts, goals=goals, grid=grid_map)
    if adapter.last_run is None:
        raise RuntimeError("CBS adapter did not record run artifacts.")

    makespan = max(len(path) for path in paths.values()) if paths else 0
    internal_paths_json_path = PROJECT_ROOT / "outputs/paths/atb033_example_internal_paths.json"
    save_internal_paths_json(paths, internal_paths_json_path)

    print("grid_map:")
    print(grid_map)
    print(f"starts: {starts}")
    print(f"goals: {goals}")
    print(f"input.yaml: {adapter.last_run.input_yaml_path}")
    print(f"output.yaml: {adapter.last_run.output_yaml_path}")
    print(f"internal paths: {paths}")
    print(f"internal paths JSON: {internal_paths_json_path}")
    print(f"path lengths: { {agent_id: len(path) for agent_id, path in paths.items()} }")
    print(f"makespan: {makespan}")
    print("collision validation: deferred to simulator module")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
