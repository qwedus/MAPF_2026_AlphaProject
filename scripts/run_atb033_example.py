"""Run a fixed example through the subprocess-based atb033 CBS adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig, GridLocation


def find_vertex_collisions(
    paths: dict[int, list[GridLocation]],
) -> list[tuple[int, tuple[int, ...], GridLocation]]:
    collisions: list[tuple[int, tuple[int, ...], GridLocation]] = []
    if not paths:
        return collisions

    makespan = max(len(path) for path in paths.values())
    for t in range(makespan):
        occupied: dict[GridLocation, list[int]] = {}
        for agent_id, path in paths.items():
            location = path[min(t, len(path) - 1)]
            occupied.setdefault(location, []).append(agent_id)
        for location, agent_ids in occupied.items():
            if len(agent_ids) > 1:
                collisions.append((t, tuple(agent_ids), location))
    return collisions


def find_edge_collisions(
    paths: dict[int, list[GridLocation]],
) -> list[tuple[int, int, int, GridLocation, GridLocation]]:
    collisions: list[tuple[int, int, int, GridLocation, GridLocation]] = []
    if not paths:
        return collisions

    agent_ids = sorted(paths)
    makespan = max(len(path) for path in paths.values())
    for t in range(makespan - 1):
        for index, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[index + 1 :]:
                a_from = paths[agent_a][min(t, len(paths[agent_a]) - 1)]
                a_to = paths[agent_a][min(t + 1, len(paths[agent_a]) - 1)]
                b_from = paths[agent_b][min(t, len(paths[agent_b]) - 1)]
                b_to = paths[agent_b][min(t + 1, len(paths[agent_b]) - 1)]
                if a_from == b_to and a_to == b_from:
                    collisions.append((t, agent_a, agent_b, a_from, a_to))
    return collisions


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

    vertex_collisions = find_vertex_collisions(paths)
    edge_collisions = find_edge_collisions(paths)
    makespan = max(len(path) for path in paths.values()) if paths else 0

    print("grid_map:")
    print(grid_map)
    print(f"starts: {starts}")
    print(f"goals: {goals}")
    print(f"input.yaml: {adapter.last_run.input_yaml_path}")
    print(f"output.yaml: {adapter.last_run.output_yaml_path}")
    print(f"internal paths: {paths}")
    print(f"path lengths: { {agent_id: len(path) for agent_id, path in paths.items()} }")
    print(f"makespan: {makespan}")
    print(f"vertex collisions: {vertex_collisions}")
    print(f"edge collisions: {edge_collisions}")

    if vertex_collisions or edge_collisions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
