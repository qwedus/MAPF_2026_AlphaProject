"""CBS expert handoff artifacts for the IL pipeline.

The handoff JSON is the boundary between Track 1 CBS planning and Track 2 IL
dataset extraction. It stores enough context to reconstruct
``extract_dataset(map_grid, agent_paths, agent_goals)`` without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.dataset_exporter import export_expert_paths_npz

GridLocation = tuple[int, int]
AgentPath = list[GridLocation]

SCHEMA_VERSION = "cbs_il_handoff_v0.2"


def build_expert_handoff_payload(
    scenario_id: str,
    source_path: str | Path,
    grid_map: Any,
    starts: Sequence[GridLocation],
    goals: Sequence[GridLocation],
    paths: Mapping[int, Sequence[GridLocation]],
    map_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable CBS expert handoff payload."""
    grid = _normalize_grid_map(grid_map)
    normalized_paths = {
        int(agent_id): [tuple(location) for location in path]
        for agent_id, path in paths.items()
    }
    normalized_starts = [tuple(location) for location in starts]
    normalized_goals = [tuple(location) for location in goals]
    validation = validate_expert_paths(grid, normalized_paths)

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": str(scenario_id),
        "source_path": str(Path(source_path).resolve()),
        "coordinate_format": "row_col",
        "map": {
            "grid_shape": [int(grid.shape[0]), int(grid.shape[1])],
            "map_file": str(map_file) if map_file is not None else None,
            "grid": grid.astype(int).tolist() if map_file is None else None,
        },
        "agents": {
            str(agent_id): {
                "start": _location_to_json(normalized_starts[agent_id]),
                "goal": _location_to_json(normalized_goals[agent_id]),
                "path": [_location_to_json(location) for location in path],
            }
            for agent_id, path in sorted(normalized_paths.items())
        },
        "validation": validation,
    }


def save_expert_handoff_json(payload: Mapping[str, Any], save_path: str | Path) -> Path:
    """Write an expert handoff payload to JSON."""
    path = Path(save_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def load_expert_handoff_json(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an expert handoff JSON file."""
    handoff_path = Path(path).resolve()
    with handoff_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported handoff schema_version: {payload.get('schema_version')!r}"
        )
    if payload.get("coordinate_format") != "row_col":
        raise ValueError("Expert handoff coordinate_format must be 'row_col'.")
    return payload


def handoff_to_training_inputs(
    handoff: Mapping[str, Any],
    base_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[int, GridLocation], dict[int, AgentPath]]:
    """Convert a handoff payload into ``map_grid, agent_goals, agent_paths``."""
    map_grid = _load_handoff_grid(handoff, base_dir)
    agents = handoff.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise ValueError("Expert handoff must include non-empty agents.")

    goals: dict[int, GridLocation] = {}
    paths: dict[int, AgentPath] = {}
    for agent_id_text, agent in agents.items():
        agent_id = int(agent_id_text)
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {agent_id} payload must be an object.")
        goals[agent_id] = _json_to_location(agent["goal"], f"agent {agent_id} goal")
        paths[agent_id] = [
            _json_to_location(location, f"agent {agent_id} path")
            for location in agent["path"]
        ]

    return map_grid, goals, paths


def export_handoff_npz(
    handoff_json_path: str | Path,
    save_path: str | Path,
) -> Path:
    """Convert an expert handoff JSON into an IL v0.2 ``.npz`` dataset."""
    handoff_path = Path(handoff_json_path).resolve()
    handoff = load_expert_handoff_json(handoff_path)
    map_grid, goals, paths = handoff_to_training_inputs(
        handoff,
        base_dir=handoff_path.parent,
    )
    return export_expert_paths_npz(
        grid_map=map_grid,
        goals=goals,
        paths=paths,
        save_path=save_path,
    )


def validate_expert_paths(
    grid_map: Any,
    paths: Mapping[int, Sequence[GridLocation]],
) -> dict[str, Any]:
    """Validate path bounds, blocked cells, vertex collisions, and edge collisions."""
    grid = _normalize_grid_map(grid_map)
    normalized_paths = {
        int(agent_id): [tuple(location) for location in path]
        for agent_id, path in paths.items()
    }
    violations = _detect_blocked_or_outside(grid, normalized_paths)
    vertex_collisions = _detect_vertex_collisions(normalized_paths)
    edge_collisions = _detect_edge_collisions(normalized_paths)
    path_lengths = {
        str(agent_id): len(path) for agent_id, path in sorted(normalized_paths.items())
    }

    return {
        "success": not violations and not vertex_collisions and not edge_collisions,
        "no_blocked_cells": not violations,
        "no_vertex_collisions": not vertex_collisions,
        "no_edge_collisions": not edge_collisions,
        "blocked_or_outside": violations,
        "vertex_collisions": vertex_collisions,
        "edge_collisions": edge_collisions,
        "path_lengths": path_lengths,
        "makespan": max(path_lengths.values(), default=0),
    }


def _normalize_grid_map(grid_map: Any) -> np.ndarray:
    grid = np.asarray(grid_map)
    if grid.ndim != 2:
        raise ValueError(f"grid_map must be 2-D, got shape {grid.shape}.")
    return (grid == 1).astype(np.uint8)


def _load_handoff_grid(
    handoff: Mapping[str, Any],
    base_dir: str | Path | None,
) -> np.ndarray:
    map_payload = handoff.get("map")
    if not isinstance(map_payload, dict):
        raise ValueError("Expert handoff must include map object.")

    map_file = map_payload.get("map_file")
    if map_file:
        map_path = Path(map_file)
        candidates = [map_path] if map_path.is_absolute() else [
            Path.cwd() / map_path,
            Path(base_dir or ".") / map_path,
        ]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return _normalize_grid_map(np.load(resolved))
        raise FileNotFoundError(
            "Could not find handoff map_file. Tried: "
            + ", ".join(str(candidate.resolve()) for candidate in candidates)
        )

    grid = map_payload.get("grid")
    if grid is None:
        raise ValueError("Expert handoff map must include map_file or grid.")
    return _normalize_grid_map(grid)


def _detect_blocked_or_outside(
    grid: np.ndarray,
    paths: Mapping[int, Sequence[GridLocation]],
) -> list[dict[str, Any]]:
    height, width = grid.shape
    violations: list[dict[str, Any]] = []
    for agent_id, path in sorted(paths.items()):
        for timestep, (row, col) in enumerate(path):
            reason: str | None = None
            if not (0 <= row < height and 0 <= col < width):
                reason = "outside"
            elif grid[row, col] == 1:
                reason = "blocked"
            if reason is not None:
                violations.append(
                    {
                        "agent_id": int(agent_id),
                        "timestep": int(timestep),
                        "location": [int(row), int(col)],
                        "reason": reason,
                    }
                )
    return violations


def _detect_vertex_collisions(
    paths: Mapping[int, Sequence[GridLocation]],
) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    makespan = max((len(path) for path in paths.values()), default=0)
    for timestep in range(makespan):
        seen: dict[GridLocation, int] = {}
        for agent_id, path in sorted(paths.items()):
            location = _location_at_time(path, timestep)
            if location in seen:
                collisions.append(
                    {
                        "timestep": int(timestep),
                        "agent_ids": [int(seen[location]), int(agent_id)],
                        "location": _location_to_json(location),
                    }
                )
            seen[location] = int(agent_id)
    return collisions


def _detect_edge_collisions(
    paths: Mapping[int, Sequence[GridLocation]],
) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    makespan = max((len(path) for path in paths.values()), default=0)
    agent_ids = sorted(paths)
    for timestep in range(max(0, makespan - 1)):
        for left_index, agent_id_a in enumerate(agent_ids):
            for agent_id_b in agent_ids[left_index + 1 :]:
                a_from = _location_at_time(paths[agent_id_a], timestep)
                a_to = _location_at_time(paths[agent_id_a], timestep + 1)
                b_from = _location_at_time(paths[agent_id_b], timestep)
                b_to = _location_at_time(paths[agent_id_b], timestep + 1)
                if a_from == b_to and a_to == b_from:
                    collisions.append(
                        {
                            "timestep": int(timestep),
                            "agent_ids": [int(agent_id_a), int(agent_id_b)],
                            "edge": [
                                _location_to_json(a_from),
                                _location_to_json(a_to),
                            ],
                        }
                    )
    return collisions


def _location_at_time(path: Sequence[GridLocation], timestep: int) -> GridLocation:
    if timestep < len(path):
        return tuple(path[timestep])
    return tuple(path[-1])


def _location_to_json(location: GridLocation) -> list[int]:
    row, col = location
    return [int(row), int(col)]


def _json_to_location(value: Any, label: str) -> GridLocation:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be [row, col].")
    return (int(value[0]), int(value[1]))
