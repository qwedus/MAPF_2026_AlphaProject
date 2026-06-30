"""Export CBS expert paths as supervised state-action samples.

This module intentionally does not validate path collisions. Collision checks
belong to the simulator module and should run before final train/val/test
datasets are generated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

GridLocation = tuple[int, int]
AgentPath = list[GridLocation]

ACTION_TO_ID = {
    "wait": 0,
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
}
ID_TO_ACTION = {action_id: action for action, action_id in ACTION_TO_ID.items()}


def extract_state_action_pairs(
    grid_map: np.ndarray,
    goals: Mapping[int, GridLocation] | Sequence[GridLocation],
    paths: Mapping[int, Sequence[GridLocation]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``states_grid``, ``goal_dirs``, and ``actions`` from expert paths.

    Output shapes:
    - ``states_grid``: ``(num_samples, 2, height, width)``
      - channel 0: obstacle map, where 1 means blocked
      - channel 1: current agent position, one-hot
    - ``goal_dirs``: ``(num_samples, 2)``
      - clipped direction from current cell to goal as ``[d_row, d_col]``
    - ``actions``: ``(num_samples,)``
      - 0 wait, 1 up, 2 down, 3 left, 4 right
    """
    grid = _normalize_grid_map(grid_map)
    height, width = grid.shape
    normalized_goals = _normalize_goals(goals)

    states_grid: list[np.ndarray] = []
    goal_dirs: list[list[int]] = []
    actions: list[int] = []

    for agent_id, path in sorted(paths.items()):
        if agent_id not in normalized_goals:
            raise ValueError(f"Missing goal for agent {agent_id}.")
        normalized_path = _normalize_path(path, agent_id, height, width)
        if len(normalized_path) < 2:
            continue

        goal = normalized_goals[agent_id]
        _validate_location(goal, height, width, f"goal for agent {agent_id}")

        for current, next_location in zip(normalized_path, normalized_path[1:]):
            state = np.zeros((2, height, width), dtype=np.float32)
            state[0] = grid.astype(np.float32)
            state[1, current[0], current[1]] = 1.0

            states_grid.append(state)
            goal_dirs.append(_goal_direction(current, goal))
            actions.append(_action_id(current, next_location))

    if not states_grid:
        return (
            np.empty((0, 2, height, width), dtype=np.float32),
            np.empty((0, 2), dtype=np.int8),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(states_grid).astype(np.float32),
        np.asarray(goal_dirs, dtype=np.int8),
        np.asarray(actions, dtype=np.int64),
    )


def save_dataset_npz(
    save_path: str | Path,
    states_grid: np.ndarray,
    goal_dirs: np.ndarray,
    actions: np.ndarray,
) -> Path:
    """Save extracted arrays to ``.npz``."""
    path = Path(save_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states_grid=states_grid,
        goal_dirs=goal_dirs,
        actions=actions,
    )
    return path


def export_expert_paths_npz(
    grid_map: np.ndarray,
    goals: Mapping[int, GridLocation] | Sequence[GridLocation],
    paths: Mapping[int, Sequence[GridLocation]],
    save_path: str | Path,
) -> Path:
    """Extract state-action pairs from paths and save them as ``.npz``."""
    states_grid, goal_dirs, actions = extract_state_action_pairs(
        grid_map=grid_map,
        goals=goals,
        paths=paths,
    )
    return save_dataset_npz(
        save_path=save_path,
        states_grid=states_grid,
        goal_dirs=goal_dirs,
        actions=actions,
    )


def create_dummy_export(save_path: str | Path) -> Path:
    """Create a small dummy export for pre-simulator smoke testing."""
    grid_map = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    goals = {
        0: (4, 0),
        1: (0, 4),
    }
    paths = {
        0: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        1: [(4, 4), (3, 4), (2, 4), (1, 4), (0, 4)],
    }
    return export_expert_paths_npz(grid_map, goals, paths, save_path)


def _normalize_grid_map(grid_map: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid_map)
    if grid.ndim != 2:
        raise ValueError(f"grid_map must be 2-D, got shape {grid.shape}.")
    return (grid == 1).astype(np.uint8)


def _normalize_goals(
    goals: Mapping[int, GridLocation] | Sequence[GridLocation],
) -> dict[int, GridLocation]:
    if isinstance(goals, Mapping):
        return {int(agent_id): tuple(location) for agent_id, location in goals.items()}
    return {agent_id: tuple(location) for agent_id, location in enumerate(goals)}


def _normalize_path(
    path: Sequence[GridLocation],
    agent_id: int,
    height: int,
    width: int,
) -> list[GridLocation]:
    normalized_path = [tuple(location) for location in path]
    for step_index, location in enumerate(normalized_path):
        _validate_location(
            location,
            height,
            width,
            f"path step {step_index} for agent {agent_id}",
        )
    return normalized_path


def _validate_location(
    location: GridLocation,
    height: int,
    width: int,
    label: str,
) -> None:
    row, col = location
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(f"{label} {location} is outside grid shape {(height, width)}.")


def _goal_direction(current: GridLocation, goal: GridLocation) -> list[int]:
    row_delta = goal[0] - current[0]
    col_delta = goal[1] - current[1]
    return [_clip_direction(row_delta), _clip_direction(col_delta)]


def _clip_direction(delta: int) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def _action_id(current: GridLocation, next_location: GridLocation) -> int:
    row_delta = next_location[0] - current[0]
    col_delta = next_location[1] - current[1]
    delta_to_action = {
        (0, 0): ACTION_TO_ID["wait"],
        (-1, 0): ACTION_TO_ID["up"],
        (1, 0): ACTION_TO_ID["down"],
        (0, -1): ACTION_TO_ID["left"],
        (0, 1): ACTION_TO_ID["right"],
    }
    try:
        return delta_to_action[(row_delta, col_delta)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported move from {current} to {next_location}. "
            "Only wait and 4-neighbor moves are supported."
        ) from exc


if __name__ == "__main__":
    output_path = create_dummy_export("outputs/datasets/dummy_expert_paths.npz")
    print(f"dummy dataset saved: {output_path}")

