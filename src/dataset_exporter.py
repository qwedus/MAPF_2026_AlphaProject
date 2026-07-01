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
    "up": 0,
    "down": 1,
    "left": 2,
    "right": 3,
    "wait": 4,
}
ID_TO_ACTION = {action_id: action for action, action_id in ACTION_TO_ID.items()}

LOCAL_GRID_CHANNELS = 3
LOCAL_GRID_SIZE = 5


def extract_state_action_pairs(
    grid_map: np.ndarray,
    goals: Mapping[int, GridLocation] | Sequence[GridLocation],
    paths: Mapping[int, Sequence[GridLocation]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``states_grid``, ``goal_dirs``, and ``actions`` from expert paths.

    Output shapes:
    - ``states_grid``: ``(num_samples, 3, 5, 5)``
      - channel 0: free space
      - channel 1: wall, obstacle, or out-of-bounds
      - channel 2: other agents
    - ``goal_dirs``: ``(num_samples, 2)``
      - raw direction from current cell to goal as ``[goal_row - row, goal_col - col]``
    - ``actions``: ``(num_samples,)``
      - 0 up, 1 down, 2 left, 3 right, 4 wait
    """
    grid = _normalize_grid_map(grid_map)
    height, width = grid.shape
    normalized_goals = _normalize_goals(goals)
    normalized_paths = {
        int(agent_id): _normalize_path(path, int(agent_id), height, width)
        for agent_id, path in paths.items()
    }

    states_grid: list[np.ndarray] = []
    goal_dirs: list[list[int]] = []
    actions: list[int] = []

    for agent_id, normalized_path in sorted(normalized_paths.items()):
        if agent_id not in normalized_goals:
            raise ValueError(f"Missing goal for agent {agent_id}.")
        if len(normalized_path) < 2:
            continue

        goal = normalized_goals[agent_id]
        _validate_location(goal, height, width, f"goal for agent {agent_id}")

        for timestep, (current, next_location) in enumerate(
            zip(normalized_path, normalized_path[1:])
        ):
            other_agent_positions = [
                _location_at_time(
                    other_path,
                    timestep,
                )
                for other_agent_id, other_path in sorted(normalized_paths.items())
                if other_agent_id != agent_id
            ]
            state = _extract_local_grid(
                grid=grid,
                current=current,
                other_agent_positions=other_agent_positions,
            )

            states_grid.append(state)
            goal_dirs.append(_goal_direction(current, goal))
            actions.append(_action_id(current, next_location))

    if not states_grid:
        return (
            np.empty(
                (0, LOCAL_GRID_CHANNELS, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
                dtype=np.float32,
            ),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(states_grid).astype(np.float32),
        np.asarray(goal_dirs, dtype=np.float32),
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


def _location_at_time(path: Sequence[GridLocation], timestep: int) -> GridLocation:
    if timestep < len(path):
        return tuple(path[timestep])
    return tuple(path[-1])


def _extract_local_grid(
    grid: np.ndarray,
    current: GridLocation,
    other_agent_positions: Sequence[GridLocation],
) -> np.ndarray:
    height, width = grid.shape
    current_row, current_col = current
    half_window = LOCAL_GRID_SIZE // 2
    other_positions = {tuple(location) for location in other_agent_positions}
    state = np.zeros(
        (LOCAL_GRID_CHANNELS, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE),
        dtype=np.float32,
    )

    for local_row in range(LOCAL_GRID_SIZE):
        for local_col in range(LOCAL_GRID_SIZE):
            row = current_row - half_window + local_row
            col = current_col - half_window + local_col
            if not (0 <= row < height and 0 <= col < width):
                state[1, local_row, local_col] = 1.0
            elif (row, col) in other_positions:
                state[2, local_row, local_col] = 1.0
            elif grid[row, col] == 1:
                state[1, local_row, local_col] = 1.0
            else:
                state[0, local_row, local_col] = 1.0

    return state


def _goal_direction(current: GridLocation, goal: GridLocation) -> list[int]:
    row_delta = goal[0] - current[0]
    col_delta = goal[1] - current[1]
    return [row_delta, col_delta]


def _action_id(current: GridLocation, next_location: GridLocation) -> int:
    row_delta = next_location[0] - current[0]
    col_delta = next_location[1] - current[1]
    delta_to_action = {
        (-1, 0): ACTION_TO_ID["up"],
        (1, 0): ACTION_TO_ID["down"],
        (0, -1): ACTION_TO_ID["left"],
        (0, 1): ACTION_TO_ID["right"],
        (0, 0): ACTION_TO_ID["wait"],
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
