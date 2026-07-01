import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset_exporter import (
    ACTION_TO_ID,
    export_expert_paths_npz,
    extract_state_action_pairs,
)


class DatasetExporterTests(unittest.TestCase):
    def test_extract_state_action_pairs_from_paths(self) -> None:
        grid_map = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        goals = {0: (1, 3), 1: (2, 4)}
        paths = {
            0: [(2, 2), (2, 3), (1, 3)],
            1: [(2, 4), (2, 4), (2, 4)],
        }

        states_grid, goal_dirs, actions = extract_state_action_pairs(
            grid_map,
            goals,
            paths,
        )

        self.assertEqual(states_grid.shape, (4, 3, 5, 5))
        self.assertEqual(
            goal_dirs.tolist(),
            [[-1.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        self.assertEqual(
            actions.tolist(),
            [
                ACTION_TO_ID["right"],
                ACTION_TO_ID["up"],
                ACTION_TO_ID["wait"],
                ACTION_TO_ID["wait"],
            ],
        )
        self.assertEqual(states_grid[0, 1, 0, 1], 1.0)  # wall at global (0, 1)
        self.assertEqual(states_grid[0, 2, 2, 4], 1.0)  # agent 1 at global (2, 4)
        self.assertEqual(states_grid[0, 0, 2, 2], 1.0)  # current agent cell is free

    def test_extract_state_action_pairs_rejects_non_neighbor_move(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported move"):
            extract_state_action_pairs(
                np.zeros((3, 3), dtype=np.uint8),
                {0: (2, 2)},
                {0: [(0, 0), (1, 1)]},
            )

    def test_empty_single_step_paths_return_empty_arrays(self) -> None:
        states_grid, goal_dirs, actions = extract_state_action_pairs(
            np.zeros((2, 2), dtype=np.uint8),
            {0: (0, 0)},
            {0: [(0, 0)]},
        )

        self.assertEqual(states_grid.shape, (0, 3, 5, 5))
        self.assertEqual(goal_dirs.shape, (0, 2))
        self.assertEqual(actions.shape, (0,))

    def test_export_expert_paths_npz_writes_expected_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = export_expert_paths_npz(
                grid_map=np.zeros((2, 2), dtype=np.uint8),
                goals=[(0, 1)],
                paths={0: [(0, 0), (0, 1)]},
                save_path=Path(tmp_dir) / "expert.npz",
            )
            data = np.load(output_path)

        self.assertEqual(data["states_grid"].shape, (1, 3, 5, 5))
        self.assertEqual(data["goal_dirs"].tolist(), [[0.0, 1.0]])
        self.assertEqual(data["actions"].tolist(), [ACTION_TO_ID["right"]])


if __name__ == "__main__":
    unittest.main()
