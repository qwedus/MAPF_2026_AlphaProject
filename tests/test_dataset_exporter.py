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
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        goals = {0: (1, 2)}
        paths = {0: [(0, 0), (1, 0), (1, 1), (1, 2)]}

        states_grid, goal_dirs, actions = extract_state_action_pairs(
            grid_map,
            goals,
            paths,
        )

        self.assertEqual(states_grid.shape, (3, 2, 2, 3))
        self.assertEqual(goal_dirs.tolist(), [[1, 1], [0, 1], [0, 1]])
        self.assertEqual(
            actions.tolist(),
            [
                ACTION_TO_ID["down"],
                ACTION_TO_ID["right"],
                ACTION_TO_ID["right"],
            ],
        )
        self.assertEqual(states_grid[0, 0].tolist(), grid_map.tolist())
        self.assertEqual(states_grid[0, 1, 0, 0], 1.0)

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

        self.assertEqual(states_grid.shape, (0, 2, 2, 2))
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

        self.assertEqual(data["states_grid"].shape, (1, 2, 2, 2))
        self.assertEqual(data["goal_dirs"].tolist(), [[0, 1]])
        self.assertEqual(data["actions"].tolist(), [ACTION_TO_ID["right"]])


if __name__ == "__main__":
    unittest.main()
