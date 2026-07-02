import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.scenario_loader import load_map_generator_scenario, xy_to_rowcol


class ScenarioLoaderTests(unittest.TestCase):
    def test_xy_to_rowcol_converts_xy_to_internal_coordinates(self) -> None:
        self.assertEqual(xy_to_rowcol([5, 2]), (2, 5))
        self.assertEqual(xy_to_rowcol((0, 3)), (3, 0))

    def test_load_map_generator_scenario_resolves_parent_relative_map_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_dir = Path(tmp_dir)
            map_path = scenario_dir / "map_ex.npy"
            grid_map = np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=np.uint8,
            )
            np.save(map_path, grid_map)
            scenario_path = scenario_dir / "scenario_ex.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "scenario_id": "scenario_ex",
                        "map_file": "map_ex.npy",
                        "agents": [
                            {"agent_id": 0, "start": [2, 1], "goal": [0, 0]},
                            {"agent_id": 1, "start": [0, 1], "goal": [2, 0]},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            loaded_grid, starts, goals = load_map_generator_scenario(scenario_path)

        np.testing.assert_array_equal(loaded_grid, grid_map)
        self.assertEqual(starts, [(1, 2), (1, 0)])
        self.assertEqual(goals, [(0, 0), (0, 2)])

    def test_load_map_generator_scenario_resolves_absolute_map_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_dir = Path(tmp_dir)
            map_path = scenario_dir / "map_abs.npy"
            grid_map = np.zeros((2, 2), dtype=np.uint8)
            np.save(map_path, grid_map)
            scenario_path = scenario_dir / "scenario_abs.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "map_file": str(map_path),
                        "agents": [{"start": [1, 0], "goal": [0, 1]}],
                    }
                ),
                encoding="utf-8",
            )

            loaded_grid, starts, goals = load_map_generator_scenario(scenario_path)

        np.testing.assert_array_equal(loaded_grid, grid_map)
        self.assertEqual(starts, [(0, 1)])
        self.assertEqual(goals, [(1, 0)])

    def test_load_map_generator_scenario_missing_map_file_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "scenario_missing.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "map_file": "missing.npy",
                        "agents": [{"start": [0, 0], "goal": [1, 1]}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Could not find"):
                load_map_generator_scenario(scenario_path)


if __name__ == "__main__":
    unittest.main()
