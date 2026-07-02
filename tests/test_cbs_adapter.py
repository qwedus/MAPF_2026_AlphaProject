import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.cbs_adapter import (
    atb033_to_internal,
    convert_schedule_to_internal_paths,
    create_atb033_input,
    internal_to_atb033,
    pad_paths,
    save_internal_paths_json,
)


class CBSAdapterFormatTests(unittest.TestCase):
    def test_coordinate_conversion(self) -> None:
        self.assertEqual(internal_to_atb033((2, 5)), [5, 2])
        self.assertEqual(atb033_to_internal({"x": 5, "y": 2}), (2, 5))

    def test_create_atb033_input_writes_solver_yaml(self) -> None:
        grid_map = [
            [0, 1, 0],
            [0, 0, 0],
        ]
        starts = [(0, 0), (1, 2)]
        goals = [(1, 0), (0, 2)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.yaml"
            create_atb033_input(grid_map, starts, goals, input_path)

            with input_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)

        self.assertEqual(payload["map"]["dimensions"], [3, 2])
        self.assertEqual(payload["map"]["obstacles"], [[1, 0]])
        self.assertEqual(
            payload["agents"],
            [
                {"start": [0, 0], "goal": [0, 1], "name": "agent0"},
                {"start": [2, 1], "goal": [2, 0], "name": "agent1"},
            ],
        )

    def test_create_atb033_input_rejects_blocked_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "blocked"):
                create_atb033_input(
                    grid_map=[[1]],
                    starts=[(0, 0)],
                    goals=[(0, 0)],
                    save_path=Path(tmp_dir) / "input.yaml",
                )

    def test_convert_schedule_sorts_steps_and_agent_ids(self) -> None:
        schedule = {
            "agent1": [{"x": 2, "y": 0, "t": 1}, {"x": 2, "y": 1, "t": 0}],
            "agent0": [{"x": 0, "y": 0, "t": 0}, {"x": 0, "y": 1, "t": 1}],
        }

        self.assertEqual(
            convert_schedule_to_internal_paths(schedule),
            {
                0: [(0, 0), (1, 0)],
                1: [(1, 2), (0, 2)],
            },
        )

    def test_pad_paths_repeats_final_location(self) -> None:
        self.assertEqual(
            pad_paths({0: [(0, 0)], 1: [(1, 0), (1, 1), (1, 2)]}),
            {
                0: [(0, 0), (0, 0), (0, 0)],
                1: [(1, 0), (1, 1), (1, 2)],
            },
        )

    def test_save_internal_paths_json_uses_string_agent_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = save_internal_paths_json(
                {1: [(4, 4)], 0: [(0, 0), (0, 1)]},
                Path(tmp_dir) / "paths.json",
            )
            with output_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["coordinate_format"], "row_col")
        self.assertEqual(
            payload["paths"],
            {
                "0": [[0, 0], [0, 1]],
                "1": [[4, 4]],
            },
        )


if __name__ == "__main__":
    unittest.main()
