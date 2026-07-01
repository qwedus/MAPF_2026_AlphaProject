import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.expert_handoff import (
    build_expert_handoff_payload,
    export_handoff_npz,
    handoff_to_training_inputs,
    save_expert_handoff_json,
)


class ExpertHandoffTests(unittest.TestCase):
    def test_build_handoff_payload_contains_il_training_inputs(self) -> None:
        grid_map = np.zeros((3, 3), dtype=np.uint8)
        starts = [(0, 0), (2, 2)]
        goals = [(0, 2), (2, 0)]
        paths = {
            0: [(0, 0), (0, 1), (0, 2)],
            1: [(2, 2), (2, 1), (2, 0)],
        }

        payload = build_expert_handoff_payload(
            scenario_id="scenario_0001",
            source_path="scenario_0001.json",
            grid_map=grid_map,
            starts=starts,
            goals=goals,
            paths=paths,
        )

        self.assertEqual(payload["schema_version"], "cbs_il_handoff_v0.2")
        self.assertEqual(payload["coordinate_format"], "row_col")
        self.assertEqual(payload["map"]["grid_shape"], [3, 3])
        self.assertEqual(payload["map"]["grid"], grid_map.tolist())
        self.assertEqual(payload["agents"]["0"]["goal"], [0, 2])
        self.assertEqual(payload["agents"]["0"]["path"], [[0, 0], [0, 1], [0, 2]])
        self.assertTrue(payload["validation"]["success"])
        self.assertTrue(payload["validation"]["no_blocked_cells"])
        self.assertTrue(payload["validation"]["no_vertex_collisions"])
        self.assertTrue(payload["validation"]["no_edge_collisions"])

    def test_handoff_json_exports_il_v02_npz(self) -> None:
        grid_map = np.zeros((3, 3), dtype=np.uint8)
        payload = build_expert_handoff_payload(
            scenario_id="scenario_0001",
            source_path="scenario_0001.json",
            grid_map=grid_map,
            starts=[(1, 1)],
            goals=[(1, 2)],
            paths={0: [(1, 1), (1, 2)]},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            handoff_path = save_expert_handoff_json(
                payload,
                Path(tmp_dir) / "expert_handoff.json",
            )
            output_path = export_handoff_npz(
                handoff_path,
                Path(tmp_dir) / "expert_dataset.npz",
            )
            data = np.load(output_path)

        self.assertEqual(data["states_grid"].shape, (1, 3, 5, 5))
        self.assertEqual(data["goal_dirs"].tolist(), [[0.0, 1.0]])
        self.assertEqual(data["actions"].tolist(), [3])

    def test_handoff_to_training_inputs_loads_inline_grid(self) -> None:
        payload = {
            "schema_version": "cbs_il_handoff_v0.2",
            "coordinate_format": "row_col",
            "map": {"grid": [[0, 1], [0, 0]], "map_file": None},
            "agents": {
                "0": {
                    "goal": [1, 1],
                    "path": [[0, 0], [1, 0], [1, 1]],
                }
            },
        }

        grid, goals, paths = handoff_to_training_inputs(payload)

        self.assertEqual(grid.tolist(), [[0, 1], [0, 0]])
        self.assertEqual(goals, {0: (1, 1)})
        self.assertEqual(paths, {0: [(0, 0), (1, 0), (1, 1)]})

    def test_validation_records_blocked_cells(self) -> None:
        payload = build_expert_handoff_payload(
            scenario_id="scenario_blocked",
            source_path="scenario_blocked.json",
            grid_map=np.array([[0, 1]], dtype=np.uint8),
            starts=[(0, 0)],
            goals=[(0, 1)],
            paths={0: [(0, 0), (0, 1)]},
        )

        self.assertFalse(payload["validation"]["success"])
        self.assertEqual(
            payload["validation"]["blocked_or_outside"],
            [
                {
                    "agent_id": 0,
                    "timestep": 1,
                    "location": [0, 1],
                    "reason": "blocked",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
