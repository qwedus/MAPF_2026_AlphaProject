import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
import numpy as np

from scripts.run_cbs_batch import discover_scenario_files, load_scenario, run_batch


class RunCBSBatchTests(unittest.TestCase):
    def test_discover_scenario_files_returns_supported_files_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenarios_dir = Path(tmp_dir)
            (scenarios_dir / "scenario_0002.yaml").write_text(
                "grid_map: [[0]]\nstarts: [[0, 0]]\ngoals: [[0, 0]]\n",
                encoding="utf-8",
            )
            (scenarios_dir / "scenario_0001.json").write_text(
                '{"grid_map": [[0]], "starts": [[0, 0]], "goals": [[0, 0]]}',
                encoding="utf-8",
            )
            (scenarios_dir / "notes.txt").write_text("ignored", encoding="utf-8")

            discovered = discover_scenario_files(scenarios_dir)

        self.assertEqual(
            [path.name for path in discovered],
            ["scenario_0001.json", "scenario_0002.yaml"],
        )

    def test_load_scenario_with_starts_and_goals_json(self) -> None:
        payload = {
            "scenario_id": "scenario_custom",
            "grid_map": [[0, 1], [0, 0]],
            "starts": [[0, 0], [1, 0]],
            "goals": [[1, 1], [0, 0]],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "scenario_0001.json"
            scenario_path.write_text(json.dumps(payload), encoding="utf-8")
            scenario = load_scenario(scenario_path)

        self.assertEqual(scenario.scenario_id, "scenario_custom")
        self.assertEqual(scenario.grid_map, [[0, 1], [0, 0]])
        self.assertEqual(scenario.starts, [(0, 0), (1, 0)])
        self.assertEqual(scenario.goals, [(1, 1), (0, 0)])

    def test_load_scenario_with_agents_yaml(self) -> None:
        payload = {
            "grid": [[0, 0], [0, 0]],
            "agents": [
                {"start": [0, 0], "goal": [1, 1]},
                {"start": [1, 0], "goal": [0, 1]},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "scenario_agents.yaml"
            scenario_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            scenario = load_scenario(scenario_path)

        self.assertEqual(scenario.scenario_id, "scenario_agents")
        self.assertEqual(scenario.grid_map, [[0, 0], [0, 0]])
        self.assertEqual(scenario.starts, [(0, 0), (1, 0)])
        self.assertEqual(scenario.goals, [(1, 1), (0, 1)])

    def test_load_scenario_accepts_map_grid(self) -> None:
        payload = {
            "map": {"grid": [[0, 0], [1, 0]]},
            "starts": [[0, 0]],
            "goals": [[1, 1]],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "scenario_map_grid.json"
            scenario_path.write_text(json.dumps(payload), encoding="utf-8")
            scenario = load_scenario(scenario_path)

        self.assertEqual(scenario.grid_map, [[0, 0], [1, 0]])

    def test_load_scenario_accepts_map_generator_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_dir = Path(tmp_dir)
            map_path = scenario_dir / "map_ex.npy"
            np.save(map_path, np.zeros((2, 3), dtype=np.uint8))
            scenario_path = scenario_dir / "scenario_ex.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "scenario_id": "scenario_ex",
                        "map_file": "map_ex.npy",
                        "agents": [
                            {"agent_id": 0, "start": [2, 1], "goal": [0, 0]}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            scenario = load_scenario(scenario_path)

        self.assertEqual(scenario.scenario_id, "scenario_ex")
        self.assertEqual(scenario.grid_map.shape, (2, 3))
        self.assertEqual(scenario.starts, [(1, 2)])
        self.assertEqual(scenario.goals, [(0, 0)])

    def test_load_scenario_rejects_missing_agents_and_goals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scenario_path = Path(tmp_dir) / "bad.json"
            scenario_path.write_text(
                json.dumps({"grid_map": [[0]], "starts": [[0, 0]]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "starts/goals"):
                load_scenario(scenario_path)

    def test_run_batch_writes_paths_and_status(self) -> None:
        class FakeCBSAdapter:
            def __init__(self, config) -> None:
                self.config = config
                self.last_run = None

            def plan(self, starts, goals, grid):
                self.last_run = SimpleNamespace(
                    input_yaml_path=self.config.work_dir / "input.yaml",
                    output_yaml_path=self.config.work_dir / "output.yaml",
                )
                return {0: [(0, 0), (0, 1)]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scenario_path = root / "scenario_0001.json"
            output_dir = root / "paths"
            scenario_path.write_text(
                json.dumps(
                    {
                        "grid_map": [[0, 0]],
                        "starts": [[0, 0]],
                        "goals": [[0, 1]],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch("scripts.run_cbs_batch.CBSAdapter", FakeCBSAdapter),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                failures = run_batch([scenario_path], output_dir)

            paths_payload = json.loads(
                (output_dir / "scenario_0001" / "paths.json").read_text(
                    encoding="utf-8"
                )
            )
            status_payload = json.loads(
                (output_dir / "scenario_0001" / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            handoff_payload = json.loads(
                (output_dir / "scenario_0001" / "expert_handoff.json").read_text(
                    encoding="utf-8"
                )
            )
            dataset = np.load(output_dir / "scenario_0001" / "expert_dataset.npz")

        self.assertEqual(failures, 0)
        self.assertEqual(paths_payload["coordinate_format"], "row_col")
        self.assertEqual(paths_payload["paths"], {"0": [[0, 0], [0, 1]]})
        self.assertEqual(handoff_payload["schema_version"], "cbs_il_handoff_v0.2")
        self.assertEqual(handoff_payload["agents"]["0"]["goal"], [0, 1])
        self.assertTrue(handoff_payload["validation"]["success"])
        self.assertEqual(dataset["states_grid"].shape, (1, 3, 5, 5))
        self.assertEqual(dataset["goal_dirs"].tolist(), [[0.0, 1.0]])
        self.assertEqual(dataset["actions"].tolist(), [3])
        self.assertTrue(status_payload["success"])
        self.assertEqual(status_payload["num_agents"], 1)
        self.assertEqual(status_payload["makespan"], 2)
        self.assertTrue(status_payload["validation"]["success"])
        self.assertEqual(
            Path(status_payload["expert_handoff_json_path"]).name,
            "expert_handoff.json",
        )
        self.assertEqual(
            Path(status_payload["expert_dataset_npz_path"]).name,
            "expert_dataset.npz",
        )


if __name__ == "__main__":
    unittest.main()
