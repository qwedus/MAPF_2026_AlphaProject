"""Build a single IL smoke dataset package from per-scenario CBS outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_exporter import ACTION_TO_ID
from src.expert_handoff import SCHEMA_VERSION

DATASET_VERSION = "il_smoke_v0.3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine successful per-scenario expert_dataset.npz files.",
    )
    parser.add_argument(
        "--paths-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/paths/il_smoke_v0_3",
        help="Directory containing per-scenario CBS batch outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/datasets/il_smoke_v0_3",
        help="Directory where the smoke dataset package is written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="smoke_dataset.npz",
        help="Combined NPZ filename.",
    )
    return parser.parse_args(argv)


def build_smoke_dataset_package(
    paths_dir: Path,
    output_dir: Path,
    dataset_name: str,
) -> dict[str, Any]:
    """Combine successful per-scenario datasets and write package metadata."""
    statuses = _load_status_payloads(paths_dir)
    successful_statuses = [
        status for status in statuses if status.get("success") is True
    ]
    failed_statuses = [status for status in statuses if status.get("success") is not True]

    if not successful_statuses:
        raise ValueError(f"No successful scenario statuses found under {paths_dir}.")

    state_batches: list[np.ndarray] = []
    goal_dir_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    scenario_id_batches: list[np.ndarray] = []
    scenario_summaries: list[dict[str, Any]] = []

    for status in successful_statuses:
        scenario_id = str(status["scenario_id"])
        dataset_path = Path(status["expert_dataset_npz_path"])
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Missing expert dataset for scenario {scenario_id}: {dataset_path}"
            )

        with np.load(dataset_path) as data:
            states_grid = np.asarray(data["states_grid"], dtype=np.float32)
            goal_dirs = np.asarray(data["goal_dirs"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.int64)

        _validate_dataset_arrays(scenario_id, states_grid, goal_dirs, actions)
        sample_count = int(actions.shape[0])
        state_batches.append(states_grid)
        goal_dir_batches.append(goal_dirs)
        action_batches.append(actions)
        scenario_id_batches.append(np.full(sample_count, scenario_id))
        scenario_summaries.append(
            {
                "scenario_id": scenario_id,
                "num_agents": int(status["num_agents"]),
                "makespan": int(status["makespan"]),
                "sample_count": sample_count,
                "status_path": str(_status_path(paths_dir, scenario_id)),
                "expert_handoff_json_path": status["expert_handoff_json_path"],
                "expert_dataset_npz_path": status["expert_dataset_npz_path"],
                "validation_success": bool(status["validation"]["success"]),
            }
        )

    states_grid = np.concatenate(state_batches, axis=0)
    goal_dirs = np.concatenate(goal_dir_batches, axis=0)
    actions = np.concatenate(action_batches, axis=0)
    scenario_ids = np.concatenate(scenario_id_batches, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = (output_dir / dataset_name).resolve()
    np.savez_compressed(
        dataset_path,
        states_grid=states_grid,
        goal_dirs=goal_dirs,
        actions=actions,
        scenario_ids=scenario_ids,
    )

    schema_path = (output_dir / "schema.json").resolve()
    _write_json(schema_path, _schema_payload())

    manifest = _manifest_payload(
        paths_dir=paths_dir,
        dataset_path=dataset_path,
        schema_path=schema_path,
        scenarios=scenario_summaries,
        failed_statuses=failed_statuses,
        actions=actions,
    )
    manifest_path = (output_dir / "manifest.json").resolve()
    _write_json(manifest_path, manifest)

    return manifest


def _load_status_payloads(paths_dir: Path) -> list[dict[str, Any]]:
    if not paths_dir.is_dir():
        raise FileNotFoundError(f"paths directory not found: {paths_dir}")

    statuses: list[dict[str, Any]] = []
    for status_path in sorted(paths_dir.glob("*/status.json")):
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        statuses.append(status)
    if not statuses:
        raise FileNotFoundError(f"No status.json files found under {paths_dir}.")
    return statuses


def _validate_dataset_arrays(
    scenario_id: str,
    states_grid: np.ndarray,
    goal_dirs: np.ndarray,
    actions: np.ndarray,
) -> None:
    if states_grid.ndim != 4 or states_grid.shape[1:] != (3, 5, 5):
        raise ValueError(
            f"{scenario_id}: states_grid must have shape (N, 3, 5, 5), "
            f"got {states_grid.shape}."
        )
    if goal_dirs.shape != (states_grid.shape[0], 2):
        raise ValueError(
            f"{scenario_id}: goal_dirs must have shape (N, 2), "
            f"got {goal_dirs.shape}."
        )
    if actions.shape != (states_grid.shape[0],):
        raise ValueError(
            f"{scenario_id}: actions must have shape (N,), got {actions.shape}."
        )
    valid_actions = set(ACTION_TO_ID.values())
    invalid_actions = sorted(set(actions.tolist()) - valid_actions)
    if invalid_actions:
        raise ValueError(f"{scenario_id}: invalid action ids {invalid_actions}.")


def _schema_payload() -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "source_handoff_schema_version": SCHEMA_VERSION,
        "coordinate_format": "row_col",
        "purpose": "IL loader and training-pipeline smoke test before simulator final validation.",
        "simulator_validation": "not included; final train/val/test must be regenerated after simulator validation.",
        "arrays": {
            "states_grid": {
                "shape": ["N", 3, 5, 5],
                "dtype": "float32",
                "channels": {
                    "0": "wall, obstacle, or out-of-bounds",
                    "1": "other agents' current positions",
                    "2": "other agents' goal positions",
                },
            },
            "goal_dirs": {
                "shape": ["N", 2],
                "dtype": "float32",
                "meaning": "[goal_row - row, goal_col - col]",
            },
            "actions": {
                "shape": ["N"],
                "dtype": "int64",
                "encoding": {name: int(value) for name, value in ACTION_TO_ID.items()},
            },
            "scenario_ids": {
                "shape": ["N"],
                "dtype": "string",
                "meaning": "source scenario id for each sample",
            },
        },
    }


def _manifest_payload(
    paths_dir: Path,
    dataset_path: Path,
    schema_path: Path,
    scenarios: list[dict[str, Any]],
    failed_statuses: list[dict[str, Any]],
    actions: np.ndarray,
) -> dict[str, Any]:
    action_counts = {
        action_name: int(np.count_nonzero(actions == action_id))
        for action_name, action_id in ACTION_TO_ID.items()
    }
    return {
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "source_paths_dir": str(paths_dir.resolve()),
        "total_samples": int(actions.shape[0]),
        "successful_scenarios": len(scenarios),
        "failed_scenarios": len(failed_statuses),
        "action_counts": action_counts,
        "scenarios": scenarios,
        "failures": [
            {
                "scenario_id": str(status.get("scenario_id")),
                "source_path": status.get("source_path"),
                "error": status.get("error"),
                "status_path": str(_status_path(paths_dir, str(status.get("scenario_id")))),
            }
            for status in failed_statuses
        ],
    }


def _status_path(paths_dir: Path, scenario_id: str) -> Path:
    return (paths_dir / scenario_id / "status.json").resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_smoke_dataset_package(
        paths_dir=args.paths_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        dataset_name=args.dataset_name,
    )
    print(f"smoke dataset: {manifest['dataset_path']}")
    print(f"schema: {manifest['schema_path']}")
    print(f"manifest: {Path(manifest['schema_path']).with_name('manifest.json')}")
    print(
        "summary: "
        f"{manifest['successful_scenarios']} succeeded, "
        f"{manifest['failed_scenarios']} failed, "
        f"{manifest['total_samples']} samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
