"""Run CBS for every scenario file in a directory.

Expected scenario formats:

1. Explicit starts/goals:
   {
     "grid_map": [[0, 0], [0, 1]],
     "starts": [[0, 0]],
     "goals": [[1, 0]]
   }

2. Agent list:
   {
     "grid": [[0, 0], [0, 1]],
     "agents": [{"start": [0, 0], "goal": [1, 0]}]
   }

Coordinates are internal project coordinates: [row, col].
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig, save_internal_paths_json

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cbs_adapter already requires PyYAML
    raise ImportError("run_cbs_batch.py requires PyYAML.") from exc

GridLocation = tuple[int, int]

SUPPORTED_SCENARIO_SUFFIXES = {".json", ".yaml", ".yml"}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    source_path: Path
    grid_map: list[list[int]]
    starts: list[GridLocation]
    goals: list[GridLocation]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CBS adapter for every JSON/YAML scenario in a directory.",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=Path,
        default=PROJECT_ROOT / "inputs/scenarios",
        help="Directory containing scenario_*.json/.yaml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/paths",
        help="Directory where per-scenario CBS outputs are written.",
    )
    parser.add_argument(
        "--solver-root",
        type=Path,
        default=None,
        help="Optional path to atb033 centralized/cbs directory.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=30,
        help="Timeout per scenario.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to run atb033 cbs.py.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed scenario instead of continuing.",
    )
    return parser.parse_args(argv)


def discover_scenario_files(scenarios_dir: Path) -> list[Path]:
    """Return sorted JSON/YAML scenario files directly under ``scenarios_dir``."""
    if not scenarios_dir.is_dir():
        raise FileNotFoundError(f"scenarios directory not found: {scenarios_dir}")
    return sorted(
        path
        for path in scenarios_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SCENARIO_SUFFIXES
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load one scenario file into the internal CBS adapter input shape."""
    source_path = Path(path).resolve()
    payload = _load_structured_file(source_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario must be a mapping/object: {source_path}")

    scenario_id = str(payload.get("scenario_id") or source_path.stem)
    grid_map = _extract_grid_map(payload)
    starts, goals = _extract_starts_goals(payload)
    return Scenario(
        scenario_id=scenario_id,
        source_path=source_path,
        grid_map=grid_map,
        starts=starts,
        goals=goals,
    )


def run_batch(
    scenario_files: Iterable[Path],
    output_dir: Path,
    solver_root: Path | None = None,
    timeout_sec: int = 30,
    python_executable: str | None = None,
    stop_on_error: bool = False,
) -> int:
    """Run CBS for every scenario file and return the number of failures."""
    failures = 0
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_file in scenario_files:
        scenario = load_scenario(scenario_file)
        scenario_output_dir = output_dir / scenario.scenario_id
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        status_path = scenario_output_dir / "status.json"

        print(f"[RUN] {scenario.scenario_id} ({scenario.source_path})")
        adapter = CBSAdapter(
            CBSAdapterConfig(
                solver_root=solver_root,
                work_dir=scenario_output_dir,
                timeout_sec=timeout_sec,
                python_executable=python_executable,
            )
        )

        try:
            paths = adapter.plan(
                starts=scenario.starts,
                goals=scenario.goals,
                grid=scenario.grid_map,
            )
            paths_json_path = save_internal_paths_json(
                paths,
                scenario_output_dir / "paths.json",
            )
            makespan = max((len(path) for path in paths.values()), default=0)
            status = {
                "scenario_id": scenario.scenario_id,
                "source_path": str(scenario.source_path),
                "success": True,
                "num_agents": len(scenario.starts),
                "makespan": makespan,
                "paths_json_path": str(paths_json_path),
                "input_yaml_path": str(adapter.last_run.input_yaml_path)
                if adapter.last_run
                else None,
                "output_yaml_path": str(adapter.last_run.output_yaml_path)
                if adapter.last_run
                else None,
                "error": None,
            }
            print(f"[OK]  {scenario.scenario_id}: {paths_json_path}")
        except Exception as exc:  # noqa: BLE001 - batch runner must record failures
            failures += 1
            status = {
                "scenario_id": scenario.scenario_id,
                "source_path": str(scenario.source_path),
                "success": False,
                "num_agents": len(scenario.starts),
                "makespan": None,
                "paths_json_path": None,
                "input_yaml_path": str(adapter.last_run.input_yaml_path)
                if adapter.last_run
                else None,
                "output_yaml_path": str(adapter.last_run.output_yaml_path)
                if adapter.last_run
                else None,
                "error": str(exc),
            }
            print(f"[FAIL] {scenario.scenario_id}: {exc}", file=sys.stderr)
            if stop_on_error:
                _write_json(status_path, status)
                break

        _write_json(status_path, status)

    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenario_files = discover_scenario_files(args.scenarios_dir)
    if not scenario_files:
        print(f"No scenario files found in {args.scenarios_dir}", file=sys.stderr)
        return 1

    failures = run_batch(
        scenario_files=scenario_files,
        output_dir=args.output_dir,
        solver_root=args.solver_root,
        timeout_sec=args.timeout_sec,
        python_executable=args.python_executable,
        stop_on_error=args.stop_on_error,
    )
    print(
        f"Batch complete: {len(scenario_files) - failures} succeeded, "
        f"{failures} failed."
    )
    return 1 if failures else 0


def _load_structured_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def _extract_grid_map(payload: dict[str, Any]) -> list[list[int]]:
    grid = payload.get("grid_map") or payload.get("grid")
    if grid is None and isinstance(payload.get("map"), dict):
        grid = payload["map"].get("grid")
    if grid is None:
        raise ValueError("Scenario must include 'grid_map', 'grid', or 'map.grid'.")
    if not isinstance(grid, list) or not grid:
        raise ValueError("Scenario grid must be a non-empty 2-D list.")

    normalized_grid: list[list[int]] = []
    width: int | None = None
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or not row:
            raise ValueError(f"Grid row {row_index} must be a non-empty list.")
        normalized_row = [int(value) for value in row]
        width = len(normalized_row) if width is None else width
        if len(normalized_row) != width:
            raise ValueError("Scenario grid rows must all have the same width.")
        normalized_grid.append(normalized_row)
    return normalized_grid


def _extract_starts_goals(
    payload: dict[str, Any],
) -> tuple[list[GridLocation], list[GridLocation]]:
    if "starts" in payload and "goals" in payload:
        return (
            _normalize_locations(payload["starts"], "starts"),
            _normalize_locations(payload["goals"], "goals"),
        )

    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Scenario must include starts/goals or non-empty agents list.")

    starts: list[GridLocation] = []
    goals: list[GridLocation] = []
    for agent_index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {agent_index} must be an object.")
        if "start" not in agent or "goal" not in agent:
            raise ValueError(f"Agent {agent_index} must include start and goal.")
        starts.append(_normalize_location(agent["start"], f"agents[{agent_index}].start"))
        goals.append(_normalize_location(agent["goal"], f"agents[{agent_index}].goal"))
    return starts, goals


def _normalize_locations(values: Any, label: str) -> list[GridLocation]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list.")
    return [
        _normalize_location(location, f"{label}[{location_index}]")
        for location_index, location in enumerate(values)
    ]


def _normalize_location(value: Any, label: str) -> GridLocation:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [row, col].")
    return (int(value[0]), int(value[1]))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
