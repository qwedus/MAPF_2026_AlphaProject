# CBS Adapter Note

This branch integrates an open-source Conflict-Based Search (CBS) solver through
a subprocess adapter.

## Current Scope

- Convert the project coordinate format `(row, col)` to atb033 `[x, y]`.
- Write an atb033-compatible `input.yaml`.
- Run `centralized/cbs/cbs.py` with `subprocess.run()`.
- Read `output.yaml` and convert the solver schedule back to internal paths.
- Save internal paths to `outputs/paths/atb033_example_internal_paths.json`.
- Run a fixed 5x5 example from `scripts/run_atb033_example.py`.

## Coordinate Convention

Project internal coordinates use `(row, col)`:

- `row` increases from top to bottom.
- `col` increases from left to right.
- The origin is the top-left cell `(0, 0)`.

The atb033 solver uses `[x, y]`, so the adapter converts:

- Internal to atb033: `(row, col) -> [col, row]`
- atb033 to internal: `[x, y] -> (y, x)`

`CBSAdapter.plan()` always accepts starts and goals as internal `(row, col)`
coordinates. The atb033 `[x, y]` conversion is performed only inside
`src/cbs_adapter.py`.

## Map Generator Scenarios

`map_generator.py` has two coordinate surfaces:

- The direct return values from `generate_map()` are `grid_map`, `starts_rc`,
  and `goals_rc`. These starts/goals are already internal `(row, col)`, so they
  can be passed directly to `CBSAdapter.plan()`.
- The saved scenario JSON uses `agents[*].start` and `agents[*].goal` in
  external `[x, y]` format.

When reading a saved map-generator scenario JSON, use
`load_map_generator_scenario()`:

```python
from src.scenario_loader import load_map_generator_scenario

grid_map, starts, goals = load_map_generator_scenario("scenarios/scenario_ex.json")
paths = adapter.plan(starts=starts, goals=goals, grid=grid_map)
```

The loader reads `scenario["map_file"]` with `np.load()` and converts each
agent start/goal from `[x, y]` to `(row, col)` before the adapter sees it.

You can run one saved map-generator scenario with:

```bash
python scripts/run_map_generator_scenario_example.py scenarios/scenario_ex.json
```

## Internal Paths JSON

The example script writes paths in a simulator-facing JSON format:

```json
{
  "coordinate_format": "row_col",
  "paths": {
    "0": [[0, 0], [1, 0]],
    "1": [[4, 4], [3, 4]]
  }
}
```

JSON object keys are strings, so agent IDs are stringified in the file. The
in-memory Python format remains `dict[int, list[tuple[int, int]]]`.

## Batch Runner

`scripts/run_cbs_batch.py` runs the adapter over scenario files from a directory:

```bash
python scripts/run_cbs_batch.py --scenarios-dir inputs/scenarios --output-dir outputs/paths
```

Supported scenario files are `.json`, `.yaml`, and `.yml`. Coordinates are
internal `[row, col]` coordinates for inline starts/goals. Three input shapes
are supported:

```json
{
  "scenario_id": "scenario_0001",
  "grid_map": [[0, 0], [0, 0]],
  "starts": [[0, 0]],
  "goals": [[1, 1]]
}
```

or:

```json
{
  "grid": [[0, 0], [0, 0]],
  "agents": [
    {"start": [0, 0], "goal": [1, 1]}
  ]
}
```

Map-generator scenario JSON files are also supported:

```json
{
  "scenario_id": "scenario_ex",
  "map_file": "scenarios/map_ex.npy",
  "agents": [
    {"agent_id": 0, "start": [0, 0], "goal": [4, 4]}
  ]
}
```

For this `map_file` shape, `start` and `goal` are interpreted as `[x, y]` and
converted through `load_map_generator_scenario()`.

Each successful scenario writes:

- `outputs/paths/<scenario_id>/input.yaml`
- `outputs/paths/<scenario_id>/output.yaml`
- `outputs/paths/<scenario_id>/paths.json`
- `outputs/paths/<scenario_id>/status.json`

`paths.json` is the simulator-facing file. `status.json` records whether CBS
succeeded and where the artifacts were written.

## Not Included Yet

- CBS algorithm implementation.
- Direct imports of atb033 `Environment` or `CBS` classes.
- Map generator integration.
- Scenario parsing beyond the fixed example script.
- Collision validation. Vertex and edge collision checks belong to the simulator
  module and can be imported into the example script when that module is ready.
