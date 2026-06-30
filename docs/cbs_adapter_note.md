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

## Not Included Yet

- CBS algorithm implementation.
- Direct imports of atb033 `Environment` or `CBS` classes.
- Map generator integration.
- Scenario parsing beyond the fixed example script.
- Collision validation. Vertex and edge collision checks belong to the simulator
  module and can be imported into the example script when that module is ready.
