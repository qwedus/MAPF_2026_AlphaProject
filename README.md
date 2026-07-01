# MAPF 2026 Alpha Project

This branch contains the CBS adapter pipeline for multi-agent path finding
(MAPF). The adapter connects our project data format to the atb033
Conflict-Based Search solver through subprocess execution.

## CBS Adapter Role

The CBS adapter is responsible for:

- receiving `grid_map`, `starts`, and `goals`
- converting internal `(row, col)` coordinates to solver `[x, y]`
- running the atb033 CBS solver
- converting solver schedules back to internal paths
- saving paths for the simulator team

Collision validation is intentionally not implemented here. Vertex and edge
collision checks belong to the simulator module.

## Coordinate Convention

Project-internal coordinates are always `(row, col)`:

- `row` increases downward
- `col` increases rightward
- origin is the top-left cell `(0, 0)`

The atb033 solver uses `[x, y]`. That conversion happens only inside
`src/cbs_adapter.py`.

Map-generator scenario JSON files store agent `start` and `goal` as `[x, y]`.
Use `src/scenario_loader.py` to convert them before calling the adapter.

## Common Commands

Run the fixed CBS smoke example:

```bash
python scripts/run_atb033_example.py
```

Run CBS for a map-generator scenario JSON:

```bash
python scripts/run_map_generator_scenario_example.py scenarios/scenario_ex.json
```

Run CBS over a directory of scenarios:

```bash
python scripts/run_cbs_batch.py --scenarios-dir inputs/scenarios --output-dir outputs/paths
```

Convert one CBS expert handoff JSON to an IL v0.2 NPZ dataset:

```bash
python scripts/export_handoff_npz.py outputs/paths/<scenario_id>/expert_handoff.json
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Outputs

Batch CBS runs write per-scenario artifacts under:

```text
outputs/paths/<scenario_id>/
```

Typical files:

- `input.yaml`: atb033 solver input
- `output.yaml`: atb033 solver output
- `paths.json`: simulator-facing internal `(row, col)` paths
- `expert_handoff.json`: CBS-to-IL handoff artifact with map, goals, paths, and validation
- `expert_dataset.npz`: IL v0.2 dataset with `states_grid`, `goal_dirs`, and `actions`
- `status.json`: success/failure metadata

Example scripts may write temporary smoke-test outputs to `/tmp` to avoid
polluting the repository.

## More Notes

See `docs/cbs_adapter_note.md` for implementation details and scenario format
examples.
