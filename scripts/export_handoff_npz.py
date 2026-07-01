"""Convert a CBS expert handoff JSON into an IL v0.2 NPZ dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.expert_handoff import export_handoff_npz


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert expert_handoff.json into an IL v0.2 .npz dataset.",
    )
    parser.add_argument(
        "handoff_json",
        type=Path,
        help="Path to expert_handoff.json.",
    )
    parser.add_argument(
        "output_npz",
        type=Path,
        nargs="?",
        help="Output .npz path. Defaults to expert_dataset.npz next to the JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handoff_json = args.handoff_json.resolve()
    output_npz = (
        args.output_npz.resolve()
        if args.output_npz is not None
        else handoff_json.parent / "expert_dataset.npz"
    )

    saved_path = export_handoff_npz(handoff_json, output_npz)
    print(f"expert handoff JSON: {handoff_json}")
    print(f"IL v0.2 NPZ: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
