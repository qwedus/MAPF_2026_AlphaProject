"""Placeholder runner for the future ATB033 CBS adapter example."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cbs_adapter import CBSAdapter, CBSAdapterConfig


def main() -> int:
    CBSAdapter(CBSAdapterConfig())
    print("CBS adapter skeleton is ready. CBS implementation is not integrated yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

