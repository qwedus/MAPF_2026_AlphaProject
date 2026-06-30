"""Skeleton adapter for future CBS solver integration.

This module intentionally does not implement Conflict-Based Search yet.
It only defines the integration surface that later CBS work can fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

GridLocation = tuple[int, int]
AgentPath = list[GridLocation]


@dataclass(frozen=True)
class CBSAdapterConfig:
    """Configuration placeholder for an external CBS implementation."""

    solver_root: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class CBSAdapter:
    """Adapter boundary for a future CBS implementation."""

    def __init__(self, config: CBSAdapterConfig | None = None) -> None:
        self.config = config or CBSAdapterConfig()

    def plan(
        self,
        starts: Sequence[GridLocation],
        goals: Sequence[GridLocation],
        grid: Any,
    ) -> dict[int, AgentPath]:
        """Return one path per agent after CBS integration is implemented."""
        raise NotImplementedError("CBS integration is not implemented yet.")

