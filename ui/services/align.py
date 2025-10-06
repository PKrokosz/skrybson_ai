"""Alignment related helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(slots=True)
class AlignItem:
    path: Path
    selected: bool = True


def discover_alignment_candidates(output_dir: Path) -> List[AlignItem]:
    candidates: List[AlignItem] = []
    if not output_dir.exists():
        return candidates
    for path in sorted(output_dir.glob("**/*.json")):
        if path.name.endswith(".aligned.json"):
            continue
        candidates.append(AlignItem(path=path))
    return candidates


__all__ = ["AlignItem", "discover_alignment_candidates"]
