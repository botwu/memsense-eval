"""Declarative flow configuration.

A *flow* binds one Resource to a set of input traces (``by``) and output
traces (``obtain``).  Wildcard segments ``~0`` … ``~9`` allow a single flow
definition to match many concrete trace paths (e.g. per-sample).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


_WILDCARDS = {f"~{i}" for i in range(10)}


def is_wildcard(segment: str) -> bool:
    return segment in _WILDCARDS


class FlowConfig(BaseModel):
    """One step in the evaluation pipeline."""

    use: str
    by: list[list[str]] | None = None
    obtain: list[list[str]] | None = None
    reuse: bool = False

    def __hash__(self) -> int:
        return hash((
            self.use,
            str(self.by) if self.by else None,
            str(self.obtain) if self.obtain else None,
            self.reuse,
        ))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FlowConfig:
        if "use" not in raw:
            raise ValueError(f"Flow config missing 'use': {raw}")

        by_raw = raw.get("by")
        obtain_raw = raw.get("obtain")

        by = [p.split(".") for p in by_raw] if by_raw else None
        obtain = [p.split(".") for p in obtain_raw] if obtain_raw else None

        return cls(
            use=raw["use"],
            by=by,
            obtain=obtain,
            reuse=raw.get("reuse", False),
        )


# ---- Wildcard matching utilities ----

def match_single_trace(
    need: list[str], ready: list[str]
) -> dict[str, str] | None:
    """Try to match *need* (may contain wildcards) against *ready*.

    Returns the wildcard mapping on success, ``None`` on failure.
    """
    if len(need) != len(ready):
        return None
    mapping: dict[str, str] = {}
    for n, r in zip(need, ready):
        if is_wildcard(n):
            mapping[n] = r
        elif n != r:
            return None
    return mapping


def match_traces(
    need_traces: list[list[str]], ready_traces: list[list[str]]
) -> tuple[bool, dict[str, str] | None, list[list[str]]]:
    """Match a set of *need_traces* against all *ready_traces*.

    Returns ``(matched, wildcard_mapping, concrete_by_traces)``.
    """
    per_need: list[list[dict[str, str]]] = []
    for need in need_traces:
        mappings = []
        for ready in ready_traces:
            m = match_single_trace(need, ready)
            if m is not None:
                mappings.append(m)
        per_need.append(mappings)

    # Intersect mappings across all need traces
    candidates = per_need[0]
    for other in per_need[1:]:
        candidates = [c for c in candidates if c in other]

    if not candidates:
        return False, None, []

    mapping = candidates[0]
    concrete = [apply_mapping(mapping, t) for t in need_traces]
    return True, mapping, concrete


def apply_mapping(mapping: dict[str, str], trace: list[str]) -> list[str]:
    return [mapping.get(seg, seg) for seg in trace]


def apply_mapping_to_traces(
    mapping: dict[str, str], traces: list[list[str]]
) -> list[list[str]]:
    return [apply_mapping(mapping, t) for t in traces]
