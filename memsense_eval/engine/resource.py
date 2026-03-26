"""Base Resource abstraction, registry, and factory.

Every evaluation capability (data loading, API calls, grading, etc.) is
implemented as a Resource with a unified ``async process(*args) -> tuple``
interface.  Resources are registered by name and instantiated from YAML
configuration via :func:`create_resource`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ResourceConfig(BaseModel):
    """Declarative config for a single resource instance."""

    name: str
    config: dict[str, Any] | None = None


class BaseResource(ABC):
    """All resources must subclass this and implement *process*."""

    @abstractmethod
    async def process(self, *args: Any) -> tuple:
        """Run the resource logic.

        Returns a **tuple** whose length must equal the number of ``obtain``
        traces declared in the corresponding FlowConfig.
        """
        ...


# ---------------------------------------------------------------------------
# Global registry  (name  →  resource class)
# ---------------------------------------------------------------------------

name_to_resource: dict[str, type[BaseResource]] = {}


def register_resource(name: str):
    """Decorator that registers a Resource class under *name*."""

    def _decorator(cls: type[BaseResource]):
        name_to_resource[name] = cls
        return cls

    return _decorator


def create_resource(config: ResourceConfig) -> BaseResource:
    """Instantiate a resource from its :class:`ResourceConfig`."""
    if config.name not in name_to_resource:
        available = ", ".join(sorted(name_to_resource)) or "(none)"
        raise KeyError(
            f"Unknown resource '{config.name}'. Available: {available}"
        )
    cls = name_to_resource[config.name]
    return cls(**(config.config or {}))
