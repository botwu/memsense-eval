"""Core evaluation engine: Resource registry, Flow config, DataManager, Pipeline."""

from memsense_eval.engine.resource import (
    BaseResource,
    ResourceConfig,
    create_resource,
    name_to_resource,
    register_resource,
)
from memsense_eval.engine.flow import FlowConfig
from memsense_eval.engine.data_manager import DataManager
from memsense_eval.engine.pipeline import PipelineEngine, PipelineConfig

__all__ = [
    "BaseResource",
    "ResourceConfig",
    "create_resource",
    "name_to_resource",
    "register_resource",
    "FlowConfig",
    "DataManager",
    "PipelineEngine",
    "PipelineConfig",
]
