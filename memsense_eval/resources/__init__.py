"""Memsense-specific resource plugins.

Importing this module registers all built-in resources with the engine's
global ``name_to_resource`` registry.
"""

# Side-effect imports — each module calls @register_resource on load.
from memsense_eval.resources import (  # noqa: F401
    locomo_reader,
    ingest,
    embedding_wait,
    qa,
    judge,
    filter,
    summary,
    qa_results_reader,
)
