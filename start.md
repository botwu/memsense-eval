# Memsense Evaluation Framework

## Quick Start

```bash
# Install dependencies

uv sync

# Run the full pipeline (ingest → QA → filter → judge → summary)
uv run python -m memsense_eval configs/full_pipeline.yaml \
    --set benchmark.token=YOUR_TOKEN

# Or just re-judge existing QA results
uv run python -m memsense_eval configs/judge_only.yaml \
    --set benchmark.qa_results_path=output/qa.my_task.jsonl
```

## Architecture

```
memsense_eval/
├── engine/             # Core framework (Resource, Flow, DataManager, Pipeline)
│   ├── resource.py     # BaseResource + registry + factory
│   ├── flow.py         # FlowConfig (declarative use/by/obtain)
│   ├── data_manager.py # Trace-addressed shared state
│   └── pipeline.py     # Async pipeline engine with auto-termination
├── resources/          # Memsense-specific resource plugins
│   ├── locomo_reader   # Read LoCoMo JSON datasets
│   ├── ingest          # Write conversations to Memsense memory API
│   ├── qa              # Run QA against Memsense responses API
│   ├── judge           # LLM-as-judge grading
│   ├── filter          # Filter out error responses
│   └── summary         # Compute accuracy metrics & reports
└── cli.py              # Unified CLI entry point
```

### Key Concepts

| Concept       | Description |
|---------------|-------------|
| **Resource**  | Pluggable processing unit with `async process(*args) -> tuple` |
| **Flow**      | Declarative binding: which resource to use, what data to read (`by`), where to write (`obtain`) |
| **Trace**     | Dot-separated path into nested state dict (e.g. `benchmark.samples.0.grades`) |
| **DataManager** | Async-safe shared state store addressed by traces |
| **Pipeline**  | Engine that polls ready traces, matches flows, dispatches work, auto-stops when done |

### Wildcard Matching

Flows support wildcards `~0` through `~9` to match dynamic keys.
For example, `benchmark.samples.~0.qa_results` matches every sample
in the samples dict, and `~0` binds to the actual key (`"0"`, `"1"`, etc.).

## YAML Configuration

Three pre-built configs are provided:

| Config | Use Case |
|--------|----------|
| `configs/full_pipeline.yaml` | Complete: ingest → QA → filter → judge → summary |
| `configs/qa_judge.yaml` | QA + judge (data already ingested) |
| `configs/judge_only.yaml` | Judge only (QA results already on disk) |

### Overriding Values

Use `--set` to override any data key:

```bash
uv run python -m memsense_eval configs/full_pipeline.yaml \
    --set benchmark.token=sk-xxx \
    --set benchmark.data_path=./data/locomo10.json
```

Environment variables in YAML are resolved via `${VAR}` syntax.

## CLI Options

```
memsense_eval <config.yaml> [options]

positional arguments:
  config               Path to YAML pipeline config

options:
  --set KEY=VALUE      Override data keys (repeatable)
  --tick SECONDS       Pipeline tick interval (default: 1.0)
  --idle-limit N       Auto-stop after N idle ticks (default: 3)
  --serve              Run as FastAPI server with /data and /stop
  --port PORT          Server port (default: 8003)
  -v, --verbose        Debug logging
```

## Server Mode

For long-running evaluations, use server mode to monitor progress:

```bash
uv run python -m memsense_eval configs/full_pipeline.yaml --serve --port 8003

# In another terminal:
curl http://localhost:8003/data    # View current pipeline state
curl http://localhost:8003/stop    # Gracefully stop
```

## Data

Place LoCoMo dataset files in the `data/` directory:

- `locomo10_small.json` — 1 sample, 4 sessions (quick test)
- `locomo10_subset.json` — subset for development
- `locomo10.json` — full 10-sample dataset

## Output

Results are written to `output/`:

- `grades.<task>.json` — per-question grades with overall score
- `summary.<task>.json` — accuracy metrics, per-category breakdown, token usage
