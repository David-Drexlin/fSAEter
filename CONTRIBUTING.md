# Contributing

Thanks for taking a look at `fSAEter`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Local checks

```bash
ruff check .
pytest
```

## Scope guardrails

- Keep this repo focused on token extraction, local SAE training, `H` construction, and lightweight inspection.
- Do not add task-specific generator or augmentation stacks here.
- Keep backbone loading explicit and generic; avoid baking in lab-specific repo assumptions.
- Treat third-party repos as references, not runtime dependencies, unless a future design pass intentionally changes that.
