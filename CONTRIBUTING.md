# Contributing to Meridian

Thanks for contributing.

## Ground rules

- Keep changes focused and traceable to `meridian-spec.md` and/or `research.md`.
- Prefer small, reviewable pull requests.
- Add or update tests for behavioral changes.
- Avoid silent fallbacks; fail clearly with actionable errors.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Baseline checks:

```bash
cargo test --quiet
cargo build --quiet
PYTHONPATH=python python -m meridian_train.cli spec-check >/dev/null
PYTHONPATH=python python -m meridian_train.cli feature-vector-check >/dev/null
PYTHONPATH=python python -m meridian_train.cli data-plan --assets AAPL,SPY --output /tmp/meridian_data_plan.json >/dev/null
PYTHONPATH=python python -m unittest discover -s python/tests -q
```

## Pull request checklist

- [ ] Change is scoped and explained in PR description.
- [ ] Related spec/research sections are referenced.
- [ ] Tests and local checks pass.
- [ ] Backward compatibility impact is noted (if any).
- [ ] Security/privacy implications are documented (if any).

## Commit guidance

Use descriptive commits and include context in the message body when touching architecture or training defaults.
