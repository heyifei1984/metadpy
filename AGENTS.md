# Repository Guidelines

## Project Structure & Module Organization
- `metadpy/` contains the core library modules (e.g., `sdt.py`, `mle.py`, `bayesian.py`, `plotting.py`) and package metadata in `__init__.py`.
- `metadpy/models/` holds model definitions (e.g., subject-level PyMC model).
- `metadpy/datasets/` provides bundled sample data files.
- `metadpy/tests/` contains the pytest suite; tests follow `test_*.py`.
- `docs/` and `docs/source/` hold the documentation sources and tutorial notebooks.
- `requirements*.txt` and `environment.yml` capture runtime, test, and docs dependencies.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `python -m pip install -r requirements-tests.txt` installs test dependencies.
- `pytest` runs the full test suite; use `pytest metadpy/tests/test_sdt.py` for a focused run.
- `python -m pip install -r requirements-docs.txt` installs docs dependencies (for building docs locally).

## Coding Style & Naming Conventions
- Follow Black formatting defaults (e.g., 88-character lines) and isort import ordering; the project advertises both.
- Prefer explicit, descriptive function names mirroring existing modules (`dprime`, `roc_auc`, `metad`).
- Use snake_case for functions and variables; keep module names short and domain-specific.
- Static typing is encouraged; mypy is listed in badges, so keep annotations consistent where used.

## Testing Guidelines
- Tests are written with pytest and live in `metadpy/tests/`.
- Name new tests `test_<feature>.py` and new test functions `test_<behavior>()`.
- Include coverage for both MLE and Bayesian paths when changing shared utilities.
- Many tests assume scientific dependencies (NumPy/SciPy/Pandas/PyMC); ensure they are installed before running.

## Commit & Pull Request Guidelines
- Git history favors short, imperative summaries (e.g., “Update conf.py”, “add conda file (#12)”).
- Keep commit messages concise, optionally include issue/PR references in parentheses.
- PRs should describe the change, list tests run, and update docs/examples when behavior changes.
- Include screenshots or notebook output updates when modifying plots or tutorials.

## Security & Configuration Tips
- Avoid committing large datasets; prefer adding small fixtures under `metadpy/datasets/`.
- Keep dependency pins in `requirements*.txt` in sync with any new minimum versions you introduce.
