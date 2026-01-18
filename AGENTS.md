# AGENTS.md

## Mission (what you are building)
This repository will add a hierarchical Bayesian *regression* extension for metacognitive efficiency (meta-d'/d' a.k.a. Mratio) to the existing meta-d'/HMeta-d-style modeling code.

Primary deliverable:
- A new public function implementing RHMeta-d-style regression on subject-level metacognitive efficiency (typically modeled on log(Mratio)), with MCMC sampling via the repo’s existing Bayesian backend.

## Non-goals (do NOT do these unless explicitly instructed)
- Do not rewrite existing meta-d' or hmeta-d implementations.
- Do not change existing public APIs unless the project plan explicitly calls for it.
- Do not introduce new probabilistic-programming frameworks (stick to what the repo already uses).
- Do not add heavy new dependencies without a written justification and approval.

## Required reading before coding
1) Read `PROJECT_PLAN.md` (milestones + acceptance criteria).
2) If present, read `docs/model_spec_rhmetad.md` (math + priors + outputs).
3) Read the existing Bayesian model implementation in this repo (e.g., the current `hmetad` function and utilities).

If `docs/model_spec_rhmetad.md` does not exist yet, create it as the first PR (see PROJECT_PLAN).

## Working style (how to operate)
- Start every task by producing a short plan with:
  - Files you will touch
  - Tests you will add/update
  - How you will validate correctness (simulation/recovery)
- Work in small, reviewable commits and PR-sized chunks.
- Prefer minimal diffs over large refactors.

## Scientific correctness rules (hard constraints)
- Any modeling choice that affects inference must be documented in `docs/model_spec_rhmetad.md`.
- If you cannot verify a detail from a reliable source (paper, official toolbox, or existing repo code), label it clearly as **UNVERIFIED** in the spec and open a tracking issue.
- Add simulation-based tests:
  - Parameter recovery for regression coefficients
  - Posterior predictive sanity checks (at least one)
- Ensure covariates align correctly to subject IDs (no silent re-ordering).

## Repo hygiene and safety
- Do not run destructive shell commands.
- Do not modify files outside the repo workspace.
- Do not fetch random external scripts.
- Prefer deterministic commands and pinned dependencies where possible.

## Lint / format / tests
Use the repo’s existing tooling. If unsure:
- Find and follow existing CI workflows and dev instructions in README.
- Run the full test suite before finishing a PR.
- Run formatting/lint steps that the repo expects (e.g., pre-commit, ruff/black, etc.), if configured.

## Review guidelines (used for @codex review)
When reviewing changes, prioritize:
1) Statistical/model correctness (likelihood, priors, parameterization)
2) Tests (especially recovery tests)
3) API stability and clear docstrings
4) Performance only after correctness is established

Treat the following as P0 issues:
- Incorrect mapping between trials → counts → likelihood
- Wrong handling of confidence rating bins
- Covariate misalignment to subjects
- Regression implemented “post-hoc” instead of inside the hierarchical model (unless explicitly intended)

## PR checklist (must pass)
- [ ] New/updated tests added and passing
- [ ] Model spec updated (or created)
- [ ] Clear docstring + example usage added for new public API
- [ ] No unnecessary refactors
- [ ] Outputs include regression posteriors and Mratio/logMratio posteriors




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
