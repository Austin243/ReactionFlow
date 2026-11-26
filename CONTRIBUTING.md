# Contributing to ReactionFlow

Focused fixes, tests, and documentation improvements are welcome. Open an issue before adding a
built-in model backend or changing the campaign format, bond detection, or pathway refinement.

## Development

ReactionFlow needs Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python -m build
```

CI runs these checks on every pull request, with `pytest` on Python 3.12 and 3.14. Its tests must
not need a GPU, model weights, network access, or a cluster, so the tests that load a real MACE or
AIMNet2 checkpoint are skipped unless the variables named in those test files are set.

The tests that run real EON need the `eon` extra, and EON runs on Linux x86_64. They are skipped
without `REACTIONFLOW_TEST_EON=1`, and a separate workflow runs them on every pull request:

```bash
python -m pip install -e '.[dev,eon]'
REACTIONFLOW_TEST_EON=1 pytest -q tests/test_eon_integration.py tests/test_eon_example.py
```

Keep pull requests focused. Include tests for changed behavior, and update the docs when a command,
a setting, or an output changes. Say in the pull request when a change alters results or stops an
existing trajectory from resuming. Never commit credentials, model weights, or unpublished
scientific inputs without authorization.

By contributing, you agree that your work is licensed under the repository's
[BSD 3-Clause License](LICENSE) and that you have the right to submit it.
