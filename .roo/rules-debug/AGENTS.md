# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Project Debug Rules (Non-Obvious Only):
- If imports fail or pytest can’t locate packages, verify symlinks in repo root and Docker image.
- ValueError on boundary θ indicates domain guards working as intended (see [ulcc-core/coords.py](ulcc-core/coords.py:4), [ulcc-core/fisher.py](ulcc-core/fisher.py:4)).
- Holonomy “zero” is expected with identity transport ([holonomy_loop()](ulcc_ddg/holonomy.py:7)); don’t “fix” to nonzero.
- Determinism bugs: ensure rng is passed and seeded via np.random.default_rng; missing threading will cause flaky PGGS tests.
- Docker repros must include RUN ln -s … lines; otherwise unit tests fail with ModuleNotFoundError.
- Test harness injects repo root into sys.path ([tests/conftest.py](tests/conftest.py:1)); absence suggests environment misconfig.