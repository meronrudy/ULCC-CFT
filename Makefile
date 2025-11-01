# ULCC symlink rule (see AGENTS.md)
.PHONY: links test expts all

links:
	@set -e; \
	if [ ! -L ulcc_core ]; then ln -s ulcc-core ulcc_core; fi; \
	if [ ! -L ulcc_ddg ]; then ln -s ulcc-ddg ulcc_ddg; fi; \
	if [ ! -L ulcc_pggs ]; then ln -s pggs-toy ulcc_pggs; fi

# Ensure test depends on links
test: links
	pytest -q

# Ensure experiments also depend on links
expts: links
	python ulcc-expts/bernoulli_vs_ngd.py

# Optional aggregate
all: test

# Launch GUI (requires Tk); depends on links for import-name symlinks
.PHONY: gui
gui: links
	python -m harness.gui.app

# Spec-first tests and notebooks
spec-first-test:
	pytest -q spec-first/tests/formal

spec-first-notebooks:
	python3 spec-first/tools/run_notebooks.py

spec-first-all: spec-first-test spec-first-notebooks

.PHONY: spec-first-artifacts spec-first-archive spec-first-notebooks
spec-first-artifacts:
	@mkdir -p spec-first/artifacts
	@echo "Running formal tests (JUnit XML) ..."
	pytest -q spec-first/tests/formal --junitxml spec-first/artifacts/test-results.xml
	@echo "Running feedback loop example ..."
	python3 spec-first/examples/run_feedback_loop.py > spec-first/artifacts/feedback_loop.txt
	@echo "Copying feedback metrics CSV if present ..."
	@test -f spec-first/examples/feedback_metrics.csv && cp -f spec-first/examples/feedback_metrics.csv spec-first/artifacts/ || true
	@echo "Writing environment manifest ..."
	python3 -c 'import json,platform,pytest; print(json.dumps({"python": platform.python_version(), "platform": platform.platform(), "pytest": getattr(pytest, "__version__", "unknown")}, indent=2))' > spec-first/artifacts/env.json
	@echo "spec-first artifacts generated in spec-first/artifacts"

spec-first-archive: spec-first-artifacts
	@tar -czf spec-first/artifacts/spec-first-artifacts.tar.gz -C spec-first artifacts
	@echo "Created archive spec-first/artifacts/spec-first-artifacts.tar.gz"

spec-first-all: spec-first-artifacts spec-first-archive
