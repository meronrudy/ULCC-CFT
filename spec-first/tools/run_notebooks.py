#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Headless rendering for Matplotlib: must be set before any matplotlib import by notebooks
os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
try:
    from nbclient import NotebookClient, CellExecutionError  # type: ignore
except Exception:
    # Fallback for nbclient where CellExecutionError is under nbclient.exceptions
    from nbclient import NotebookClient  # type: ignore
    from nbclient.exceptions import CellExecutionError  # type: ignore

ARTIFACTS_DIR = Path("spec-first/artifacts")
FIGURES_DIR = ARTIFACTS_DIR / "figures"

DEFAULT_NOTEBOOKS = [
    Path("spec-first/notebooks/bernoulli.ipynb"),
    Path("spec-first/notebooks/gaussian.ipynb"),
    Path("spec-first/notebooks/ricci_step.ipynb"),
    Path("spec-first/notebooks/pggs_demo.ipynb"),
]

def parse_nb_list() -> List[Path]:
    """Parse NB_LIST env var into a list of notebook Paths, else return default order."""
    env = os.getenv("NB_LIST")
    if env:
        items = [s.strip() for s in env.split(",") if s.strip()]
        paths = [Path(p) for p in items]
    else:
        paths = list(DEFAULT_NOTEBOOKS)
    # Validate paths exist
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"Notebook(s) not found: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)
    return paths

def execute_one(path: Path, timeout: int) -> Tuple[float, nbformat.NotebookNode]:
    """Execute a single notebook at 'path' with the given timeout; returns (elapsed_sec, executed_nb)."""
    start = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    # Execute with CWD as the notebook's directory for relative paths
    cwd_prev = Path.cwd()
    try:
        os.chdir(path.parent)
        client.execute()
    except CellExecutionError as e:
        # Re-raise with concise context
        raise e
    finally:
        os.chdir(cwd_prev)
    elapsed = time.perf_counter() - start
    return elapsed, nb

def write_executed_copy(nb: nbformat.NotebookNode, src_path: Path) -> Path:
    """Write executed copy under artifacts as {stem}-executed.ipynb; returns destination path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{src_path.stem}-executed.ipynb"
    nbformat.write(nb, out_path)
    return out_path

def main() -> None:
    """Execute notebooks serially, fail-fast, summarizing status."""
    # Ensure artifacts/figures directories exist before running
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        timeout = int(os.getenv("NB_TIMEOUT", "600"))
    except ValueError:
        print("NB_TIMEOUT must be an integer number of seconds", file=sys.stderr)
        raise SystemExit(1)

    notebooks = parse_nb_list()
    print(f"Executing {len(notebooks)} notebook(s) with timeout {timeout}s each (MPLBACKEND={os.environ.get('MPLBACKEND')})")

    for nb_path in notebooks:
        print(f"START {nb_path}")
        try:
            elapsed, executed_nb = execute_one(nb_path, timeout)
            dest = write_executed_copy(executed_nb, nb_path)
            print(f"OK    {nb_path} in {elapsed:.1f}s &#45;> {dest}")
        except CellExecutionError as e:
            # Print a concise first line and exit
            msg = str(e).splitlines()
            snippet = msg[0] if msg else repr(e)
            print(f"FAIL  {nb_path}: {snippet}", file=sys.stderr)
            raise SystemExit(1)
        except Exception as e:
            print(f"FAIL  {nb_path}: {e}", file=sys.stderr)
            raise SystemExit(1)

    print("All notebooks executed successfully.")

if __name__ == "__main__":
    main()