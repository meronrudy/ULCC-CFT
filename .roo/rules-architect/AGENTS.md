# AGENTS.md

This file provides guidance to agents when working with code in this repository.

Project Architecture Rules (Non-Obvious Only):
- Dual namespace scheme: import packages use underscores (ulcc_core, ulcc_ddg, ulcc_pggs) while source dirs are hyphenated; symlink bridge is required at repo root and in Docker.
- Identity stubs are architectural placeholders relied upon by tests and higher layers (transport/holonomy); keep them inert (see [ulcc_ddg/transport.py](ulcc-ddg/transport.py:5), [ulcc_ddg/holonomy.py](ulcc_ddg/holonomy.py:7)).
- Deterministic execution is a cross-cutting constraint; RNG must be threaded through PGGS algorithms to keep results reproducible.
- Control loop stages and pack schema are phase-gated; preserve staged protocol ordering ([control/gcu.py](control/gcu.py:82), :102, :138, :165, :184, :217) and permissive schema checks ([control/gcu.py](control/gcu.py:295), :307).