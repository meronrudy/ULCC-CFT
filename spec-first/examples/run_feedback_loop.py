#!/usr/bin/env python3
"""Deterministic end-to-end feedback loop demonstration.

Wires: PGGS (fast proposal) -> export (J,B) -> CFE residual/update (slow loop).
Logs metrics and writes CSV.

Runtime: < 1s. NumPy-only. Deterministic.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Ensure imports resolve when running as a script
THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from pggs.guide import CausalAtlas
from pggs.sampler import softmax_neg_potential
from pggs.export import export_results
from geom.curvature import project_information_tensor, einstein_operator_simple
from geom.cfe_update import cfe_residual, cfe_update_step_residual
from tests.fixtures.graphs import basis_vectors
from utils.logging import get_logger, log_metrics, csv_logger


def main() -> None:
    # 1) Paths and causal atlas potentials
    paths = list(range(6))
    atlas = CausalAtlas()
    for k in paths:
        atlas.update(k, 0.0 if k in (0, 1) else 1.0)

    # 2) Proposal weights
    q = softmax_neg_potential(paths, atlas, temperature=1.0)

    # 3) Map to samples in R^3 via basis vectors
    E = basis_vectors(3)  # [e0, e1, e2]
    samples = [E[i % 3] for i in paths]
    J, B = export_results(samples, q)

    # 4) Initial metric and information tensor
    g0 = np.eye(3, dtype=np.float64)
    I = project_information_tensor(B)

    # 5) Residual (use simple operator for guaranteed contraction)
    operator = einstein_operator_simple
    r0 = cfe_residual(g0, I, kappa=1.0, operator=operator)

    # 6) Update step with target_factor=0.1
    g1 = cfe_update_step_residual(
        g0, I, kappa=1.0, alpha=0.95, operator=operator, target_factor=0.1
    )

    # 7) Recompute residual and summarize
    r1 = cfe_residual(g1, I, kappa=1.0, operator=operator)
    norm_r0 = float(np.linalg.norm(r0, ord="fro"))
    norm_r1 = float(np.linalg.norm(r1, ord="fro"))
    reduction = float(norm_r1 / norm_r0) if norm_r0 > 0.0 else 0.0
    delta_g = float(np.linalg.norm(g1 - g0, ord="fro"))

    # Console print
    print(
        f"Feedback loop: ||r0||={norm_r0:.6g} ||r1||={norm_r1:.6g} "
        f"reduction={reduction:.6g} ||g1-g0||_F={delta_g:.6g}"
    )

    # 8) Structured logging and CSV
    logger = get_logger()
    log_metrics(
        {"norm_r0": norm_r0, "norm_r1": norm_r1, "reduction": reduction, "delta_g_fro": delta_g},
        step=1,
        logger=logger,
    )
    csv_path = os.path.join(THIS_DIR, "feedback_metrics.csv")
    write_csv = csv_logger(csv_path)
    write_csv({"norm_r0": norm_r0, "norm_r1": norm_r1, "reduction": reduction, "delta_g_fro": delta_g, "step": 1.0})

    sys.exit(0)


if __name__ == "__main__":
    main()
