"""Deterministic formal test verifying approximate energy conservation for metric-aware wave with J=0."""

import os
import sys
import numpy as np

# Ensure 'spec-first/' is importable so 'from field.wave import ...' works.
THIS_DIR = os.path.dirname(__file__)
SPEC_FIRST_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if SPEC_FIRST_ROOT not in sys.path:
    sys.path.insert(0, SPEC_FIRST_ROOT)

from field.wave import wave_leapfrog_step, wave_energy


def test_wave_energy_conservation_flat_no_source() -> None:
    n = 16
    g = np.eye(n, dtype=float)
    J = np.zeros(n, dtype=float)
    dt = 1e-3
    steps = 2000

    # Deterministic initial conditions without RNG
    phi = np.linspace(-1.0, 1.0, n, dtype=float)
    pi_half = np.zeros(n, dtype=float)

    E0 = wave_energy(phi, pi_half, g)
    assert np.isfinite(E0)
    assert E0 >= -1e-12

    for _ in range(steps):
        phi, pi_half = wave_leapfrog_step(phi, pi_half, J, g, kappaC=1.0, dt=dt)

    E_final = wave_energy(phi, pi_half, g)
    assert np.isfinite(E_final)
    assert E_final >= -1e-12

    rel_drift = float(abs(E_final - E0) / max(1e-12, abs(E0)))
    assert rel_drift <= 1e-3 + 1e-12