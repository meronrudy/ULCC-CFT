import pytest
import numpy as np
from ulcc_core.fisher import fisher_metric_bernoulli, fisher_metric_inv_bernoulli
from ulcc_core.christoffel import gamma_theta
from ulcc_core.coords import theta_to_phi, phi_to_theta
from ulcc_core.dynamics import ngd_step, overdamped_second_order

def test_fisher_and_inverse():
    th = 0.3
    g = fisher_metric_bernoulli(th)
    ginv = fisher_metric_inv_bernoulli(th)
    assert np.isclose(g * ginv, 1.0)

def test_gamma_formula():
    th = 0.3
    gam = gamma_theta(th)
    expected = (2*th - 1) / (2 * th * (1-th))
    assert np.isclose(gam, expected)

def test_coord_roundtrip():
    th = 0.37
    phi = theta_to_phi(th)
    th2 = phi_to_theta(phi)
    assert np.isclose(th, th2, atol=1e-12)

def test_domain_guards_bernoulli():
    bad_thetas = [-1e-6, 0.0, 1.0, 1.0 + 1e-6]
    for th in bad_thetas:
        with pytest.raises(ValueError):
            fisher_metric_bernoulli(th)
        with pytest.raises(ValueError):
            fisher_metric_inv_bernoulli(th)
        with pytest.raises(ValueError):
            gamma_theta(th)
        with pytest.raises(ValueError):
            theta_to_phi(th)

def test_phi_geodesic_linearization():
    # Reparameterization invariance: linear path in phi should satisfy geodesic ODE in theta.
    th0, th1 = 0.2, 0.8
    phi0, phi1 = theta_to_phi(th0), theta_to_phi(th1)
    N = 1001  # dt = 1e-3 for precise central differences
    t = np.linspace(0.0, 1.0, N, dtype=float)
    phi = phi0 + t * (phi1 - phi0)
    theta = phi_to_theta(phi)  # vectorized inverse map
    dt = t[1] - t[0]

    # Central differences on interior points
    theta_prime = (theta[2:] - theta[:-2]) / (2.0 * dt)
    theta_second = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / (dt**2)
    theta_mid = theta[1:-1]
    Gamma_vals = np.array([gamma_theta(float(th)) for th in theta_mid], dtype=float)

    residual = theta_second + Gamma_vals * (theta_prime ** 2)
    max_abs = np.max(np.abs(residual))
    assert max_abs < 1e-5

def test_overdamped_vs_ngd():
    # Objective: f(theta) = 0.5 * (theta - theta_star)^2, grad f = theta - theta_star
    # Discretization alignment: NGD lr = dt / gamma
    theta_star = 0.7

    def grad(theta):
        return float(theta - theta_star)

    theta0 = 0.3  # strict interior start
    dt = 5e-4     # stable for gamma up to 1e3 (dt*gamma <= 0.5)
    steps = 8000  # total "time" T = steps * dt = 4.0
    gammas = [1e1, 1e2, 1e3]
    diffs = []

    for gamma in gammas:
        theta_od = theta0
        v = 0.0
        theta_ngd = theta0
        lr = dt / gamma

        for _ in range(steps):
            gL_od = grad(theta_od)
            theta_od, v = overdamped_second_order(theta_od, v, gL_od, gamma, dt)

            gL_ngd = grad(theta_ngd)
            theta_ngd = ngd_step(theta_ngd, gL_ngd, lr)

        diffs.append(abs(theta_od - theta_ngd))

    # Errors decrease with friction and meet tolerances
    assert diffs[1] < diffs[0] and diffs[2] < diffs[1]
    assert diffs[0] < 5e-3
    assert diffs[1] < 5e-4
    assert diffs[2] < 5e-5
