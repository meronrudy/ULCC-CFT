"""Bernoulli: compare geodesic parameterization vs NGD step mapping."""
import numpy as np
from ulcc_core.coords import theta_to_phi, phi_to_theta

def main():
    thetas = np.linspace(0.1, 0.9, 5)
    phis = [theta_to_phi(t) for t in thetas]
    recon = [phi_to_theta(p) for p in phis]
    print("theta → phi → theta roundtrip (first/last):", thetas[0], recon[0], "|", thetas[-1], recon[-1])

if __name__ == "__main__":
    main()
