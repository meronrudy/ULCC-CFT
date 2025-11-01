"""
ULCC Core package.

Re-exports are kept when imported as a proper package (via the ulcc_core
symlink). When pytest imports this file directly as a test package module
without a parent package, skip relative imports to avoid ImportError.
"""

if __package__:
    # Re-export submodules when imported as a package
    from . import fisher, christoffel, dynamics, coords  # type: ignore[attr-defined]

__all__ = ["fisher", "christoffel", "dynamics", "coords"]
