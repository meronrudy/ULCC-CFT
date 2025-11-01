"""PGGS algebra module.

Invariants
- Noncommutativity captured where required
- Invariant: AB ≠ BA unless commutative.
"""

from typing import Any, Tuple
import numpy as np


class Operator:
    """
    Linear operator over R^n represented as a 2D ndarray (float64).

    Invariant: AB ≠ BA unless commutative.
    """

    def __init__(self, M: Any):
        """
        Initialize an Operator from any 2D numpy-like input.

        Parameters
        ----------
        M : array-like
            2D matrix-like object.

        Raises
        ------
        ValueError
            If M is not 2D.
        """
        A = np.asarray(M, dtype=np.float64)
        if A.ndim != 2:
            raise ValueError("Operator requires a 2D matrix")
        self.M: np.ndarray = A

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the matrix shape."""
        return self.M.shape

    def __matmul__(self, other: "Operator") -> "Operator":
        """
        Matrix composition using '@'.

        Returns
        -------
        Operator
            The composed operator self @ other.

        Raises
        ------
        TypeError
            If other is not an Operator.
        ValueError
            If shapes are incompatible for matrix multiplication.
        """
        if not isinstance(other, Operator):
            raise TypeError("Right operand must be an Operator")
        a, b = self.M, other.M
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: {a.shape} @ {b.shape}")
        return Operator(a @ b)

    def commutator(self, other: "Operator") -> "Operator":
        """
        Compute the commutator [A,B] = AB - BA.

        Returns
        -------
        Operator
            The commutator operator.

        Raises
        ------
        TypeError
            If other is not an Operator.
        ValueError
            If shapes are incompatible for matrix multiplication.
        """
        if not isinstance(other, Operator):
            raise TypeError("Other must be an Operator")
        a, b = self.M, other.M
        if a.shape[1] != b.shape[0] or b.shape[1] != a.shape[0]:
            raise ValueError(f"Incompatible shapes for commutator: {a.shape} and {b.shape}")
        return Operator(a @ b - b @ a)

    def __repr__(self) -> str:
        return f"Operator(shape={self.M.shape})"


__all__ = ["Operator"]
