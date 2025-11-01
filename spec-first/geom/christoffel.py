"""Levi-Civita connection (Christoffel symbols).

Invariant: Levi-Civita, torsion-free, metric-compatible; coordinate-covariant.
TODO: Add explicit metric-compatibility checks in future tests.
"""

from __future__ import annotations

import numpy as np


def christoffel(g: np.ndarray, dg: np.ndarray) -> np.ndarray:
    """
    Compute the Christoffel symbols Γ^a_{bc} of the Levi-Civita connection
    from a metric tensor and its first partial derivatives.

    Shapes and conventions
    - g has shape (n, n), symmetric positive-definite (SPD).
    - dg has shape (n, n, n) with dg[k, i, j] = ∂_k g_{ij}.
    - Return Γ with shape (n, n, n) where Γ[a, b, c] = Γ^a_{bc}.

    Formula
    Γ^a_{bc} = 0.5 * g^{aδ} * ( ∂_b g_{δc} + ∂_c g_{δb} − ∂_δ g_{bc} )

    Parameters
    ----------
    g : np.ndarray
        Metric tensor (SPD), shape (n, n).
    dg : np.ndarray
        Partial derivatives of g; dg[k, i, j] = ∂_k g_{ij}, shape (n, n, n).

    Returns
    -------
    np.ndarray
        Christoffel symbols Γ with shape (n, n, n), ordered as Γ[a, b, c] = Γ^a_{bc}.

    Raises
    ------
    ValueError
        If shapes are inconsistent, g is not symmetric or not positive-definite.
    """
    g = np.asarray(g, dtype=float)
    dg = np.asarray(dg, dtype=float)

    # Validate metric shape
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("g must be a square 2D array with shape (n, n).")
    n = g.shape[0]

    # Validate dg shape
    if dg.shape != (n, n, n):
        raise ValueError(f"dg must have shape (n, n, n) matching g; got {dg.shape} for n={n}.")

    # Symmetry check
    if not np.allclose(g, g.T, atol=1e-12):
        raise ValueError("g must be symmetric within tolerance.")

    # Positive-definiteness check (use symmetric part for numerical stability)
    g_sym = 0.5 * (g + g.T)
    w = np.linalg.eigvalsh(g_sym)
    if not np.all(w > 0.0):
        raise ValueError("g must be positive-definite (all eigenvalues > 0).")

    # Inverse metric
    g_inv = np.linalg.inv(g_sym)

    # Allocate Christoffel symbols
    Gamma = np.zeros((n, n, n), dtype=float)

    # Compute Γ[:, b, c] for each (b, c)
    # S_δbc = ∂_b g_{δc} + ∂_c g_{δb} − ∂_δ g_{bc}
    for b in range(n):
        for c in range(n):
            S = dg[b, :, c] + dg[c, :, b] - dg[:, b, c]  # shape (n,)
            Gamma[:, b, c] = 0.5 * (g_inv @ S)

    return Gamma


def discrete_metric_derivative(
    g_nodes: np.ndarray,
    coords: np.ndarray,
    edges: np.ndarray,
    *,
    weight: str = "inv_length",
    ridge: float = 1e-12,
) -> np.ndarray:
    """
    Discrete metric derivative on a graph via weighted least squares with Tikhonov regularization.

    Goal
    ----
    Estimate the coordinate derivatives ∂_k g_{ab} at each graph node from
    metric samples g at nodes and coordinate differences along incident edges.

    Shapes and conventions
    ----------------------
    - g_nodes: array (N, d, d). For each node i in {0..N-1}, g_nodes[i] is the
      symmetric positive-definite (SPD) metric tensor g_{ab} at that node.
    - coords:  array (N, p). Coordinates (chart) at each node. We require p == d.
    - edges:   array (M, 2) of integer node index pairs (undirected graph edges).
    - Output:  dg_nodes with shape (N, d, d, d), holding partials
      dg_nodes[i, k, a, b] = ∂_k g_{ab} at node i.
      This matches the ordering expected by christoffel(): for a single node
      with metric g (d,d) and dg (d,d,d), christoffel(g, dg) is valid.

    Method (per node i)
    -------------------
    For each incident edge (i, j), define:
      Δx = coords[j] - coords[i]    (shape (d,))
      Δg_{ab} = g[j, a, b] - g[i, a, b]
    We fit, for each pair (a,b), the local linear model:
      Δg_{ab} ≈ Δx · β_{ab}
    where β_{ab}[k] ≈ ∂_k g_{ab} at node i.

    We solve a weighted least-squares with Tikhonov regularization:
      minimize ||W (X β - y)||_2
      with X = stack(Δx)^T ∈ R^{K×d}, y = stack(Δg_{ab}) ∈ R^{K},
      W = diag(weights), and ridge >= 0.
    Normal equations:
      (Xᵀ W² X + ridge I) β = Xᵀ W² y

    We share X across all (a,b) and solve a multi-target system, so β is
    computed for all (a,b) at once. If the system is ill-conditioned or
    underdetermined (degree < d), the ridge usually suffices; on failure we
    fall back to a pseudoinverse.

    We finally enforce symmetry in (a,b) by averaging:
      ∂_k g_{ab} ← 0.5 (∂_k g_{ab} + ∂_k g_{ba})

    Parameters
    ----------
    g_nodes : np.ndarray, shape (N, d, d)
        SPD metric at each node (finite, symmetric within 1e-12, eigenvalues > 0).
    coords : np.ndarray, shape (N, p)
        Coordinates per node; must satisfy p == d.
    edges : np.ndarray, shape (M, 2)
        Undirected edges as integer index pairs (0 ≤ idx < N).
    weight : {"none","inv_length"}, default "inv_length"
        If "inv_length", each edge contributes with weight 1/||Δx||.
        If "none", uniform weights are used.
    ridge : float, default 1e-12
        Nonnegative Tikhonov regularization strength added to (Xᵀ W² X).
        Helps stabilize solves when degrees are small or geometry is coarse.

    Returns
    -------
    np.ndarray
        dg_nodes with shape (N, d, d, d), ordered as dg_nodes[i, k, a, b] = ∂_k g_{ab}.

    Raises
    ------
    ValueError
        - Shapes inconsistent (e.g., g_nodes not (N,d,d), coords not (N,p), edges not (M,2)).
        - coords dimension p != metric dimension d, or d < 1.
        - Non-finite entries in g_nodes or coords.
        - g_nodes not symmetric within 1e-12 or not SPD at some node.
        - edges contain invalid indices.
        - A node has zero incident neighbors (cannot estimate derivatives).

    Numerical caveats
    -----------------
    - Boundary nodes or nodes with degree < d: least-squares is underdetermined;
      ridge and/or pseudoinverse provide a stable estimate but with higher bias.
    - Coarse graphs reduce accuracy; choose tolerance in tests accordingly.
    - Duplicate coordinates or extremely short edges are down-weighted by
      "inv_length" (capped by a small epsilon).

    Relation to DDG spirit
    ----------------------
    This is a finite-difference-on-edges approach consistent with discrete differential
    geometry intuition: node-wise local linear models inferred from edge differences.

    """
    g_nodes = np.asarray(g_nodes, dtype=float)
    coords = np.asarray(coords, dtype=float)
    edges = np.asarray(edges)

    # Basic shape checks
    if g_nodes.ndim != 3 or g_nodes.shape[1] != g_nodes.shape[2]:
        raise ValueError("g_nodes must have shape (N, d, d) with square per-node metrics.")
    N, d, d2 = g_nodes.shape
    if d != d2:
        raise ValueError("g_nodes' last two dimensions must be equal (square metric).")
    if d < 1:
        raise ValueError("Metric dimension d must be >= 1.")

    if coords.ndim != 2 or coords.shape[0] != N:
        raise ValueError("coords must have shape (N, p) with the same N as g_nodes.")
    p = coords.shape[1]
    if p != d:
        raise ValueError(f"Coordinate dimension p must equal metric dimension d; got p={p}, d={d}.")

    # Finite checks
    if not np.all(np.isfinite(g_nodes)):
        raise ValueError("g_nodes must be finite.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must be finite.")

    # Symmetry check within tolerance and SPD check per node
    if not np.allclose(g_nodes, np.swapaxes(g_nodes, -1, -2), atol=1e-12):
        raise ValueError("g_nodes must be symmetric within tolerance 1e-12.")

    # Use symmetric part for SPD test robustness
    g_nodes_sym = 0.5 * (g_nodes + np.swapaxes(g_nodes, -1, -2))
    for i in range(N):
        w = np.linalg.eigvalsh(g_nodes_sym[i])
        if not np.all(w > 0.0):
            raise ValueError(f"g_nodes must be SPD at all nodes; node {i} has non-positive eigenvalue(s).")

    # Edges validation
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (M, 2) of integer node index pairs.")
    if edges.size > 0:
        if not np.issubdtype(edges.dtype, np.integer):
            # allow float arrays that are integer-like, but force/validate safely
            if np.allclose(edges, np.round(edges)):
                edges = np.round(edges).astype(int)
            else:
                raise ValueError("edges must contain integer indices.")
        if np.min(edges) < 0 or np.max(edges) >= N:
            raise ValueError("edges contain out-of-range node indices.")
    # Build adjacency
    neighbors = [[] for _ in range(N)]
    for e0, e1 in edges:
        neighbors[int(e0)].append(int(e1))
        neighbors[int(e1)].append(int(e0))

    if ridge < 0:
        raise ValueError("ridge must be nonnegative.")

    # Prepare output: dg[i, k, a, b] = ∂_k g_{ab} at node i
    dg = np.zeros((N, d, d, d), dtype=float)

    # Small epsilon to prevent division by zero in 1/||Δx||
    eps = 1e-12

    for i in range(N):
        nbrs = neighbors[i]
        K = len(nbrs)
        if K == 0:
            raise ValueError(f"Node {i} has zero incident edges; cannot estimate derivatives.")

        X = coords[np.asarray(nbrs, dtype=int), :] - coords[i]          # (K, d)
        Y = g_nodes[np.asarray(nbrs, dtype=int), :, :] - g_nodes[i]     # (K, d, d)

        # Weights per edge
        if weight == "inv_length":
            lens = np.linalg.norm(X, axis=1)                            # (K,)
            w = 1.0 / np.maximum(lens, eps)
        elif weight == "none":
            w = np.ones((K,), dtype=float)
        else:
            raise ValueError('weight must be one of {"none", "inv_length"}.')

        # Weighted LS with Tikhonov
        Xw = X * w[:, None]                                             # (K, d)
        Yw = (Y.reshape(K, d * d)) * w[:, None]                         # (K, d*d)

        # Normal equations: (Xwᵀ Xw + ridge I) B = Xwᵀ Yw
        A = Xw.T @ Xw
        if ridge > 0.0:
            A = A + ridge * np.eye(d, dtype=float)

        try:
            B = np.linalg.solve(A, Xw.T @ Yw)                           # (d, d*d)
        except np.linalg.LinAlgError:
            # Under-determined or ill-conditioned; pseudo-inverse fallback
            B = np.linalg.pinv(Xw) @ Yw                                 # (d, d*d)

        dg_i = B.reshape(d, d, d)                                       # (k, a, b)
        dg[i] = dg_i

    # Enforce symmetry in (a,b)
    dg = 0.5 * (dg + np.swapaxes(dg, 2, 3))

    return dg
