"""
LGC — block-aware local mean on the SPD manifold P(n).

Enforces local geometric consistency on a sequence of SPD covariance matrices,
strictly within block boundaries to prevent leakage across class changes.

Two averaging modes are supported:
- mean="riemann" : Riemannian (Fréchet) mean — default, proposed method
- mean="euclid"  : Euclidean (arithmetic) mean — ablation baseline
"""

import numpy as np
from pyriemann.utils.mean import mean_riemann


def local_riemannian_mean_blockwise(
    X_cov: np.ndarray,
    block_ids: np.ndarray,
    half_window: int = 10,
    mean: str = "riemann",
) -> np.ndarray:
    """
    Enforce local geometric consistency strictly within block boundaries.

    Each covariance matrix is replaced by the local mean of its temporal
    neighbors that belong to the same block. Neighbors from adjacent blocks
    are excluded, preserving class boundaries.

    Parameters
    ----------
    X_cov : np.ndarray, shape (N, C, C)
        Sequence of SPD covariance matrices.
    block_ids : np.ndarray, shape (N,)
        Block ID for each matrix. LGC never crosses block boundaries.
    half_window : int
        Neighborhood radius in matrix-index space. The effective window
        size is at most 2*half_window+1, clipped at block edges.
        half_window=0 → no LGC (identity operation).
    mean : str
        Averaging method for the local neighborhood:
        "riemann" → Riemannian (Fréchet) mean on P(n) — proposed method.
        "euclid"  → Euclidean (arithmetic) mean — ablation baseline.

    Returns
    -------
    X_lgc : np.ndarray, shape (N, C, C)
        Block-aware LGC-processed covariance sequence.
    """
    if mean not in ("riemann", "euclid"):
        raise ValueError(f"mean must be 'riemann' or 'euclid', got '{mean}'")

    X_cov = np.asarray(X_cov)
    block_ids = np.asarray(block_ids)
    N = X_cov.shape[0]

    if half_window == 0:
        return X_cov.copy()

    X_lgc = np.empty_like(X_cov)

    for i in range(N):
        current_block = block_ids[i]
        start = max(0, i - half_window)
        end = min(N, i + half_window + 1)

        block_mask = block_ids[start:end] == current_block
        neighbors = X_cov[start:end][block_mask]

        if mean == "riemann":
            X_lgc[i] = mean_riemann(neighbors)
        else:
            X_lgc[i] = np.mean(neighbors, axis=0)

    return X_lgc
