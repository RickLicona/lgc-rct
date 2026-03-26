"""
LGC — block-aware local Riemannian mean on the SPD manifold P(n).

Enforces local geometric consistency on a sequence of SPD covariance matrices,
strictly within block boundaries to prevent leakage across class changes.
"""

import numpy as np
from pyriemann.utils.mean import mean_riemann


def local_riemannian_mean_blockwise(
    X_cov: np.ndarray,
    block_ids: np.ndarray,
    half_window: int = 10,
) -> np.ndarray:
    """
    Enforce local geometric consistency strictly within block boundaries.

    Each covariance matrix is replaced by the local Riemannian (Fréchet) mean
    of its temporal neighbors that belong to the same block. Neighbors from
    adjacent blocks are excluded, preserving class boundaries.

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

    Returns
    -------
    X_lgc : np.ndarray, shape (N, C, C)
        Block-aware LGC-processed covariance sequence.
    """
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

        X_lgc[i] = mean_riemann(neighbors)

    return X_lgc
