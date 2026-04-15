"""
LGCRCTPipeline — Local Geometric Consistency + Riemannian Centering Transformation.

Unsupervised transductive cross-subject EEG transfer learning pipeline.

Pipeline steps
--------------
1. Covariance estimation: Covariances(estimator).transform(X) — windows to SPD matrices.
2. LGC (optional): block-aware local Riemannian mean, applied per domain.
   Block boundaries are inferred from label transitions in y — no explicit block IDs needed.
3. RCT alignment: TLCenter recenters each domain to the Riemannian identity.
4. FgMDM classifier trained on source domains only (target domain_weight=0).

Standard API — same signature as pyriemann transfer learning:
    X : np.ndarray, shape (N, C, T)   EEG windows, bandpass-filtered
    y : np.ndarray, shape (N,)        class labels
    domains : np.ndarray, shape (N,)  subject/domain IDs

Usage
-----
    from lgcrct import LGCRCTPipeline

    pipe = LGCRCTPipeline(half_window=10)                        # LGC-RCT
    pipe = LGCRCTPipeline(half_window=0)                         # plain RCT
    pipe = LGCRCTPipeline(half_window=10, cov_estimator="oas")   # custom estimator

    pipe.fit(X, y, domains, target_domain="subject_01")
    y_pred = pipe.predict(X_test, y_test, domains_test)
"""
from __future__ import annotations

import numpy as np
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.transfer import encode_domains, TLCenter, TLClassifier
from sklearn.pipeline import make_pipeline

from .lgc import local_riemannian_mean_blockwise


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------

def infer_blocks_from_labels(y: np.ndarray, domains: np.ndarray) -> np.ndarray:
    """
    Infer block IDs from label transitions within each domain.

    A new block starts whenever the class label changes within a domain.
    Each contiguous run of the same label = one block. Block IDs are
    globally unique across domains.

    This produces block IDs equivalent to explicit recording-block labels
    for datasets with alternating-class block structure (e.g. Team Metrics).

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Class labels.
    domains : np.ndarray, shape (N,)
        Domain identifier for each window.

    Returns
    -------
    block_ids : np.ndarray, shape (N,), dtype int
        Block ID for each window.

    Example
    -------
    y       = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    domains = ['s1'] * 6 + ['s2'] * 6
    → blocks = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    """
    y = np.asarray(y)
    domains = np.asarray(domains, dtype=str)
    block_ids = np.zeros(len(y), dtype=int)
    block_counter = 0

    for dom in np.unique(domains):
        mask = np.where(domains == dom)[0]
        y_dom = y[mask]
        for i, global_idx in enumerate(mask):
            if i == 0 or y_dom[i] != y_dom[i - 1]:
                block_counter += 1
            block_ids[global_idx] = block_counter

    return block_ids


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LGCRCTPipeline:
    """
    Unsupervised transductive cross-subject transfer learning via LGC-RCT.

    Parameters
    ----------
    half_window : int
        LGC neighborhood radius (in window-index space).
        half_window=0  → plain RCT (no LGC).
        half_window=10 → best-performing configuration (Team Metrics, 34 pilots).
    cov_estimator : str
        Covariance estimator passed to pyriemann.estimation.Covariances.
        Default 'lwf' (Ledoit-Wolf) — best for this project.
        Other options: 'scm', 'oas', 'mcd', 'corr', etc.
        See pyriemann.utils.covariance.covariances for the full list.
    metric : dict
        FgMDM metric configuration. Defaults match the published results.
    """

    def __init__(
        self,
        half_window: int = 10,
        cov_estimator: str = "lwf",
        lgc_mean: str = "riemann",
        metric: dict | None = None,
    ):
        self.half_window = half_window
        self.cov_estimator = cov_estimator
        self.lgc_mean = lgc_mean
        self.metric = metric or {
            "mean": "logeuclid",
            "distance": "riemann",
            "map": "logeuclid",
        }
        self._target_domain: str | None = None
        self._pipe = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domains: np.ndarray,
        target_domain: str,
    ) -> "LGCRCTPipeline":
        """
        Fit the pipeline on source + target EEG windows.

        The proposed pipeline requires no labeled data from the target subject.
        Domain alignment is performed via RCT (Zanini et al., 2018): each domain
        (source and target) is recentered to the identity matrix using its own
        Riemannian mean. For the target domain, this mean is computed from all
        available unlabeled covariance matrices. This constitutes a transductive
        transfer learning setting (Pan & Yang, 2010, Def. 3). Crucially, the
        pipeline is fully unsupervised with respect to the target domain: no
        target labels are used at any stage.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)
            EEG windows for all domains. C channels, T time samples.
        y : np.ndarray, shape (N,)
            Class labels. Used only to infer LGC segment boundaries.
            Target labels are never used for adaptation or classification.
        domains : np.ndarray, shape (N,)
            Domain identifier for each window (e.g. 'subject_01').
        target_domain : str
            Domain ID of the held-out subject (e.g. 'subject_01').
        """
        domains = np.asarray(domains, dtype=str)
        unique_domains = np.unique(domains)
        self._target_domain = target_domain

        # Step 1: windows (N, C, T) → SPD covariance matrices (N, C, C)
        X_cov = self._estimate_cov(X)

        # Step 2: LGC (local geometric consistency) per domain — block boundaries from y
        if self.half_window > 0:
            blocks = infer_blocks_from_labels(y, domains)
            X_cov = self._apply_lgc(X_cov, domains, blocks, self.lgc_mean)

        # Step 3: RCT + FgMDM — target weight=0 (no target labels used)
        dom_weights = {d: 1.0 for d in unique_domains}
        dom_weights[target_domain] = 0.0

        X_enc, y_enc = encode_domains(X_cov, y, domains)

        self._pipe = make_pipeline(
            TLCenter(target_domain=target_domain),
            TLClassifier(
                target_domain=target_domain,
                estimator=FgMDM(metric=self.metric, n_jobs=1),
                domain_weight=dom_weights,
            ),
        )
        self._pipe.fit(X_enc, y_enc)
        return self

    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        """
        Predict class labels for EEG windows.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)
            EEG windows.
        y : np.ndarray, shape (N,)
            Class labels — used only to infer LGC block boundaries.
            Not used by the classifier.
        domains : np.ndarray, shape (N,)

        Returns
        -------
        y_pred : np.ndarray, shape (N,)
        """
        self._check_fitted()
        X_cov = self._estimate_cov(X)
        if self.half_window > 0:
            blocks = infer_blocks_from_labels(y, domains)
            X_cov = self._apply_lgc(X_cov, domains, blocks, self.lgc_mean)
        return self._pipe.predict(X_cov).astype(int)

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate covariances, apply LGC + RCT alignment.

        Returns aligned SPD matrices — useful for inspection or custom classifiers.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)
        y : np.ndarray, shape (N,)
            Used only for LGC block boundary inference.
        domains : np.ndarray, shape (N,)

        Returns
        -------
        X_aligned : np.ndarray, shape (N, C, C)
        """
        self._check_fitted()
        X_cov = self._estimate_cov(X)
        if self.half_window > 0:
            blocks = infer_blocks_from_labels(y, domains)
            X_cov = self._apply_lgc(X_cov, domains, blocks, self.lgc_mean)
        return self._pipe[0].transform(X_cov)

    @property
    def use_lgc(self) -> bool:
        return self.half_window > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_cov(self, X: np.ndarray) -> np.ndarray:
        """Windows (N, C, T) → SPD covariance matrices (N, C, C)."""
        return Covariances(estimator=self.cov_estimator).transform(X)

    def _apply_lgc(
        self,
        X_cov: np.ndarray,
        domains: np.ndarray,
        blocks: np.ndarray,
        lgc_mean: str = "riemann",
    ) -> np.ndarray:
        """Apply LGC (local geometric consistency) per domain."""
        X_out = np.empty_like(X_cov)
        for dom in np.unique(domains):
            mask = domains == dom
            X_out[mask] = local_riemannian_mean_blockwise(
                X_cov[mask], blocks[mask], self.half_window, mean=lgc_mean
            )
        return X_out

    def _check_fitted(self):
        if self._pipe is None:
            raise RuntimeError("Call fit() before predict() or transform().")
