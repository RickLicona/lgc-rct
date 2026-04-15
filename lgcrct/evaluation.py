"""
LOSO evaluation — Leave-One-Subject-Out cross-subject protocol.

Standard evaluation for cross-subject EEG transfer learning.
Dataset-agnostic: expects (X, y, domains) already prepared by the user.

Usage
-----
    from lgcrct import run_loso

    # X : np.ndarray, shape (N, C, T) — bandpass-filtered EEG windows
    # y : np.ndarray, shape (N,)      — class labels
    # domains : np.ndarray, shape (N,) — subject IDs

    results = run_loso(X, y, domains, half_window=10, cov_estimator="lwf", lgc_mean="riemann")
    print(results[["target", "acc", "f1_macro"]].to_string(index=False))
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .pipeline import LGCRCTPipeline


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    half_window: int = 10,
    cov_estimator: str = "lwf",
    lgc_mean: str = "riemann",
) -> pd.DataFrame:
    """
    Leave-One-Subject-Out cross-subject evaluation.

    For each subject (target domain):
    - All remaining subjects are used as source.
    - Target samples are passed unlabeled — labels never reach the classifier.
    - Evaluated on all target samples (transductive, valid for RCT/LGC-RCT).

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        EEG windows — bandpass-filtered and sliding-window segmented.
        The user is responsible for preprocessing before calling this function.
    y : np.ndarray, shape (N,)
        Class labels.
    domains : np.ndarray, shape (N,)
        Subject/domain identifier for each window.
    half_window : int
        LGC neighborhood radius.
        half_window=0  → plain RCT (no LGC).
        half_window=10 → best configuration (Team Metrics, 34 pilots).
    cov_estimator : str
        Covariance estimator for pyriemann.estimation.Covariances.
        Default 'lwf' (Ledoit-Wolf) — validated on Team Metrics dataset.
    lgc_mean : str
        Averaging method for LGC smoothing.
        "riemann" → Riemannian (Fréchet) mean — proposed method.
        "euclid"  → Euclidean (arithmetic) mean — ablation baseline.

    Returns
    -------
    results : pd.DataFrame
        One row per subject with columns:
        target, n_tgt, acc, f1_macro, recall_macro, train_time_s, infer_time_s.
    """
    domains = np.asarray(domains, dtype=str)
    y = np.asarray(y)
    unique_domains = np.unique(domains)

    if half_window > 0:
        method = f"LGC-RCT K={half_window} ({lgc_mean})"
    else:
        method = "RCT"
    print(f"\nLOSO | {method} | cov_estimator={cov_estimator} | "
          f"subjects={len(unique_domains)}")

    rows = []

    for tgt_dom in unique_domains:
        print(f"\n{'='*52}")
        print(f"TARGET → {tgt_dom}")
        print("=" * 52)

        src_mask = domains != tgt_dom
        tgt_mask = domains == tgt_dom

        X_all = X
        y_all = y.astype(int)
        d_all = domains

        pipe = LGCRCTPipeline(
            half_window=half_window,
            cov_estimator=cov_estimator,
            lgc_mean=lgc_mean,
        )

        t0 = time.perf_counter()
        pipe.fit(X_all, y_all, d_all, target_domain=tgt_dom)
        train_time_s = time.perf_counter() - t0

        X_tgt = X[tgt_mask]
        y_tgt = y[tgt_mask].astype(int)
        d_tgt = domains[tgt_mask]

        t1 = time.perf_counter()
        y_hat = pipe.predict(X_tgt, y_tgt, d_tgt)
        infer_time_s = time.perf_counter() - t1

        acc = accuracy_score(y_tgt, y_hat)
        f1  = f1_score(y_tgt, y_hat, average="macro")
        rec = recall_score(y_tgt, y_hat, average="macro")

        print(f"ACC={acc:.3f}  F1={f1:.3f}  Recall={rec:.3f}")

        rows.append({
            "target":        tgt_dom,
            "n_tgt":         int(tgt_mask.sum()),
            "acc":           acc,
            "f1_macro":      f1,
            "recall_macro":  rec,
            "train_time_s":  train_time_s,
            "infer_time_s":  infer_time_s,
            "estimator":     "FgMDM",
            "cov_estimator": cov_estimator,
            "half_window":   half_window,
            "lgc_mean":      lgc_mean,
        })

    df = pd.DataFrame(rows)

    print(f"\n{'='*52}")
    print("GLOBAL SUMMARY")
    print(df[["target", "acc", "f1_macro", "recall_macro"]].to_string(index=False))

    metrics = ["acc", "f1_macro", "recall_macro"]
    mean = df[metrics].mean()
    std  = df[metrics].std()

    print("\n" + "-"*52)
    for m in metrics:
        print(f"  {m:<14} {mean[m]:.4f} ± {std[m]:.4f}")
    print("-"*52)
    print(f"  Mean train time : {df['train_time_s'].mean():.2f}s")
    print(f"  Mean infer time : {df['infer_time_s'].mean():.4f}s")

    return df
