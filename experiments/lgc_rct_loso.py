"""
LGC-RCT LOSO experiment — Team Metrics dataset (private, TNO).

Requires
--------
- Team Metrics CSV (dataset_clean_34_pilots.csv) — proprietary, contact TNO.
- gumpy — EEG signal processing library used for bandpass filtering.
  Not available on PyPI. Install from source:
  https://github.com/gumpy-bci/gumpy
  IMPORTANT: gumpy defaults to fs=256 Hz internally. This script always
  passes fs=128 explicitly to match the Team Metrics sampling rate.

Usage
-----
    # LGC-RCT (proposed method, K=10):
    python lgc_rct_loso.py --data-path /path/to/dataset_clean_34_pilots.csv \\
                            --band alpha --use-lgc --lgc-half-window 10

    # Plain RCT baseline:
    python lgc_rct_loso.py --data-path /path/to/dataset_clean_34_pilots.csv \\
                            --band alpha
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from lgcrct import run_loso

# ── EEG channel montage (32-channel) ──────────────────────────────────────────
CHANNELS_32 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5",
    "P7", "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8",
    "CP6", "CP2", "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2",
    "Fz", "Cz",
]

# ── Frequency band definitions (Hz) ───────────────────────────────────────────
BAND_FREQ = {
    "theta":       (4,  7),
    "alpha":       (8, 13),
    "theta-alpha": (4, 13),
}


# =============================================================================
# EEG PREPROCESSING  (Team Metrics dataset format)
# =============================================================================

def extract_overlapping_windows(
    df,
    channels,
    window_sec=4,
    step_sec=2,
    sampling_rate=128,
    lo=4,
    hi=7,
    return_blocks=False,
):
    """
    Extract bandpass-filtered overlapping windows from one subject's EEG.

    Parameters
    ----------
    df : pd.DataFrame
        Single-subject slice with columns: block, class, <channel names>.
    channels : list[str]
    window_sec, step_sec : int
    sampling_rate : int
        Must be 128 Hz for Team Metrics. Do NOT use 256 (gumpy default).
    lo, hi : float
        Bandpass cutoff frequencies in Hz.
    return_blocks : bool
        If True, also return block IDs per window (required for LGC).

    Returns
    -------
    X_windows : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    block_ids : np.ndarray, shape (N,)  — only when return_blocks=True
    """
    import gumpy

    df_columns = df.columns.tolist()
    data_np = df.to_numpy()

    idx_block = df_columns.index("block")
    idx_class = df_columns.index("class")
    idx_channels = [df_columns.index(c) for c in channels]

    window_len = window_sec * sampling_rate
    step_len = step_sec * sampling_rate

    X_list, y_list, block_list = [], [], []

    for block_id in sorted(df["block"].unique()):
        block_data = data_np[data_np[:, idx_block] == block_id]
        class_label = int(block_data[0, idx_class])
        channel_data = block_data[:, idx_channels].astype(float)

        channel_data = gumpy.signal.butter_bandpass(
            channel_data,
            lo=lo,
            hi=hi,
            axis=0,
            fs=sampling_rate,   # explicit — gumpy defaults to fs=256 (wrong)
        )

        num_samples = channel_data.shape[0]
        for start in range(0, num_samples - window_len + 1, step_len):
            X_list.append(channel_data[start:start + window_len, :])
            y_list.append(class_label)
            block_list.append(block_id)

    # shape: (N, C, T) — channels first, required by pyriemann
    X_windows = np.stack(X_list).astype(np.float32).transpose(0, 2, 1)
    y = np.array(y_list, dtype=np.int64)

    if return_blocks:
        return X_windows, y, np.array(block_list, dtype=int)
    return X_windows, y


def build_subject_features(df_subj, channels, band, window_sec, step_sec):
    """
    Windowing + bandpass for one subject. Returns epochs (N, C, T).
    Covariance estimation and LGC are handled by LGCRCTPipeline.

    Returns
    -------
    X_epochs : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    """
    lo, hi = BAND_FREQ[band]
    X_epochs, y = extract_overlapping_windows(
        df_subj, channels,
        window_sec=window_sec, step_sec=step_sec,
        sampling_rate=128,
        lo=lo, hi=hi,
        return_blocks=False,
    )
    return X_epochs, y


def build_dataset(df, channels, band, window_sec, step_sec):
    """
    Extract epochs for all subjects.

    Returns
    -------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    domains : np.ndarray, shape (N,)   — 'subject_XX' strings
    """
    X_list, y_list, dom_list = [], [], []

    for subj_id, df_subj in df.groupby("subject"):
        dom_str = f"subject_{int(subj_id):02d}"
        X_epochs, y = build_subject_features(
            df_subj, channels, band, window_sec, step_sec,
        )
        X_list.append(X_epochs)
        y_list.append(y)
        dom_list.append(np.full(len(y), dom_str, dtype=object))

    return (
        np.concatenate(X_list),
        np.concatenate(y_list),
        np.concatenate(dom_list),
    )


# =============================================================================
# CLI
# =============================================================================

def _build_results_path(args) -> str:
    method = f"LGC-RCT_K{args.lgc_half_window}" if args.use_lgc else "RCT"
    fname = f"{method}_{args.band.upper()}_LOSO-34pilots-FgMDM_4s_32ch.csv"
    return os.path.join(args.results_dir, fname)


def main():
    parser = argparse.ArgumentParser(description="LGC-RCT LOSO experiment — Team Metrics dataset")
    parser.add_argument(
        "--data-path", required=True,
        help="Path to dataset_clean_34_pilots.csv",
    )
    parser.add_argument(
        "--band", default="alpha",
        choices=["theta", "alpha", "theta-alpha"],
        help="Frequency band (default: alpha).",
    )
    parser.add_argument(
        "--use-lgc", action="store_true", default=False,
        help="Apply LGC (local geometric consistency). Omit for plain RCT.",
    )
    parser.add_argument(
        "--lgc-half-window", type=int, default=10,
        help="LGC neighborhood radius (default: 10). Only used with --use-lgc.",
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="Directory for output CSV (default: ./results).",
    )
    args = parser.parse_args()

    # ── Team Metrics preprocessing ────────────────────────────────────────────
    df = pd.read_csv(args.data_path)
    X, y, domains = build_dataset(
        df, CHANNELS_32, args.band,
        window_sec=4, step_sec=2,
    )

    # ── LGC-RCT LOSO (lgcrct package) ────────────────────────────────────────
    half_window = args.lgc_half_window if args.use_lgc else 0
    df_results = run_loso(X, y, domains, half_window=half_window, cov_estimator="lwf")

    # ── Save results (dataset-specific metadata) ──────────────────────────────
    df_results["band"] = args.band
    df_results["use_lgc"] = args.use_lgc
    df_results["half_window"] = half_window

    results_path = _build_results_path(args)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved → {results_path}")


if __name__ == "__main__":
    main()
