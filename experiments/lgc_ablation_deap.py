"""
LGC ablation experiment — DEAP dataset.

Compares LGC-Riemannian vs LGC-Euclidean across K values for valence and arousal.

Results table for IEEE MetroXRaine paper:

    K    | LGC-Riemann (valence) | LGC-Euclid (valence) | ...
    0    | RCT baseline          | —
    1    | ...                   | ...
    3    | ...                   | ...
    5    | ...                   | ...
    10   | ...                   | ...

Usage
-----
    python experiments/lgc_ablation_deap.py [--deap-path /path/to/deap]

Output
------
    results/deap_ablation/
        LGC-RCT_K{k}_{mean}_DEAP_{task}_15-36Hz_LOSO.csv  — per-config
        deap_ablation_summary.csv                           — combined table
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt

from lgcrct import run_loso

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BAND          = (15, 36)   # Hz
WINDOW_SEC    = 4
STEP_SEC      = 2
FS            = 128        # Hz
COV_ESTIMATOR = "lwf"
LABEL_IDX     = {"valence": 0, "arousal": 1}
LABEL_THRESHOLD = 5.0

# Ablation grid — (half_window, lgc_mean)
# LGC-Riemannian K=1,3,5,10 already exist from previous runs.
# Only missing: RCT baseline (K=0) and all LGC-Euclidean configs.
CONFIGS = [
    (0,  "riemann"),   # RCT baseline (K=0 — lgc_mean irrelevant, no LGC applied)
    (1,  "euclid"),
    (3,  "euclid"),
    (5,  "euclid"),
    (10, "euclid"),
]

TASKS = ["valence", "arousal"]

# ---------------------------------------------------------------------------
# DEAP preprocessing (self-contained — same logic as demos/02_demo_deap.py)
# ---------------------------------------------------------------------------

def load_deap_subject(filepath: Path) -> tuple:
    mat = scipy.io.loadmat(str(filepath))
    X = np.array(mat["data"])[:, :32, :]   # (40, 32, 8064) — EEG only
    y_raw = np.array(mat["labels"])        # (40, 4)
    return X, y_raw


def binarize_labels(ratings: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    return (ratings >= threshold).astype(np.int64)


def bandpass_filter(signal: np.ndarray, lo: float, hi: float, fs: int = 128) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal, axis=1)


def extract_windows(signal: np.ndarray, label: int,
                    window_sec: int = 4, step_sec: int = 2,
                    fs: int = 128) -> tuple:
    C, T = signal.shape
    win_len  = window_sec * fs
    step_len = step_sec * fs
    windows  = []
    for start in range(0, T - win_len + 1, step_len):
        windows.append(signal[:, start:start + win_len])
    X_windows = np.stack(windows).astype(np.float32)
    y_windows = np.full(len(windows), label, dtype=np.int64)
    return X_windows, y_windows


def build_deap_dataset(deap_path: Path, label_target: str) -> tuple:
    lo, hi   = BAND
    col_idx  = LABEL_IDX[label_target]
    X_list, y_list, dom_list = [], [], []

    mat_files = sorted(deap_path.glob("s*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {deap_path}")

    print(f"\nLoading DEAP | task={label_target} | band={BAND} Hz | {len(mat_files)} subjects")
    for filepath in mat_files:
        subj_id = int(filepath.stem[1:])
        dom_str = f"subject_{subj_id:02d}"

        X_subj, y_raw = load_deap_subject(filepath)
        y_binary = binarize_labels(y_raw[:, col_idx], threshold=LABEL_THRESHOLD)

        for trial_idx in range(X_subj.shape[0]):
            signal_filt = bandpass_filter(X_subj[trial_idx], lo, hi, fs=FS)
            X_win, y_win = extract_windows(
                signal_filt, y_binary[trial_idx],
                window_sec=WINDOW_SEC, step_sec=STEP_SEC, fs=FS,
            )
            X_list.append(X_win)
            y_list.append(y_win)
            dom_list.append(np.full(len(y_win), dom_str, dtype=object))

    X       = np.concatenate(X_list)
    y       = np.concatenate(y_list)
    domains = np.concatenate(dom_list)

    print(f"Total windows: {X.shape[0]}  shape: {X.shape}  "
          f"class balance: {y.mean():.2f}")
    return X, y, domains


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(deap_path: Path, results_dir: Path, skip_existing: bool = True) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for task in TASKS:
        # Load DEAP once per task (shared across all K configs)
        X, y, domains = build_deap_dataset(deap_path, task)

        for half_window, lgc_mean in CONFIGS:
            label = "RCT" if half_window == 0 else f"LGC-{lgc_mean}-K{half_window}"
            csv_name = f"{label}_DEAP_{task.upper()}_{BAND[0]}-{BAND[1]}Hz_LOSO.csv"
            csv_path = results_dir / csv_name

            if skip_existing and csv_path.exists():
                print(f"\n[SKIP] {csv_name} already exists — loading from disk")
                df = pd.read_csv(csv_path)
            else:
                print(f"\n{'='*60}")
                print(f"CONFIG: task={task}  K={half_window}  mean={lgc_mean}")
                print("=" * 60)
                t0 = time.perf_counter()
                df = run_loso(
                    X, y, domains,
                    half_window=half_window,
                    cov_estimator=COV_ESTIMATOR,
                    lgc_mean=lgc_mean,
                )
                elapsed = time.perf_counter() - t0
                print(f"\nTotal wall time: {elapsed/60:.1f} min")

                df["task"]       = task
                df["half_window"] = half_window
                df["lgc_mean"]   = lgc_mean if half_window > 0 else "—"
                df["band_lo"]    = BAND[0]
                df["band_hi"]    = BAND[1]
                df.to_csv(csv_path, index=False)
                print(f"Saved → {csv_path}")

            summary_rows.append({
                "task":       task,
                "method":     label,
                "half_window": half_window,
                "lgc_mean":   lgc_mean if half_window > 0 else "—",
                "acc_mean":   df["acc"].mean(),
                "acc_std":    df["acc"].std(),
                "f1_mean":    df["f1_macro"].mean(),
                "f1_std":     df["f1_macro"].std(),
            })

    # Combined summary table
    summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / "deap_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(summary[["task", "method", "acc_mean", "acc_std"]].to_string(index=False))
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="LGC ablation — DEAP dataset")
    parser.add_argument(
        "--deap-path",
        type=Path,
        default=repo_root / "data" / "deap",
        help="Folder containing s01.mat ... s32.mat",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results" / "deap_ablation",
        help="Output directory for per-config CSVs and summary",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run all configs even if CSV already exists",
    )
    args = parser.parse_args()

    main(
        deap_path=args.deap_path,
        results_dir=args.results_dir,
        skip_existing=not args.no_skip,
    )
