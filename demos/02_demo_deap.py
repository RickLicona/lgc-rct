# %% [markdown]
# # LGC-RCT Demo — DEAP Dataset
# Cross-subject EEG transfer learning for affective state classification.
#
# **Dataset**: DEAP — A Dataset for Emotion Analysis using EEG, Physiological and Video Signals
# Koelstra et al. (2012). IEEE Transactions on Affective Computing.
# https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
#
# **Method**: LGC-RCT (Local Geometric Consistency + Riemannian Centering Transformation)
# Licona Muñoz et al. — NAT 2026, Berlin.
#
# **Protocol**: Leave-One-Subject-Out (LOSO), 32 subjects, binary classification.
# Each subject: 40 trials × ~30 windows/trial (4s, 50% overlap) = ~1200 epochs.

# %% [markdown]
# ## 1. Imports & Setup

# %%
import numpy as np
import scipy.io
from pathlib import Path

from lgcrct import run_loso

# %% [markdown]
# ## 2. Configuration

# %%
# Path to DEAP .mat files folder (contains s01.mat ... s32.mat)
# See data/FORMAT.md for download instructions.
DEAP_PATH = Path(__file__).parent.parent / "data" / "deap"

# Classification target: "valence" or "arousal"
LABEL_TARGET = "valence"

# Frequency band — 15-36 Hz
BAND = (15, 36)   # Hz (lo, hi)

# LGC-RCT configuration
HALF_WINDOW    = 10
COV_ESTIMATOR  = "lwf"

# Windowing — 4s windows, 50% overlap, consistent with Team Metrics
WINDOW_SEC     = 4
STEP_SEC       = 2
SAMPLING_RATE  = 128   # Hz

# Binarization threshold for DEAP continuous ratings (scale 1-9)
LABEL_THRESHOLD = 5.0

# Label column index in mat['labels'] — shape (40, 4): [valence, arousal, dominance, liking]
LABEL_IDX = {"valence": 0, "arousal": 1}

# %% [markdown]
# ## 3. DEAP Preprocessing

# %%
def load_deap_subject(filepath: str) -> tuple:
    """
    Load one DEAP subject .mat file.

    Returns
    -------
    X : np.ndarray, shape (40, 32, 8064)  — EEG only (first 32 channels)
    y_raw : np.ndarray, shape (40, 4)     — continuous ratings [valence, arousal, dominance, liking]
    """
    mat = scipy.io.loadmat(filepath)
    X = np.array(mat["data"])[:, :32, :]   # (40, 32, 8064) — EEG channels only
    y_raw = np.array(mat["labels"])        # (40, 4)
    return X, y_raw


def binarize_labels(ratings: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Binarize continuous DEAP ratings at threshold.

    class 0 = below threshold (negative / low arousal)
    class 1 = at or above threshold (positive / high arousal)

    Parameters
    ----------
    ratings : np.ndarray, shape (40,) — continuous ratings on scale 1-9
    threshold : float — default 5.0 (midpoint of 1-9 scale)
    """
    return (ratings >= threshold).astype(np.int64)


def bandpass_filter(signal: np.ndarray, lo: float, hi: float,
                    fs: int = 128) -> np.ndarray:
    """
    Butterworth bandpass filter along the time axis.

    Parameters
    ----------
    signal : np.ndarray, shape (C, T)
        Single trial EEG. C=channels, T=time samples.
        Filtering is applied along axis=1 (T axis).
    lo, hi : float — cutoff frequencies in Hz
    fs : int — sampling rate in Hz

    Returns
    -------
    filtered : np.ndarray, shape (C, T)
    """
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal, axis=1)   # axis=1 → time axis of (C, T)


def extract_windows(signal: np.ndarray, label: int,
                    window_sec: int = 4, step_sec: int = 2,
                    fs: int = 128) -> tuple:
    """
    Sliding window segmentation of one trial.

    Parameters
    ----------
    signal : np.ndarray, shape (C, T) — one filtered trial
    label : int — class label for this trial (all windows share the same label)
    window_sec, step_sec : int
    fs : int

    Returns
    -------
    X_windows : np.ndarray, shape (n_windows, C, T_win)
    y_windows : np.ndarray, shape (n_windows,)
    """
    C, T = signal.shape
    win_len = window_sec * fs    # 4 * 128 = 512 samples
    step_len = step_sec * fs     # 2 * 128 = 256 samples (50% overlap)

    windows = []
    for start in range(0, T - win_len + 1, step_len):
        windows.append(signal[:, start:start + win_len])

    X_windows = np.stack(windows).astype(np.float32)   # (n_windows, C, T_win)
    y_windows = np.full(len(windows), label, dtype=np.int64)
    return X_windows, y_windows


def build_deap_dataset(deap_path: str, label_target: str,
                       band: tuple, window_sec: int = 4,
                       step_sec: int = 2) -> tuple:
    """
    Load and preprocess all DEAP subjects.

    Parameters
    ----------
    deap_path : str — folder containing s01.mat ... s32.mat
    label_target : str — "valence" or "arousal"
    band : tuple — (lo, hi) in Hz
    window_sec, step_sec : int

    Returns
    -------
    X : np.ndarray, shape (N, C, T)    — bandpass-filtered EEG epochs
    y : np.ndarray, shape (N,)         — binary class labels
    domains : np.ndarray, shape (N,)   — subject IDs ('subject_01', ...)
    """
    lo, hi = band
    col_idx = LABEL_IDX[label_target]

    X_list, y_list, dom_list = [], [], []

    mat_files = sorted(Path(deap_path).glob("s*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {deap_path}")

    for filepath in mat_files:
        subj_id = int(filepath.stem[1:])   # s01 → 1
        dom_str = f"subject_{subj_id:02d}"

        X_subj, y_raw = load_deap_subject(str(filepath))
        # X_subj : (40, 32, 8064)
        # y_raw  : (40, 4)

        y_binary = binarize_labels(y_raw[:, col_idx], threshold=LABEL_THRESHOLD)

        for trial_idx in range(X_subj.shape[0]):
            signal = X_subj[trial_idx]             # (32, 8064) — (C, T)
            signal_filt = bandpass_filter(signal, lo, hi, fs=SAMPLING_RATE)

            X_win, y_win = extract_windows(
                signal_filt, y_binary[trial_idx],
                window_sec=window_sec, step_sec=step_sec, fs=SAMPLING_RATE,
            )
            X_list.append(X_win)
            y_list.append(y_win)
            dom_list.append(np.full(len(y_win), dom_str, dtype=object))

        print(f"{dom_str} | trials: {X_subj.shape[0]} | "
              f"windows/trial: {len(X_win)} | "
              f"class balance: {y_binary.mean():.2f}")

    X       = np.concatenate(X_list)
    y       = np.concatenate(y_list)
    domains = np.concatenate(dom_list)

    print(f"\nTotal epochs : {X.shape[0]}")
    print(f"Shape        : {X.shape}  (N, C, T)")
    print(f"Class counts : {np.bincount(y.astype(int))}")

    return X, y, domains


# %% [markdown]
# ## 4. Load Dataset

# %%
print(f"Loading DEAP | label={LABEL_TARGET} | band={BAND} Hz")
X, y, domains = build_deap_dataset(
    DEAP_PATH,
    label_target=LABEL_TARGET,
    band=BAND,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
)

# %% [markdown]
# ## 5. Run LGC-RCT LOSO

# %%
results = run_loso(X, y, domains, half_window=HALF_WINDOW, cov_estimator=COV_ESTIMATOR)

# %% [markdown]
# ## 6. Results Summary

# %%
print(f"\nLGC-RCT K={HALF_WINDOW} | DEAP {LABEL_TARGET} | band={BAND} Hz")
print(f"Mean ACC    : {results['acc'].mean():.4f} ± {results['acc'].std():.4f}")
print(f"Mean F1     : {results['f1_macro'].mean():.4f} ± {results['f1_macro'].std():.4f}")
print(f"Mean Recall : {results['recall_macro'].mean():.4f} ± {results['recall_macro'].std():.4f}")

# %%
results["label_target"] = LABEL_TARGET
results["band_lo"]      = BAND[0]
results["band_hi"]      = BAND[1]

results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)
results.to_csv(
    results_dir / f"LGC-RCT_K{HALF_WINDOW}_DEAP_{LABEL_TARGET.upper()}_{BAND[0]}-{BAND[1]}Hz_LOSO.csv",
    index=False,
)
print("Results saved.")
