# Data Directory

This directory is **not included in the repository** — raw EEG data is never committed.
Each dataset must be downloaded separately and placed here by the user.

---

## DEAP Dataset (required for `demos/02_demo_deap.py`)

**Full name**: DEAP: A Dataset for Emotion Analysis using EEG, Physiological and Video Signals
**Reference**: Koelstra et al. (2012). IEEE Transactions on Affective Computing, 3(1), 18–31.
**Access**: Request access at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

### Version required
Download the **preprocessed MATLAB version** (`data_preprocessed_matlab.zip`).
Do NOT use the Python/pickle version (`.dat` files) — the demo script reads `.mat` files.

### After downloading
Extract and place the files so the structure looks like:

```
data/
└── deap/
    ├── s01.mat
    ├── s02.mat
    ├── ...
    └── s32.mat
```

The script resolves `DEAP_PATH` automatically relative to its own location:
```python
DEAP_PATH = Path(__file__).parent.parent / "data" / "deap"
```
No manual path configuration is needed as long as the files are placed as shown above.

### Dataset structure (per subject)
Each `.mat` file contains:
- `data`   — shape `(40, 40, 8064)`: 40 trials × 40 channels × 8064 samples
  - First 32 channels are EEG — the script selects these automatically
  - Sampling rate: **128 Hz** (downsampled from original 512 Hz)
  - Trial length: 63 seconds (8064 / 128)
- `labels` — shape `(40, 4)`: continuous ratings on scale 1–9
  - Column 0: Valence   (1 = negative, 9 = positive)
  - Column 1: Arousal   (1 = calm/low, 9 = excited/high)
  - Column 2: Dominance
  - Column 3: Liking

### Binarization (applied in the script)
Ratings are binarized at threshold = 5.0:
- class 0 = below threshold
- class 1 = at or above threshold

---

## Team Metrics Dataset (required for `experiments/lgc_rct_loso.py`)

**Proprietary** — TNO, Netherlands. Not publicly available.
Contact: Anne-Marie Brouwer (TNO) for data access requests.

This dataset was used to produce the published results in:
> Licona Muñoz et al. — NAT 2026, Berlin.
