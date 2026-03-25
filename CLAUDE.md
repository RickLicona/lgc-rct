# CLAUDE.md — LGC-RCT GitHub Repository Project

## What This Project Is

Open-source Python package implementing **LGC-RCT** (Local Geometric Consistency +
Riemannian Centering Transformation) — a calibration-free cross-subject EEG transfer
learning method for passive Brain-Computer Interfaces.

This repository accompanies the conference paper:
> *"Local Geometric Consistency for Cross-Subject EEG Transfer Learning"*
> Ricardo Licona Muñoz et al. — NAT 2026, Berlin.

The parent research project (full experimental pipeline) lives at:
`/Users/ricardolicona/PycharmProjects/AirForceproject/`

---

## Repository Structure

```
lgc-rct/
├── lgcrct/                          # Installable Python package (core contribution)
│   ├── smoothing.py                 # LGC: block-aware Riemannian moving average
│   └── pipeline.py                 # LGCRCTPipeline: fit / predict / transform
├── demos/
│   └── 02_demo_deap.py
├── experiments/                     # Requires Team Metrics dataset (private, TNO)
│   ├── lgc_rct_loso.py
│   ├── rct_baseline.py
│   ├── mdm_baseline.py
│   └── launcher.py
├── data/
│   └── FORMAT.md
├── results/
│   └── loso_34pilots_published.csv
├── docs/internal/                   # NOT pushed to GitHub (.gitignore)
│   ├── TL_methodology.md            # PyRiemann conventions, key findings — READ FIRST
│   ├── dataset_structure.md         # Dataset details, windowing, fs bug
│   ├── code_fixes.md                # Bugs fixed and why
│   └── scripts_guide.md            # Original experimental pipeline reference
└── CLAUDE.md                        # This file
```

---

## Technical Stack

- Python 3.11
- PyRiemann >= 0.7 — Riemannian geometry, TLCenter, TLClassifier, MDWM, FgMDM
- scikit-learn — pipeline, metrics
- NumPy, SciPy, pandas
- MNE-Python — EEG preprocessing (in parent project)

---

## The LGC-RCT Method

### Pipeline
```
EEG -> band-pass filter -> sliding windows (4s, 50% overlap, fs=128Hz)
    -> covariance estimation (Ledoit-Wolf, pyriemann Covariances('lwf'))
    -> LGC smoothing (block-aware Riemannian moving average, K neighbors)
    -> RCT alignment (TLCenter: recenters each domain to identity)
    -> FgMDM classifier (source domains only - target unlabeled)
```

### LGC Smoothing — Critical Implementation Details
- Riemannian moving average over covariance sequences using geodesic interpolation
- Block-aware: smoothing must NOT cross block boundaries (class changes at block edges)
- half_window=10 (K=10) is the best-performing configuration on Team Metrics
- K=0 -> plain RCT (no smoothing), K=1 -> minimal smoothing
- Smoothing is applied BEFORE RCT alignment

### Key Algorithm Property
LGC captures temporal geometric structure in covariance sequences — local Riemannian
consistency among neighboring windows. This is a property of signal temporal organization,
NOT of spectral content. Hence LGC-RCT is robust to frequency band selection.

---

## Published Results — NAT26 Berlin (Team Metrics, 34 pilots, LOSO)
Methods reported: MDM, RCT, LGC-RCT. MDWM was NOT included in NAT26 (ICCAS France only).

| Method        | Alpha ACC      | Theta ACC      | Calib-free? |
|---------------|:--------------:|:--------------:|:-----------:|
| MDM           | 50.56 ± 1.63   | 51.34 ± 2.15   |      —      |
| RCT           | 57.74 ± 2.56   | 57.94 ± 3.46   |     YES     |
| LGC-RCT K=1  | 63.11 ± 4.04   | 60.61 ± 3.42   |     YES     |
| LGC-RCT K=10 | 77.73 ± 6.39   | 73.91 ± 6.05   |     YES     |

These are the exact numbers from the paper text. Do NOT change them.

### ICCAS France — MDWM situation (dt=0.6 inductive, confirmed correct)
WARNING: previous numbers (~62-71%) from ChatGPT were WRONG.
MDWM dt=0.6 inductive gives ~51-55% across all bands/calib — barely above MDM.
Ablation dt in {0.1, 0.2, 0.3} currently running to find optimal configuration.

| Method         | Alpha         | Theta         | Theta-Alpha   | Calib-free? |
|----------------|:-------------:|:-------------:|:-------------:|:-----------:|
| MDM            | 52.17 +- 3.39 | 50.50 +- 1.53 | 50.70 +- 1.89 |      -      |
| RCT            | 60.01 +- 3.82 | 57.86 +- 2.54 | 60.03 +- 3.31 |     YES     |
| MDWM dt=0.6 6c | 54.99 +- 6.58 | 51.64 +- 4.49 | 53.67 +- 6.09 |     NO      |
| MDWM published | 69.20 +- 4.58 | 70.19 +- 4.69 | 67.55 +- 5.61 | INVALID (leakage) |

---

## DEAP Dataset Results (32 subjects, LOSO, 15-36 Hz, lgc-rct package)

### LGC-RCT K=10, cov_estimator='lwf' (Ledoit-Wolf covariance estimator)

| Task         | ACC           | Calib-free? | Status      |
|--------------|:-------------:|:-----------:|-------------|
| Valence      | 79.52 ± 5.87  |     YES     | preliminary |
| Arousal      | 75.51 ± 5.68  |     YES     | preliminary |

Key finding: LGC-RCT K=10 generalizes across paradigms with same configuration (K=10, cov_estimator='lwf' (Ledoit-Wolf), 15-36 Hz):
- Workload (Team Metrics): 77.73% ± 6.39
- Valence (DEAP):          79.52% ± 5.87
- Arousal (DEAP):          75.51% ± 5.68

Three tasks, two datasets, two cognitive domains, same method — journal paper material.

---

## Critical Scientific Context

### Frequency Band Robustness (key finding for journal paper)
All Riemannian methods are robust to frequency band selection in this passive BCI task.
Validated empirically: results statistically indistinguishable across alpha, theta,
theta-alpha bands — even when wrong frequencies were used (fs bug).

Why: spatial covariance encodes cross-channel synchrony structure associated with
workload that exists across multiple frequency ranges simultaneously. Contrasts with
active BCI (motor imagery) where band filtering is critical (mu/beta concentration).

This is a positive scientific finding, not a limitation. Frame as:
"LGC-RCT demonstrates robust performance across frequency configurations, suggesting
it captures temporal covariance structure that transcends specific spectral content —
particularly relevant for passive BCI where no gold-standard frequency band exists."

### MDWM is NOT Calibration-Free (corrected understanding)
- ICCAS France paper described MDWM as "calibration-free" — this is incorrect
- PyRiemann domain_tradeoff=1.0 uses labeled target class means (opposite of Kalunga paper)
- Correct result: MDWM needs 6 labeled calibration blocks to achieve ~70%
- Only RCT (and LGC-RCT) are genuinely calibration-free in the ICCAS France paper
- See docs/internal/TL_methodology.md for full analysis

### PyRiemann Convention (OPPOSITE to Kalunga 2018)
- domain_tradeoff=0.0 -> source class means only (= MDM, ~51%)
- domain_tradeoff=1.0 -> target class means only (needs labeled target)
- Higher domain_tradeoff = MORE target influence (opposite of Kalunga paper)

---

## Dataset — Team Metrics (PRIVATE)

- 34 pilots, UAV supervision task, binary workload classification
- 8 blocks per pilot, alternating classes 0,1,0,1,0,1,0,1
- 149 windows/block x 8 = 1192 windows/pilot
- Proprietary (TNO, Netherlands) — NOT shareable
- Contact: Anne-Marie Brouwer (TNO) for access

What IS shareable:
- lgcrct/ package code (public)
- Pre-computed accuracy metrics CSV (public)
- Possibly covariance matrices .npz — verify with TNO first

---

## Input Format for lgcrct/ Package

X       : np.ndarray  (n_trials, n_channels, n_channels)  # SPD covariance matrices
y       : np.ndarray  (n_trials,)                         # labels {0, 1}
domains : np.ndarray  (n_trials,)                         # subject IDs (str or int)
blocks  : np.ndarray  (n_trials,)                         # block ID per trial (int)

Full specification: data/FORMAT.md

---

## Release Strategy

| Version | When             | Content                                                    |
|---------|------------------|------------------------------------------------------------|
| v0.1.0  | Before Berlin    | lgcrct/ package, synthetic notebook, FORMAT.md, results CSV |
| v0.2.0  | After Berlin     | MOABB demo notebook, possibly covariance matrices .npz     |
| v1.0.0  | With journal sub | Full ablation, second dataset, statistical tests           |

Generate Zenodo DOI at v0.1.0 for formal software citation.

---

## Instructions for AI Assistants

- Read docs/internal/TL_methodology.md before implementing any TL method
- The parent project at /Users/ricardolicona/PycharmProjects/AirForceproject/
  contains the full experimental pipeline (97 experimental runs, launcher.py, etc.)
- This repo is the PUBLIC-FACING package — keep code clean, minimal, well-documented
- The lgcrct/ package must be dataset-agnostic: no hardcoded paths, no Team Metrics assumptions
- Do not add dependencies beyond: pyriemann, numpy, scipy, scikit-learn
- All public code must run without access to the Team Metrics dataset
- Notebooks must work with synthetic data at minimum
- Preserve reproducibility: random_state=42 where applicable
- Do not push docs/internal/ to GitHub — it is in .gitignore

## Context Files (docs/internal/ — NOT on GitHub)
Read these for full methodology context before any implementation:
- TL_methodology.md   — PyRiemann conventions, transductive vs inductive, frequency robustness
- dataset_structure.md — Team Metrics structure, windowing, critical fs=128 rule
- code_fixes.md        — gumpy fs bug and other fixes (Fix 18 is critical)
- scripts_guide.md     — original experimental pipeline reference
