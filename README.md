# LGC-RCT — Local Geometric Consistency + Riemannian Centering Transformation

Unsupervised transductive cross-subject EEG transfer learning for passive Brain-Computer Interfaces.

> **Licona Muñoz, R., Guetschel, P., Bruin, J., Yilmaz, E. & Brouwer, A.-M.** — *"Local Geometric Consistency for Cross-Subject EEG Transfer Learning"*
> NAT 2026, Berlin. *(citation forthcoming)*

---

## What is LGC-RCT?

LGC-RCT is an unsupervised transductive method for cross-subject EEG transfer learning
designed for passive BCI applications. It combines two components:

- **RCT (Riemannian Centering Transformation)** [[1]](#references): a transfer learning method
  that re-centers covariance representations across domains to the identity matrix.
  Domains are treated as independent prior to the re-centering and classification phases.

- **LGC (Local Geometric Consistency)** *(proposed)*: a local-geometry phase applied prior
  to RCT. For each covariance matrix, a neighborhood including its immediate temporal
  neighbors on both sides is defined. A local Riemannian mean is computed over this
  neighborhood, and the matrix is replaced by this local estimate. The parameter K
  controls the degree of local geometric consistency enforced between neighboring
  window-level covariance matrices.

In this work, we investigate whether enforcing local geometric consistency among
neighboring covariance matrices prior to RCT improves cross-subject workload classification.

### Pipeline

**Notation** — used consistently throughout the diagram:
- `S` : number of subjects (domains)
- `C` : number of EEG channels
- `T` : number of time samples per window
- `N` : total number of windows across all subjects  (`N = S × trials × windows_per_trial`)
- `[f_lo, f_hi]` : frequency band (Hz) — dataset-specific (e.g. 8–12 Hz, 15–36 Hz)

```
 Continuous EEG — S subjects · C channels · [f_lo, f_hi] Hz band
        │  shape per subject: (C, T_total)
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Band-pass filter  [f_lo, f_hi] Hz (Butterworth order 4)   │
 │  + Sliding windows (4 s · 50% overlap · 128 Hz)             │
 └─────────────────────────────────────────────────────────────┘
        │  (N, C, T)   — N windows across all subjects
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Covariance matrix estimation                               │
 │  Ledoit-Wolf · pyriemann Covariances('lwf')                 │
 └─────────────────────────────────────────────────────────────┘
        │  (N, C, C)   — N symmetric positive-definite matrices
        ▼
 ╔═════════════════════════════════════════════════════════════╗
 ║  LGC — Local Geometric Consistency          [proposed]      ║
 ║  Replaces each C_i with the Riemannian mean of its          ║
 ║  K nearest temporal neighbors within the same class segment ║
 ╚═════════════════════════════════════════════════════════════╝
        │  (N, C, C)   — LGC-processed SPD matrices
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  RCT — Riemannian Centering Transformation                  │
 │  Re-centers each domain to the identity matrix              │
 └─────────────────────────────────────────────────────────────┘
        │  (N, C, C)   — domain-aligned SPD matrices
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  FgMDM classifier (MDM with geodesic filtering)             │
 │  Trained on source domains only                             │
 │  Target labels never used → unsupervised transductive       │
 └─────────────────────────────────────────────────────────────┘
        │  (N,)
        ▼
  Class labels — 0 or 1
```

### Evaluation protocol

The target domain participates in alignment — RCT computes its Riemannian mean from
all available unlabeled covariance matrices — but no target labels are used at any stage.
This constitutes an **unsupervised transductive** transfer learning setting (Pan & Yang, 2010).
In active BCI, RCT estimates the reference matrix from resting-state periods; in passive BCI —
where no rest states exist — we use all available unlabeled target windows.

### Key properties
- **Unsupervised transductive**: no target labels required at any stage — domain alignment uses only unlabeled target data
- **Standard API**: follows the `(X, y, domains)` convention of pyriemann transfer learning
- **Paradigm-agnostic**: validated on mental workload and affective state classification (preliminary)

---

## Installation

```bash
pip install lgc-rct
```

For exact reproducibility of published results:
```bash
pip install -r requirements-freeze.txt
pip install lgc-rct
```

---

## Quick Start

```python
from lgcrct import LGCRCTPipeline, run_loso

# X : np.ndarray, shape (N, C, T) — bandpass-filtered EEG windows
# y : np.ndarray, shape (N,)      — class labels {0, 1}
# domains : np.ndarray, shape (N,) — subject IDs

# LGC-RCT (proposed method — Riemannian mean, K=10)
pipe = LGCRCTPipeline(half_window=10, cov_estimator="lwf")
pipe.fit(X, y, domains, target_domain="subject_01")
y_pred = pipe.predict(X_test, y_test, domains_test)

# Plain RCT baseline (no LGC)
pipe_rct = LGCRCTPipeline(half_window=0)

# LGC-Euclidean ablation (arithmetic mean instead of Riemannian)
pipe_euclid = LGCRCTPipeline(half_window=10, lgc_mean="euclid")

# Full LOSO evaluation
results = run_loso(X, y, domains, half_window=10, cov_estimator="lwf")
results_euclid = run_loso(X, y, domains, half_window=10, lgc_mean="euclid")
```

### Input format
```
X       : np.ndarray (N, C, T)   bandpass-filtered EEG windows
y       : np.ndarray (N,)        class labels {0, 1}
domains : np.ndarray (N,)        subject IDs (str or int)
```

> **Recommendation**: apply sliding-window segmentation before calling LGC-RCT.
> LGC relies on temporal neighbors within each class segment — more windows
> per segment means richer temporal structure and stronger geometric consistency.
> A 4-second window with 50% overlap (2-second step) at 128 Hz is a validated configuration.

---

## Validated Results

### Mental Workload — Team Metrics dataset (private, TNO)
34 pilots, UAV supervision task, binary workload classification, LOSO, 128 Hz.

| Method           | Alpha ACC        | Theta ACC        | No target labels? |
|------------------|:----------------:|:----------------:|:-----------:|
| MDM              | 50.56 ± 1.63     | 51.34 ± 2.15     |      —      |
| RCT              | 57.74 ± 2.56     | 57.94 ± 3.46     |     YES     |
| LGC-RCT K=1      | 63.11 ± 4.04     | 60.61 ± 3.42     |     YES     |
| **LGC-RCT K=10** | **77.73 ± 6.39** | **73.91 ± 6.05** |   **YES**   |

As reported in NAT26 Berlin proceedings.

### Affective State Classification — DEAP dataset *(preliminary)*
32 subjects, emotion recognition, binary classification, LOSO, 128 Hz, 15–36 Hz.

| Task    | Method        |      ACC      | No target labels? | Status      |
|---------|---------------|:------------:|:-----------:|-------------|
| Valence | LGC-RCT K=10  | 79.52 ± 5.87 |     YES     | preliminary |
| Arousal | LGC-RCT K=10  | 75.51 ± 5.68 |     YES     | preliminary |

> These results are **preliminary** and obtained without any dataset-specific tuning.
> The method configuration (K=10, cov_estimator='lwf' (Ledoit-Wolf covariance estimator)) is identical
> to the one used for workload; only the frequency band differs (15–36 Hz for DEAP vs. alpha/theta for Team Metrics).
> They suggest that LGC-RCT generalizes across passive BCI paradigms.
> Full analysis forthcoming.

---

## Repository Structure

```
lgc-rct/
├── lgcrct/                                        # Installable Python package
│   ├── lgc.py                                     # LGC: block-aware local Riemannian mean on P(n)
│   ├── pipeline.py                                # LGCRCTPipeline: fit / predict / transform
│   └── evaluation.py                              # run_loso: LOSO cross-subject evaluation
├── demos/
│   └── 02_demo_deap.py                            # Demo on DEAP public dataset
├── experiments/                                   # Requires Team Metrics dataset (private, TNO)
│   ├── lgc_rct_loso.py                            # Full LOSO experiment script
│   └── lgc_ablation_deap.py                       # LGC-Riemannian vs LGC-Euclidean ablation (DEAP)
├── data/
│   └── FORMAT.md                                  # Dataset format and download instructions
└── results/
    ├── loso_34pilots_published.csv                # NAT26 Berlin — K ablation (alpha & theta)
    ├── LGC-RCT_K10_DEAP_VALENCE_15-36Hz_LOSO.csv # DEAP valence (preliminary)
    └── LGC-RCT_K10_DEAP_AROUSAL_15-36Hz_LOSO.csv # DEAP arousal (preliminary)
```

---

## Reproducibility

The LGC-RCT pipeline is fully deterministic: Ledoit-Wolf covariance estimation,
Riemannian mean computation, and RCT alignment all have unique, closed-form or
convergent solutions on the SPD manifold. LOSO partitioning is determined entirely
by subject IDs. No random seed is required — results are exactly reproducible
given the same data and dependency versions (see `requirements-freeze.txt`).

This property extends to third-party use: researchers who apply lgc-rct to their
own datasets and publish results can guarantee exact replication by any laboratory,
without dependence on random initialization, hardware, or number of runs.

### Dataset availability

The Team Metrics dataset is proprietary (TNO, Netherlands) and cannot be shared.
To access it for replication of the NAT26 Berlin results, contact Anne-Marie Brouwer (TNO).

The DEAP demo (`demos/02_demo_deap.py`) runs on the publicly available DEAP dataset.
See `data/FORMAT.md` for download instructions.

---

## Dependencies

| Package      | Version  |
|--------------|----------|
| Python       | ≥ 3.9    |
| numpy        | ≥ 1.26.4 |
| pyriemann    | ≥ 0.9    |
| scikit-learn | ≥ 1.6.1  |

> `experiments/lgc_rct_loso.py` additionally requires **gumpy** for EEG bandpass filtering.
> gumpy is not available on PyPI — install from: https://github.com/gumpy-bci/gumpy
> Always pass `fs=128` explicitly (gumpy defaults to fs=256).

---

## Citation

If you use this method, please cite the paper:

```bibtex
@inproceedings{licona2026lgcrct,
  title     = {Local Geometric Consistency for Cross-Subject {EEG} Transfer Learning},
  author    = {Licona Mu{\~n}oz, Ricardo and
               Guetschel, Pierre and
               Bruin, Juliette and
               Yilmaz, Efecan and
               Brouwer, Anne-Marie},
  booktitle = {Proceedings of the Fifth Neuroadaptive Technology Conference (NAT'26)},
  year      = {2026},
  note      = {Berlin, Germany}
}
```

*Paper DOI will be added upon proceedings publication.*

If you use this software specifically, please also cite the software release:

```bibtex
@software{licona2026lgcrct_software,
  author    = {Licona Mu{\~n}oz, Ricardo and
               Guetschel, Pierre and
               Bruin, Juliette and
               Yilmaz, Efecan and
               Brouwer, Anne-Marie},
  title     = {lgc-rct: Local Geometric Consistency + Riemannian Centering Transformation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.2.0},
  doi       = {10.5281/zenodo.19225508}
}
```

---

## References

[1] Zanini, P., Congedo, M., Jutten, C., Said, S., & Berthoumieu, Y. (2018).
Transfer Learning: A Riemannian Geometry Framework With Applications to
Brain–Computer Interfaces. *IEEE Transactions on Biomedical Engineering*, 65(5), 1107–1116.
https://doi.org/10.1109/TBME.2017.2742541

---

## License

MIT License — see `LICENSE` for details.
