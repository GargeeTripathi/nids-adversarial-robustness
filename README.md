# Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS

Research codebase studying how deep-learning network intrusion detectors fail under
adversarial pressure, and whether those failures are real or artefacts of an unrealistic
threat model.

---

## Research objectives

**RO1 — Quantify robustness of CNN, LSTM, Transformer and GNN NIDS against adversarial
attacks.** Evasion (FGSM, PGD, black-box transfer) and poisoning (label-flipping,
clean-label), with all four architectures trained under identical conditions so that
differences are attributable to architecture rather than to training setup.

**RO2 — Constrain adversarial perturbations to realistic network traffic.** Standard
adversarial ML assumes an image-like threat model in which any feature may move in any
direction. Network flows do not work that way. This project defines a projection operator
onto the set of *physically realizable* flows and measures how much reported attack success
is inflated by ignoring it.

---

## The central argument

An unconstrained gradient attack on flow features produces records like:

| Proposed adversarial value | Why it cannot exist |
|---|---|
| `duration = −0.4 s` | time does not run backwards |
| `src_bytes = 1204.7` | bytes are integers |
| `protocol_type = 0.3·tcp + 0.7·udp` | a packet has one protocol |
| `src_bytes = 12,000,000`, `duration = 0.001` | ≈ 12 GB in a millisecond |
| `serror_rate = 0.83`, `count = 1` | a rate over one connection is 0 or 1 |
| attacker *reduces* `dst_bytes` | the victim sent those bytes |

An adversarial example that cannot be transmitted is not an attack. Six constraint families
are enforced (`src/nids/attacks/constraints.py`), the most consequential being
**directionality**: a real attacker can pad, delay and add, but cannot un-send a packet or
retroactively shorten a flow.

---

## Repository layout

```
D:\NIDS\
├── src/nids/
│   ├── data/
│   │   ├── schema.py          canonical feature space, per-feature mutability,
│   │   │                      cross-domain mappings
│   │   ├── loaders.py         NSL-KDD / UNSW-NB15 / CICIDS2017 → uniform container
│   │   └── preprocess.py      encoding, log1p scaling, column-provenance tracking
│   ├── models/
│   │   ├── architectures.py   CNN, LSTM, Transformer, GNN + registry
│   │   └── trainer.py         shared training loop, adversarial training
│   ├── attacks/
│   │   ├── constraints.py     RO2 — projection onto realizable traffic
│   │   ├── evasion.py         FGSM, PGD, black-box transfer
│   │   └── poisoning.py       label-flipping, clean-label
│   ├── eval/
│   │   ├── metrics.py         detection metrics, robustness scores
│   │   └── figures.py         JSON results → tables and plots
│   └── experiments/
│       ├── run_all.py         5-phase driver, resumable
│       └── cross_domain.py    train-on-A / test-on-B
├── scripts/build_colab_notebook.py
├── notebooks/NIDS_GPU_Experiments.ipynb
├── docs/
│   ├── part1_foundations.md   NIDS + adversarial ML from first principles, glossary
│   ├── part2_methodology.md   design, architectures, attack implementations
│   └── part3_results.md       results and analysis
├── data/raw/nsl_kdd/          KDDTrain+.txt, KDDTest+.txt
└── outputs/{models,metrics,figures,tables,logs,adversarial}/
```

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The CPU wheel of torch is ~200 MB; the default CUDA wheel is ~2.5 GB and is unnecessary
without a GPU.

---

## Data

| Dataset | Rows | Access |
|---|---:|---|
| **NSL-KDD** | 148,517 | free download, included via `scripts/` |
| **UNSW-NB15** | 257,673 | [registration form](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| **CICIDS2017** | ~2.8M | [registration form](https://www.unb.ca/cic/datasets/ids-2017.html) |

NSL-KDD runs locally on CPU. The other two are gated behind forms that cannot be scripted;
once obtained, `notebooks/NIDS_GPU_Experiments.ipynb` runs the identical pipeline on GPU.

### Two dataset properties that shape every result

**1. NSL-KDD's official split is a zero-day test.** `KDDTest+` deliberately contains attack
families absent from `KDDTrain+`. Measured with a logistic-regression control:

| Evaluation | Accuracy | ROC-AUC |
|---|---:|---:|
| Validation (same distribution) | 97.98% | 0.998 |
| Official `KDDTest+` | 74.37% | 0.776 |

A 23-point gap from the same model. Papers reporting ~99% on NSL-KDD have shuffled the two
files together. **This codebase preserves the official split.**

**2. The three datasets barely overlap.** The genuine shared feature space is five columns —
`duration`, `protocol_type`, `src_bytes`, `dst_bytes`, `dst_host_count`. Cross-domain
results are correspondingly low, and that low number is the finding: flow features are not
portable across capture environments. Zero-filling absent features (which lets the model
identify its domain from the padding) and fine-tuning on the target (which measures domain
adaptation, not transferability) are both deliberately avoided.

---

## Running

```bash
# full suite on NSL-KDD (CPU, several hours; resumable)
python -m nids.experiments.run_all --dataset nsl_kdd --data-dir data/raw/nsl_kdd

# selected phases
python -m nids.experiments.run_all --phases 1,2 --epochs 20

# figures and tables from saved results
python -m nids.eval.figures --dataset nsl_kdd

# cross-domain (needs ≥2 datasets)
python -m nids.experiments.cross_domain \
    --nsl-kdd data/raw/nsl_kdd --unsw-nb15 data/raw/unsw_nb15
```

| Phase | Experiment |
|---|---|
| 1 | Baseline training, 4 architectures |
| 2 | White-box evasion: {FGSM, PGD} × {constrained, unconstrained} × 4 ε |
| 3 | Black-box transfer, all 16 (surrogate, target) pairs |
| 4 | Poisoning: 3 label-flip strategies × 3 rates + clean-label |
| 5 | Adversarial-training defence |

---

## Design decisions worth knowing

**Identical training for every architecture.** Same optimiser, schedule, early stopping,
batch size, seed. Much of the published NIDS-robustness literature compares models trained
with different budgets, which confounds architecture with optimisation.

**log1p + min-max scaling, not z-score.** `src_bytes` spans 0 to 1.4 billion with a median
of 44. Under standardisation nearly all mass collapses near zero and a fixed ε means wildly
different things per feature. Under log1p, a perturbation budget becomes a *multiplicative*
change in bytes — the physically meaningful notion — and all features share [0, 1] so one ε
is comparable across them. Monotonicity also makes the directionality constraint a simple
element-wise maximum.

**Attack success measured on attack-class samples only.** An evasion adversary wants
malicious traffic classified benign. Perturbing benign traffic into the attack class is
false-positive injection — a different threat. Merging both makes numbers incomparable.

**Chunked LSTM and Transformer inputs.** Both group features into blocks of 8 rather than
one token/timestep per feature. For the Transformer this was decisive: self-attention is
O(n²), and 124 single-feature tokens cost 4,029 ms/batch on CPU versus 283 ms chunked — a
14× speed-up that made the study feasible.

---

## Known limitations

* **The GNN is not topology-aware.** NSL-KDD has no host identifiers, so a host-
  communication graph cannot be built from it. The k-NN feature-similarity graph is a
  genuine relational inductive bias but is *not* E-GraphSAGE-style structural detection, and
  results should not be read as evidence about those systems.
* **The LSTM does not model time.** A single flow record has no time axis; this is the
  standard "LSTM on tabular data" baseline, included as a recurrent-architecture datapoint.
* **Dependency repair is sound but incomplete.** Rate features are snapped to achievable
  granularities (a rate over *n* connections must be a multiple of 1/*n*), but the
  underlying per-connection outcomes are not recoverable from a single record, so full
  generative reconstruction is not possible.
* **Constraints encode one threat model.** An attacker who controls both endpoints, or who
  can influence the victim's responses, faces fewer restrictions than modelled here.
* **NSL-KDD is 25 years old** and its content features assume unencrypted payloads.

---

## Documentation

Full write-up in `docs/`:

* **Part I — Foundations.** NIDS from first principles (signature vs anomaly vs deep
  learning, the base-rate problem), adversarial ML (evasion vs poisoning, threat models,
  transferability), the three datasets, why network traffic is not an image, and a glossary.
* **Part II — Methodology.** Experimental design, the four architectures with their
  assumptions and limitations, preprocessing, attack implementations.
* **Part III — Results and Analysis.**
