# Part II — Methodology and Implementation

*Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS*

---

## 6. Experimental Design

### 6.1 Research objectives, restated operationally

**RO1 — Quantify robustness of CNN, LSTM, Transformer and GNN NIDS against adversarial
attacks.**
Operationally: train four architectures under identical conditions; subject each to
white-box evasion (FGSM, PGD), black-box transfer, and poisoning (label-flipping,
clean-label); report attack success rate, detection-rate loss and false-positive change for
each.

**RO2 — Constrain adversarial perturbations to realistic network traffic.**
Operationally: define a projection operator onto the set of physically realizable flows;
run every evasion attack twice, once unconstrained and once constrained; report the
difference in attack success, and independently measure what fraction of *unconstrained*
adversarial examples are infeasible.

The second objective is what makes the first honest. Unconstrained numbers measure a
mathematical property of the decision boundary; constrained numbers measure a security
property of the system.

### 6.2 Controlled comparison

Every architecture receives **identical** treatment:

| Held constant | Value |
|---|---|
| Input representation | same 123-dimensional encoded vector |
| Output head | 2 logits (benign / attack) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | ReduceLROnPlateau (factor 0.5, patience 2) |
| Early stopping | patience 6 on validation loss, best weights restored |
| Batch size | 512 |
| Max epochs | 20 |
| Class weighting | balanced |
| Gradient clipping | max-norm 5.0 |
| Seed | 42 |

Any observed difference in robustness is therefore attributable to architecture, not to
training setup. This is the single most important methodological control: much of the
published NIDS-robustness literature compares models trained with different budgets and
hyper-parameters, which confounds architecture with optimisation.

### 6.3 Data splits

```
KDDTrain+  (125,973)  ──┬── train      (107,077)  model fitting
                        └── validation  (18,896)  early stopping, LR schedule
KDDTest+    (22,544)  ───── test                  touched once, at the end
```

The validation split is carved from *training* data only; the official test split is never
used for model selection. The official train/test partition is preserved rather than
reshuffled, for the reason established in Part I §3.1: `KDDTest+` contains attack families
absent from training, and merging the two converts a zero-day generalisation test into an
interpolation test.

### 6.4 Phase structure

| Phase | Experiment | Output |
|---|---|---|
| 1 | Baseline training, 4 architectures | clean accuracy, DR, FPR, per-attack-family breakdown |
| 2 | White-box evasion: {FGSM, PGD} × {constrained, unconstrained} × 4 ε values | ASR, perturbation norms, feasibility violation rates |
| 3 | Black-box transfer: all 16 (surrogate, target) pairs × 2 regimes | transfer matrix |
| 4 | Poisoning: 3 label-flip strategies × 3 rates + clean-label × 3 rates | DR drop, FPR change |
| 5 | Adversarial training defence | ASR reduction vs clean-accuracy cost |
| 6 | Cross-domain transfer (needs ≥2 datasets) | in-domain vs out-of-domain gap |

Phases are independently resumable — each writes results on completion. On CPU the full
suite runs for hours, and a failure in phase 4 must not discard phases 1–3.

---

## 7. Architectures

All four consume the same flat feature vector and emit two logits. Adapting sequence and
graph models to tabular flow records requires design choices that are usually
under-documented in the literature; each is stated explicitly below, **including its
limitation**.

### 7.1 CNN

1-D convolution over the feature vector: three Conv1d blocks (32→64→64 channels, kernel 3,
BatchNorm + ReLU), global average pooling, then a two-layer MLP head with dropout.

*Assumption and limitation.* Convolution presupposes that adjacent positions are related.
For a flow record this is only loosely true — the canonical feature ordering groups related
quantities (basic, then content, then windowed traffic statistics), so the kernel acts as a
**local feature-interaction detector** rather than a spatial one. It cannot see interactions
between distant features except through depth.

*Why global pooling rather than flatten.* It makes the head's parameter count independent of
input width, so identical hyper-parameters work across datasets with different feature
counts — necessary for the cross-domain experiments.

**23,298 parameters.**

### 7.2 LSTM

Bidirectional LSTM reading the feature vector as a **chunked** pseudo-sequence: features are
grouped into blocks of 8, each block forming one timestep (16 steps for 123 features).
Mean-pooled across steps, then an MLP head.

*Why chunked rather than one feature per timestep.* Two reasons, one practical and one
representational. Recurrence cannot be parallelised across timesteps, so a 123-step sequence
costs ~8× the sequential work of a 16-step one — measured at 95 ms/batch versus a projected
~760 ms, which was the difference between a feasible and infeasible study on CPU.
Representationally, a single scalar gives a recurrent cell almost nothing to gate on,
whereas a block of related features is a coherent sub-record.

*Limitation, stated plainly.* **This is not temporal modelling.** A single flow record has
no time axis. Genuine temporal NIDS would sequence *multiple flows* per host over time.
This is the standard "LSTM on tabular data" baseline found throughout the NIDS literature,
and its value here is as a recurrent-architecture datapoint in the robustness comparison —
not as evidence about sequence models for traffic.

**93,890 parameters.**

### 7.3 Transformer

Feature-group tokenising encoder. Features are chunked into blocks of 8; each block is
projected to a 64-dimensional embedding and given a learned positional embedding
identifying *which* group it is. A CLS token is prepended, two pre-norm encoder layers with
4-head self-attention run, and classification reads from the CLS position.

*Why chunked.* Self-attention is O(n_tokens²). At one token per feature (124 tokens) this
cost **4,029 ms/batch** on CPU — 4.7 hours for a single 20-epoch run, and phase 4 retrains
each model twelve times. Chunking to 16 tokens reduced it to **283 ms/batch**, a 14×
speed-up, at no measured accuracy cost.

*Advantage over the CNN.* No adjacency assumption — attention models all-pairs interactions
between feature groups directly. This is the closest of the four to a principled tabular
architecture (cf. TabTransformer, FT-Transformer).

**101,890 parameters.**

### 7.4 GNN

GraphSAGE-style message passing over a **k-nearest-neighbour similarity graph** of flow
records, built per batch from Euclidean distance in feature space (k = 10). Two message-
passing layers (`h' = W_self·h + W_neigh·mean_{j∈N(i)} h_j`), LayerNorm, dropout, linear
head. Implemented directly rather than via torch-geometric, to keep the dependency
footprint small and the mechanics visible.

*The honest caveat, and it is significant.* NSL-KDD contains **no host identifiers** — no
source or destination IP. A host-communication graph, which is what "GNN for NIDS" normally
means (e.g. E-GraphSAGE), **cannot be built from it**. The k-NN feature-similarity graph is
a genuine relational inductive bias, and a legitimate architecture to compare, but it is
*not* topology-aware intrusion detection. Results here should not be read as evidence about
graph-based NIDS that exploit network structure.

*A second subtlety.* Because the graph is rebuilt per batch, perturbing one node changes
both its own features *and* its neighbourhood. This is arguably realistic — altering traffic
does change its similarity to other traffic — but it makes the GNN's attack surface subtly
larger than the other three. The adjacency is `detach()`ed so gradients flow through node
features rather than through the discrete top-k selection, which has zero gradient almost
everywhere.

**82,690 parameters.**

---

## 8. Preprocessing

### 8.1 Pipeline

```
raw CSV
  → binarise labels (benign=0, attack=1); retain fine-grained category for breakdowns
  → drop leakage columns (NSL-KDD `difficulty`, UNSW `id`)
  → drop constant columns (NSL-KDD `num_outbound_cmds` is identically zero)
  → map unseen test-set categories to `__unseen__`
  → numeric:     log1p → min-max to [0,1]
    rates/binary: passthrough (already normalised)
    categorical:  one-hot with a reserved `__unseen__` slot
  → 123-dimensional float32 vector
```

`__unseen__` is not defensive boilerplate: `KDDTest+` genuinely contains service values
absent from `KDDTrain+`, as part of its deliberate difficulty.

### 8.2 Column provenance tracking

The `Preprocessor` records, for every original feature, **which output columns it occupies**
and what its physical type and mutability are. This bookkeeping is what makes RO2
implementable: to forbid perturbation of `protocol_type`, the constraint layer must know
which columns the one-hot expansion of `protocol_type` occupies.

For NSL-KDD the 123 encoded columns partition as:

| Mutability class | Columns | Meaning |
|---|---:|---|
| Immutable | 87 | attacker cannot change (incl. all one-hot categoricals) |
| Increase-only | 6 | attacker may raise but not lower |
| Free | 23 | genuinely attacker-controlled both ways |
| Derived | 7 | recomputed from parents, never perturbed directly |

Seven dependency rules are auto-discovered: `serror_rate`, `rerror_rate`, `same_srv_rate`,
`diff_srv_rate`, `dst_host_same_srv_rate`, `dst_host_serror_rate`, `dst_host_rerror_rate`.

---

## 9. Attack Implementations

### 9.1 FGSM

```
x_adv = Π( x + ε · sign(∇ₓ L(f(x), y)) )
```
Single step. Cheap, and a useful lower bound on vulnerability.

### 9.2 PGD

```
x_adv ← x + Uniform(−ε, ε)                     # random start
repeat T times:
    x_adv ← x_adv + α · sign(∇ₓ L(f(x_adv), y))
    x_adv ← x + clip(x_adv − x, −ε, +ε)        # projection 1: ε-ball
    x_adv ← Π(x_adv, x)                        # projection 2: realizable set
```

with α = 2.5ε/T and T = 20. The **double projection** is the technically interesting part of
applying PGD to network traffic: the iterate must satisfy both the perturbation budget and
the traffic semantics, and their intersection can be far smaller than either alone. Random
start helps escape flat-gradient regions where single-step FGSM underestimates
vulnerability.

### 9.3 Black-box transfer

Craft on a surrogate with PGD; evaluate on an unseen target. Success is computed **only over
samples the target originally classified correctly** — flipping an already-missed sample is
not an attack achievement, and counting it inflates rates.

### 9.4 Label-flip poisoning

Three strategies, with rates expressed as a fraction of the **whole** training set (not of
the affected class) so they remain comparable across datasets with different class balance:

* `attack_to_benign` — the operationally sensible one. Teaches the model that malicious
  patterns are normal; induces false negatives, the costly error. Also stealthier against
  class-balance audits than symmetric flipping.
* `benign_to_attack` — induces false positives; a denial-of-service on the analyst team
  rather than on the network.
* `symmetric` — random both directions; the classic academic setting, included as a
  reference point.

### 9.5 Clean-label poisoning

Feature-collision. Benign training points are perturbed — within budget and within realistic
constraints — so their feature representation drifts toward the attack manifold, while their
benign label is **retained**. Gradient *descent* on the loss w.r.t. the attack class (the
opposite sign to evasion).

*Why this is the dangerous one.* Nothing about a poisoned record is detectably mislabelled.
An analyst auditing the training set sees benign traffic labelled benign. Only the features
are subtly off — and "subtly off" is indistinguishable from natural variation in a dataset
of millions of flows.

### 9.6 Evaluation protocol

Evasion is evaluated on **attack-class samples only** (y = 1). This matters and is a common
reporting error: an evasion adversary wants malicious traffic classified benign. Perturbing
*benign* traffic into the attack class is false-positive injection — a different threat with
different consequences. Merging both into one "attack success rate" makes numbers
incomparable across papers.

---

## 10. Software

### 10.1 Layout

```
D:\NIDS\
├── src/nids/
│   ├── data/
│   │   ├── schema.py         canonical features, mutability, cross-domain mappings
│   │   ├── loaders.py        NSL-KDD / UNSW-NB15 / CICIDS2017 → uniform container
│   │   └── preprocess.py     encoding, scaling, column provenance tracking
│   ├── models/
│   │   ├── architectures.py  CNN, LSTM, Transformer, GNN + registry
│   │   └── trainer.py        shared training loop, adversarial training
│   ├── attacks/
│   │   ├── constraints.py    RO2 — projection onto realizable traffic
│   │   ├── evasion.py        FGSM, PGD, black-box transfer
│   │   └── poisoning.py      label-flip, clean-label
│   ├── eval/metrics.py       detection metrics, robustness scores
│   └── experiments/
│       ├── run_all.py        5-phase driver, resumable
│       └── cross_domain.py   train-on-A / test-on-B
├── scripts/build_colab_notebook.py
├── notebooks/NIDS_GPU_Experiments.ipynb
├── data/raw/nsl_kdd/
├── outputs/{models,metrics,figures,logs,adversarial}/
└── docs/
```

### 10.2 Dependencies

`torch` (2.13, CPU build), `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`,
`scipy`. No torch-geometric — the GNN is implemented directly, keeping the install small
and the message-passing explicit.

### 10.3 Reproducibility

Seeds fixed (42) for Python, NumPy and Torch. Note that seeding alone does not give
bit-exact GPU determinism (that needs `torch.use_deterministic_algorithms(True)`, at a
significant speed cost); what it does guarantee is stable splits and initialisation, which
is what makes runs comparable.

### 10.4 Running

```bash
# NSL-KDD, full suite (CPU, several hours)
python -m nids.experiments.run_all --dataset nsl_kdd --data-dir data/raw/nsl_kdd

# selected phases only
python -m nids.experiments.run_all --phases 1,2 --epochs 20

# cross-domain (needs ≥2 datasets)
python -m nids.experiments.cross_domain --nsl-kdd data/raw/nsl_kdd --unsw-nb15 data/raw/unsw
```

UNSW-NB15 and CICIDS2017 are gated behind registration forms and cannot be scripted;
`notebooks/NIDS_GPU_Experiments.ipynb` runs the identical pipeline on GPU once they are
obtained.
