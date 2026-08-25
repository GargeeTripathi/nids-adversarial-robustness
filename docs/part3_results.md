# Part III — Results and Analysis

*Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS*

> **Status.** All five NSL-KDD experiment phases are complete: baselines, white-box evasion,
> black-box transfer, poisoning (64 retrains) and the adversarial-training defence. The one
> section still marked *(pending)* is §17, cross-domain transferability, which requires the
> two datasets behind registration forms (UNSW-NB15, CICIDS2017) and is half of RO1.
>
> No number in this document is estimated, extrapolated, or illustrative — every figure
> quoted is read from a completed run, and the pending section is left empty rather than
> filled with plausible values. Where a phase contradicted the expectation recorded here
> before it ran, the original expectation is stated and marked wrong rather than replaced.

---

## 11. How to read these results

Three conventions apply throughout, each chosen to avoid a specific common distortion.

**Accuracy is never reported alone.** On a dataset that is 80% benign, a detector that never
alarms scores 80%. Detection rate (fraction of attacks caught) and false-positive rate
(fraction of benign traffic wrongly alarmed) are reported alongside it, and the FPR is
usually the more operationally decisive of the two — see Part I §1.4 on the base-rate
problem.

**Attack success rate is computed over correctly-classified attack samples only.** Flipping
a sample the model already got wrong is not an attack achievement; counting it inflates the
rate. Formally, with *C* = {i : f(xᵢ) = yᵢ = 1}:

> ASR = |{i ∈ C : f(x'ᵢ) ≠ yᵢ}| / |C|

**Every evasion result appears twice** — once unconstrained (the conventional threat model)
and once constrained to realizable traffic. The gap between them is Research Objective 2's
answer, and neither number alone tells the truth: the unconstrained figure describes the
decision boundary's geometry, the constrained figure describes the security of the system.

---

## 12. Baseline performance

### 12.1 Results

| Model | Params | Val acc | Test acc | Gap | DR | FPR | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 23,298 | 0.9916 | **0.8026** | −0.189 | 0.6728 | **0.0259** | 579 |
| LSTM | 93,890 | 0.9941 | **0.8083** | −0.186 | **0.7207** | 0.0759 | 409 |
| Transformer | 101,890 | 0.9966 | 0.7720 | −0.225 | 0.6537 | 0.0716 | 1212 |
| GNN | 82,690 | **0.9986** | 0.7740 | −0.225 | 0.6617 | 0.0774 | 118 |

Reference controls on the same encoded features:

| Control | Val acc | Test acc | Test ROC-AUC |
|---|---:|---:|---:|
| Logistic regression | 0.9798 | 0.7437 | 0.7764 |
| Random forest | — | 0.7762 | 0.9591 |

### 12.2 The generalisation gap is the headline

Both completed architectures reach **>99% on validation** and **~80% on the official
`KDDTest+`** — a gap of roughly 19 points, from the same model on the same features.

This is not overfitting in the usual sense. The validation split is drawn from `KDDTrain+`
and is i.i.d. with training data; the official test set deliberately contains **attack
families absent from training**. The gap therefore measures something specific and
important: *how well a detector recognises attacks it has never seen*. That is the
zero-day question, and it is the question a NIDS exists to answer.

Three consequences:

1. **Published NSL-KDD results near 99% are measuring the wrong thing.** Reaching that
   figure requires pooling `KDDTrain+` and `KDDTest+` and re-splitting at random, which
   converts a novel-attack test into an interpolation test. The 19-point gap quantifies
   exactly how much such a protocol flatters the model.
2. **The deep models beat the linear control on novel attacks, modestly.** CNN 0.8026 and
   LSTM 0.8083 versus logistic regression 0.7437 — a real but unspectacular ~5-point
   advantage, considerably smaller than the difference in model complexity would suggest.
3. **Random forest's ROC-AUC (0.9591) exceeds its accuracy-implied ranking.** Its threshold
   is poorly calibrated for this shifted distribution while its *ordering* of samples
   remains good. This is a reminder that a single operating point can misrepresent a
   detector, and is why ROC-AUC is reported alongside.

### 12.3 The detection-rate / false-positive trade-off

LSTM catches more attacks (DR 0.721 vs 0.673) but alarms on benign traffic nearly three
times as often (FPR 0.076 vs 0.026). Which is preferable is not a machine-learning question
but an operational one, and the base-rate arithmetic of Part I §1.4 decides it: at a
realistic attack prevalence, tripling the false-positive rate triples the analyst workload
for a ~5-point detection gain. On a link carrying 1M flows/day, CNN's FPR produces ~25,900
false alarms daily and LSTM's ~75,900 — both unmanageable without further filtering, which
is itself a finding about the practical readiness of these models.

---

## 13. White-box evasion

Phase 2 output: `outputs/metrics/nsl_kdd_phase2_evasion.json`. Attack success rate over the
12,833 attack-class samples of `KDDTest+`, at four perturbation budgets.

### 13.1 Results

**CNN**

| Attack / regime | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---:|---:|---:|---:|
| FGSM unconstrained | 0.999 | 1.000 | 1.000 | 1.000 |
| FGSM constrained | 0.060 | 0.281 | 0.678 | 0.575 |
| PGD unconstrained | 0.999 | 1.000 | 1.000 | 1.000 |
| **PGD constrained** | **0.057** | **0.227** | **0.716** | **0.962** |

**LSTM**

| Attack / regime | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---:|---:|---:|---:|
| FGSM unconstrained | 0.078 | 0.325 | 0.460 | 0.578 |
| FGSM constrained | 0.004 | 0.099 | 0.124 | 0.179 |
| PGD unconstrained | 0.080 | 0.410 | 0.773 | 0.902 |
| **PGD constrained** | **0.004** | **0.036** | **0.084** | **0.173** |

**Transformer**

| Attack / regime | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---:|---:|---:|---:|
| FGSM unconstrained | 0.022 | 0.210 | 0.275 | 0.454 |
| FGSM constrained | 0.004 | 0.030 | 0.056 | 0.227 |
| PGD unconstrained | 0.023 | 0.282 | 0.493 | 0.989 |
| **PGD constrained** | **0.004** | **0.022** | **0.058** | **0.187** |

**GNN**

| Attack / regime | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---:|---:|---:|---:|
| FGSM unconstrained | 0.055 | 0.152 | 0.220 | 0.288 |
| FGSM constrained | 0.010 | 0.054 | 0.073 | 0.137 |
| PGD unconstrained | 0.069 | 0.210 | 0.461 | 0.824 |
| **PGD constrained** | **0.008** | **0.039** | **0.066** | **0.121** |

**Summary — PGD constrained (the security-relevant row)**

| Model | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---:|---:|---:|---:|
| CNN | 0.057 | 0.227 | 0.716 | **0.962** |
| LSTM | 0.004 | 0.036 | 0.084 | 0.173 |
| Transformer | 0.004 | 0.022 | 0.058 | 0.187 |
| GNN | 0.008 | 0.039 | 0.066 | **0.121** |

### 13.2 RO2 — the overstatement is large, and it depends on the budget

Comparing PGD unconstrained against PGD constrained for the CNN:

| ε | Unconstrained | Constrained | Overstatement factor |
|---|---:|---:|---:|
| 0.01 | 0.999 | 0.057 | **17.5×** |
| 0.05 | 1.000 | 0.227 | 4.4× |
| 0.10 | 1.000 | 0.716 | 1.4× |
| 0.20 | 1.000 | 0.962 | 1.04× |

The single most important qualification this study can offer is that **the overstatement is
not a constant**. At a tight budget the constraints are devastating to the attacker — a 17×
inflation of reported success. At a loose budget they barely matter.

The security reading follows from what a large ε *means* physically. Under the log1p
scaling (Part I §4.3), an L∞ budget is a multiplicative change in traffic volume, so ε=0.2
licenses substantial inflation of `duration`, `src_bytes` and connection counts. An attacker
can indeed still evade within the realizable set — but only by making the kind of large,
volumetric change that rate-limiting, volume baselining, or a simple flow-size histogram
would flag on entirely different grounds. The constraints do not eliminate the threat; they
push it into a regime where it is no longer *stealthy*, which is most of what an evasion
attack was for.

At every budget, 100% of the unconstrained adversarial flows violate at least one physical
constraint, while 0% of the constrained ones do (§13.4). A reported "100% evasion at
ε=0.01" is therefore not a weaker version of the truth — it is a statement about flows that
cannot be transmitted at all.

### 13.3 RO1 — architecture changes adversarial exposure far more than it changes accuracy

The two architectures measured so far are within half a point on clean test accuracy
(0.8026 vs 0.8083) and differ by more than an order of magnitude in fragility:

| PGD unconstrained, ε=0.01 | ASR |
|---|---:|
| CNN | 0.999 |
| LSTM | 0.080 |

A 1% perturbation flips essentially every attack sample for the CNN, and one in twelve for
the LSTM. Whatever else the two models have in common, the CNN's decision boundary lies far
closer to the attack-class data manifold.

This matters for practice because clean accuracy is the metric on which architectures are
usually selected. On this evidence it carries **no information** about adversarial exposure:
the near-tied models are not near-tied under attack.

### 13.4 FGSM understates vulnerability — the gradient-masking check

For the LSTM, PGD substantially outperforms FGSM in the unconstrained regime:

| ε | FGSM | PGD |
|---|---:|---:|
| 0.05 | 0.325 | 0.410 |
| 0.10 | 0.460 | 0.773 |
| 0.20 | 0.578 | 0.902 |

At ε=0.2 the iterative attack succeeds on 90% of samples where the single-step attack
managed 58%. This is the signature of **gradient masking** (Athalye et al., 2018): the
single-step gradient is locally uninformative, so FGSM's failure reflects a property of the
*attack*, not robustness of the *model*.

The methodological consequence is direct: **a robustness claim supported only by FGSM is not
supported.** Any of these models evaluated with FGSM alone would look materially safer than
it is — the LSTM by a factor of about 1.6 at ε=0.2.

The CNN shows no such gap (both attacks saturate at 1.000), which is itself informative: its
gradients are perfectly usable, and it is simply fragile.

### 13.5 A non-monotonicity worth not smoothing over

Constrained FGSM on the CNN is non-monotonic in ε — success *falls* from 0.678 at ε=0.1 to
0.575 at ε=0.2 — while constrained PGD rises monotonically to 0.962.

The explanation is mechanical. FGSM takes a single step of size ε; past a certain size the
step overshoots the nearest boundary crossing, and the projection back onto the realizable
set (which restores immutable features, undoes decreases, rounds counts, and re-hardens
one-hot blocks) lands the iterate somewhere less adversarial than a smaller step would have.
PGD, taking many small steps with projection after each, navigates the feasible set instead
of leaping across it.

This is a useful reminder that ε is a *budget*, not a *dose*: with a single-step attack,
spending more of it can achieve less.

### 13.6 Technical note on the ε budget

The constrained attack's measured L∞ perturbation slightly exceeds its nominal budget
(0.1036 against ε=0.1 in a spot check). This is a necessary consequence of integrality:
rounding a count to the nearest integer *in original units* can push the scaled value
marginally outside a real-valued ball. The alternative — always rounding inward — would bias
every perturbation downward and systematically understate attack strength. The excess is
bounded by half a count unit, and is reported rather than concealed.

---

## 14. Black-box transfer

Phase 3 output: `outputs/metrics/nsl_kdd_phase3_transfer.json`. PGD, ε=0.1, 2,000
attack-class samples. Rows = surrogate (the attacker's own model); columns = target (the
deployed model). Diagonal entries are white-box and serve as a consistency check.

### 14.1 Unconstrained

| ↓ Surrogate / Target → | CNN | LSTM | Transformer | GNN |
|---|---:|---:|---:|---:|
| **CNN** | *1.000* | 0.025 | 0.050 | 0.001 |
| **LSTM** | **1.000** | *0.774* | 0.082 | 0.026 |
| **Transformer** | **1.000** | 0.051 | *0.495* | 0.016 |
| **GNN** | **1.000** | 0.101 | 0.117 | *0.408* |

Diagonal mean 0.669; off-diagonal mean 0.289.

### 14.2 Constrained

| ↓ Surrogate / Target → | CNN | LSTM | Transformer | GNN |
|---|---:|---:|---:|---:|
| **CNN** | *0.717* | 0.034 | 0.015 | 0.022 |
| **LSTM** | 0.081 | *0.081* | 0.024 | 0.027 |
| **Transformer** | 0.126 | 0.038 | *0.059* | 0.028 |
| **GNN** | 0.087 | 0.030 | 0.019 | *0.062* |

Diagonal mean 0.230; off-diagonal mean **0.044**.

### 14.3 Implementation validity check

The diagonals reproduce the phase 2 white-box numbers through an entirely separate code
path, which is a meaningful check that the transfer harness is correct:

| Model | Phase 3 diagonal | Phase 2 PGD ε=0.1 |
|---|---:|---:|
| CNN unconstrained | 1.000 | 1.000 |
| LSTM unconstrained | 0.774 | 0.773 |
| Transformer unconstrained | 0.495 | 0.493 |
| GNN unconstrained | 0.408 | 0.461 |
| CNN constrained | 0.717 | 0.716 |
| LSTM constrained | 0.081 | 0.084 |
| Transformer constrained | 0.059 | 0.058 |

Three of four match to within 0.003. The GNN discrepancy (0.408 vs 0.461) is expected and
informative: the GNN rebuilds its k-NN adjacency **per batch**, so a node's classification
depends on which other flows share its batch. Phase 2 and phase 3 batch the data
differently, so the GNN's numbers legitimately differ between them. This is a real property
of the architecture — flagged in advance in `models/architectures.py` — and it means GNN
results carry more variance than the other three and should be read with that in mind.

### 14.4 Transfer is far weaker than the image literature predicts

Off-diagonal transfer averages **28.9% unconstrained** and **4.4% constrained**. Excluding
the CNN column (discussed below), unconstrained transfer among the other three architectures
averages just **5.2%**.

This runs against the non-robust-features account (Part I §2.4), under which models trained
on identical data learn the same brittle patterns and adversarial examples should transfer
readily. That account was developed on images, where every architecture ingests the same
spatial structure. Tabular flow features are heterogeneous — a byte count, a protocol
one-hot, a windowed error rate — and each architecture here imposes a very different
inductive bias on them (local convolution, chunked recurrence, all-pairs attention,
neighbourhood aggregation). The resulting decision boundaries appear to share far less
geometry than image models do.

**Defensive implication.** Architectural diversity is a genuine, cheap defence in this
setting. A defender running an ensemble of dissimilar architectures, or simply keeping their
choice of architecture private, materially raises the bar for a black-box attacker.

### 14.5 The CNN is a universal victim — but only unconstrained

Every surrogate achieves **exactly 1.000** against the CNN in the unconstrained regime,
including the CNN-crafted examples' 0.025 against the LSTM. The transfer relation is
strikingly asymmetric: everything reaches the CNN; the CNN reaches nothing.

The explanation is the CNN's boundary geometry, already visible in phase 2 — a 1%
perturbation flips 99.9% of its attack samples. When the boundary lies that close to the
data manifold in essentially every direction, the *specific* direction a surrogate provides
stops mattering; almost any perturbation of sufficient size crosses it.

**Constraints dissolve this effect.** Transfer to the CNN falls from 1.000 to 0.081–0.126.
The mechanism is dimensional: constraints leave roughly 29 of 123 encoded columns movable
(23 free, 6 increase-only), and within that narrow subspace the useful directions turn out
to be model-specific. The unconstrained attack could exploit all 123 dimensions — including
the 87 it had no right to touch — and those generic directions crossed the CNN's nearby
boundary regardless of provenance.

This is the clearest single illustration of why RO2 matters. Under conventional evaluation
one would conclude that *a CNN-based NIDS can be defeated blind by any surrogate*. Under
realizable-traffic constraints, the same detector is reachable only white-box. Those are
completely different security postures, and only the second one is true.

---

## 15. Poisoning

Phase 4 output: `outputs/metrics/nsl_kdd_phase4_poisoning.json`. Sixty-four cells: four
architectures × four conditions (three label-flip strategies plus clean-label) × four poison
rates, each cell an independent retrain from scratch on poisoned data followed by evaluation
on the clean `KDDTest+`. Damage is measured on **clean** test data throughout — the point of
a poisoning attack is that it degrades ordinary behaviour permanently, regardless of what the
attacker sends afterwards.

Poison rates are fractions of the **training split**, which holds 107,076 rows after the
validation carve-out: 49,835 attack and 57,241 benign. A requested rate is therefore bounded
by the pool the strategy draws from, and `label_flip_poison` clamps to it. The headline case
is `attack_to_benign` at 0.50, which saturates at an effective 0.465 — *every attack label in
the training split flipped*.

Section 15 revises the expectations that stood here before the phase ran. Two of the three
predictions were wrong, and saying so is more useful than quietly replacing them.

### 15.1 The only reliable effect is a cliff at saturation

Across all four architectures, no label-flip rate below saturation produces reliable damage,
and saturation produces total failure. Detection rate on `KDDTest+`:

| Strategy | Architecture | clean | 0.05 | 0.20 | 0.35 | 0.50 |
|---|---|---:|---:|---:|---:|---:|
| attack→benign | CNN | 0.673 | 0.711 | 0.694 | 0.820 | **0.000** |
| | LSTM | 0.721 | 0.731 | 0.734 | 0.739 | **0.000** |
| | Transformer | 0.654 | 0.671 | 0.674 | 0.741 | **0.000** |
| | GNN | 0.662 | 0.660 | 0.620 | 0.667 | **0.000** |
| symmetric | CNN | 0.673 | 0.643 | 0.661 | 0.705 | **0.000** |
| | LSTM | 0.721 | 0.649 | 0.674 | 0.641 | **0.000** |
| | Transformer | 0.654 | 0.617 | 0.648 | 0.590 | **0.000** |
| | GNN | 0.662 | 0.663 | 0.658 | 0.601 | **0.9998** |

At rate 0.35 under `attack_to_benign`, 37,476 of 49,835 attack labels are flipped — **75.2% of
all attack supervision is wrong** — and every architecture still detects at or above its clean
rate. The damage function is not gradual. It is flat, then a cliff.

This directly contradicts the expectation recorded here before the run, that `attack_to_benign`
would be "the most damaging per unit of poison". Per unit of poison it is, below saturation,
worth nothing at all.

### 15.2 The collapse is to a constant classifier, and which constant is arbitrary

The saturation cells are not "degraded models". They are constant predictors:

| Architecture | Strategy | DR | FPR | Accuracy | Predicts |
|---|---|---:|---:|---:|---|
| CNN | symmetric 0.50 | 0.0000 | 0.0000 | 0.4308 | everything benign |
| LSTM | symmetric 0.50 | 0.0000 | 0.0000 | 0.4308 | everything benign |
| Transformer | symmetric 0.50 | 0.0000 | 0.0000 | 0.4308 | everything benign |
| **GNN** | symmetric 0.50 | **0.9998** | **0.9998** | **0.5692** | **everything attack** |

`KDDTest+` is 9,711 benign and 12,833 attack of 22,544 rows — priors of exactly 0.4308 and
0.5692. Each collapsed model's accuracy is precisely the prior of the class it settled on.

The GNN is the informative case. Under identical treatment it degenerates to the *opposite*
constant. Nothing about the data selects benign over attack: at 50% symmetric noise the labels
carry no information, the loss surface has no reason to prefer either constant, and the model
falls into whichever basin initialisation and batch ordering deliver it to. The finding is
therefore **"degenerates to a constant classifier"**, not "degenerates to benign" — a
distinction that only became visible because a fourth architecture happened to fall the other
way.

The two collapse routes are mechanically different. Under `attack_to_benign` at effective
0.465 the attack class is *empty* — there is nothing left to learn. Under `symmetric` at 0.50
the classes are full but the labels are pure noise. Both destroy the label signal; neither is
a subtle attack.

### 15.3 Why `benign_to_attack` never collapses

`benign_to_attack` is the one strategy that does not produce a cliff, at any rate, on any
architecture. The reason is structural rather than statistical: flipping benign records to
"attack" leaves every genuine attack label intact and merely adds noise to the attack class.
The class the detector needs still exists and is still correctly labelled for all 49,835 of
its true members. There is no rate at which this strategy erases the attack concept, so there
is no rate at which the model has nothing to learn.

### 15.4 Below saturation, no effect is distinguishable from training-run variance

This is the section's most consequential negative result, and it constrains everything above.

Detection-rate drop under `benign_to_attack`, the strategy that showed the largest
sub-saturation movement:

| Architecture | 0.05 | 0.20 | 0.35 | 0.50 | Shape |
|---|---:|---:|---:|---:|---|
| CNN | +0.043 | +0.080 | +0.191 | +0.232 | monotonic increasing |
| LSTM | +0.110 | +0.054 | +0.047 | +0.031 | monotonic **decreasing** |
| Transformer | +0.036 | −0.046 | +0.096 | +0.102 | non-monotonic, sign change |
| GNN | +0.028 | +0.029 | +0.081 | +0.048 | non-monotonic |

Four architectures, four qualitatively different curves, no shared direction. A genuine
dose-response relationship does not reverse sign between models under an identical procedure.

The mechanism is visible in the false-positive rates. Every LSTM cell at a sub-saturation rate,
across all four conditions, lands at one of two values with **nothing in between**:

```
0.0280  0.0289  0.0306  0.0306  0.0315  0.0317  0.0319     (7 cells)
0.0732  0.0742  0.0745  0.0759  0.0770                     (5 cells)
```

A continuous input variable producing a two-valued output is the signature of two optimisation
basins, not of a dose response. The LSTM converges to one of two solutions depending on data
ordering, and its clean baseline (0.0759) sits in the upper basin. The transformer shows the
same structure less cleanly. Spread across the twelve sub-saturation cells per architecture:

| Architecture | DR range | DR sd | FPR range | FPR sd |
|---|---|---:|---|---:|
| CNN | 0.482–0.820 | 0.080 | 0.019–0.161 | 0.038 |
| LSTM | 0.601–0.739 | 0.047 | 0.028–0.077 | 0.023 |
| Transformer | 0.557–0.741 | 0.051 | 0.026–0.079 | 0.021 |
| GNN | 0.581–0.667 | 0.027 | 0.072–0.078 | **0.002** |

Most of the "effects" tabulated above are smaller than their own architecture's spread. The
GNN is the exception in the other direction: it is highly stable (FPR sd 0.002), and it is
also the architecture showing the least sub-saturation movement — consistent with the movement
elsewhere being optimisation noise rather than poisoning.

**This spread is an upper bound on noise, not a clean estimate of it**, because these cells
differ in their training data as well as their random trajectory. Separating the two requires a
control this study has not run: repeated training at poison rate 0 across several seeds, giving
a null band. Until that exists, the honest statement is that sub-saturation poisoning effects
on NSL-KDD are **not measurable at single-seed resolution**, and §15 claims nothing about them.
This is a concrete instance of the single-seed limitation already recorded in §18.

### 15.5 The damage metric inverts — a worked example

The GNN's `symmetric` 0.50 cell scores a detection-rate drop of **−0.338**: by that metric, the
best-performing cell in the entire phase, a model that "improved detection by 34 points". It is
in fact the most complete failure available, alerting on 9,709 of 9,711 benign flows.

Detection-rate drop alone is therefore not a sufficient damage measure for a NIDS, and a figure
plotting it alone would rank a constant-alarm classifier first. `fig5_poisoning` plots
detection-rate drop and false-positive-rate change as paired rows for this reason. The same
inversion appears in milder form at CNN `attack_to_benign` 0.35, which reads as −0.147
"improvement" while its false-positive rate rises 0.026 → 0.161.

The general point belongs with the base-rate argument of §1.4: for a detector deployed against
overwhelmingly benign traffic, a metric that ignores false positives cannot express damage.

### 15.6 Clean-label poisoning does nothing, and the constraints are why

Clean-label poisoning is null on three of four architectures (CNN, Transformer, GNN: |ΔDR| ≤
0.039 at every rate). The LSTM shows a modest drop (ΔDR +0.064 to +0.119) with its
false-positive rate pinned at 0.074–0.076, which is the pattern clean-label is theorised to
produce — but at one architecture in four, and inside the variance band established in §15.4,
it cannot be claimed.

The mechanism behind the null is measurable. `mean_attack_probability_drift` records how far
the perturbation moves a poisoned record's attack-probability:

| Architecture | drift at rate 0.05 | 0.20 | 0.35 | 0.50 |
|---|---:|---:|---:|---:|
| CNN | 0.0755 | 0.0755 | 0.0764 | 0.0764 |
| LSTM | 0.0213 | 0.0178 | 0.0178 | 0.0176 |
| Transformer | 0.0282 | 0.0260 | 0.0262 | 0.0259 |
| GNN | 0.0331 | 0.0308 | 0.0304 | 0.0300 |

(Rate-independence is expected: this is a per-point mean, and it converges as *n* grows.)

Within ε=0.1 **and under the realizable-traffic projection**, a poisoned benign record can be
moved only 1.8–7.6 percentage points toward the attack manifold. That is far too small a shift
to relocate a decision boundary, however many such records are inserted. The same constraint
machinery that deflates evasion success in §13 also deflates clean-label poisoning — which is
consistent, and is a point in favour of the constraint model rather than against it. An
unconstrained clean-label attack would move further; it would also propose training records
that could not have been captured from a real network, and a defender harvesting real traffic
would never ingest them.

### 15.7 What this means for an attacker

The operational conclusion is a **threshold, not a curve**. Against a deep NIDS on NSL-KDD,
label-flip poisoning requires the attacker to corrupt essentially *all* attack supervision —
an effective rate of 0.465, or 50% symmetric corruption — before anything happens. Below that,
across 48 sub-saturation cells spanning four architectures and four strategies, nothing
survives a variance check.

This is a far stronger assumption than the poisoning literature usually grants. An adversary
who can flip every attack label in the training corpus does not need a poisoning attack; they
already own the labelling pipeline, and simpler abuses are available to them. The realistic
reading is that **label-flip poisoning is not a practical threat to this class of detector at
attainable corruption rates** — with the important caveat that "attainable" is doing real work
in that sentence, and that a targeted attacker interested in one attack family rather than
global detection rate is not measured by this experiment at all.

### 15.8 What would strengthen this section

1. **A seed-variance control** — retraining at poison rate 0 across ~5 seeds per architecture,
   ~3.2 hours of CPU, giving §15.4 a null band instead of an upper bound.
2. **Per-family damage.** Global detection rate can hide a targeted effect: poisoning that
   destroys detection of one attack family while leaving the aggregate untouched would be
   invisible here. `per_attack_breakdown` in `eval/metrics.py` supports this and it was not
   applied to phase 4.
3. **Rates between 0.35 and 0.465**, to locate the cliff edge precisely rather than bounding it
   between two measured points.

---

## 16. Adversarial training defence

Phase 5 output: `outputs/metrics/nsl_kdd_phase5_defence.json`. Each architecture is retrained
from scratch with 50% of every batch replaced by constrained PGD adversarial examples
(7 steps, ε=0.1), then evaluated against a stronger constrained PGD attack (20 steps, same
budget) and on the clean `KDDTest+`.

### 16.1 Results

| Architecture | ASR undefended | ASR defended | Absolute | Relative | ΔDR | ΔFPR |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.7157 | **0.1063** | −0.6094 | **−85.2%** | +0.0011 | +0.0042 |
| LSTM | 0.0836 | 0.0621 | −0.0215 | −25.7% | −0.0496 | −0.0002 |
| Transformer | 0.0579 | 0.0476 | −0.0103 | −17.8% | +0.0487 | +0.0016 |
| GNN | 0.0657 | **0.0163** | −0.0494 | **−75.2%** | +0.0135 | −0.0046 |

### 16.2 The defence works, and it is close to free

Attack success falls on all four architectures. The CNN — the fragile outlier of §13.3 — falls
from 0.716 to 0.106, and the GNN reaches the lowest absolute vulnerability measured anywhere in
this study, 0.0163.

The expectation recorded in this section before the phase ran was that the defence would buy
robustness at the price of clean accuracy, and that for a NIDS the bill would arrive as false
positives. **That did not happen.** False-positive rates move by at most 0.0046 in either
direction across all four architectures — 0.026→0.030, 0.076→0.076, 0.072→0.073, 0.077→0.073 —
which is inside the run-to-run variance established in §15.4 for every architecture. There is
no measurable false-positive cost to report.

The detection-rate column is more interesting for what it reveals about resolution than about
the defence. The LSTM loses 0.0496 and the transformer gains 0.0487: equal magnitudes, opposite
signs, each approximately one standard deviation of its own architecture's phase-4 spread
(0.047 and 0.051 respectively). Two draws from a noise distribution, not two responses to a
treatment. Read individually, the LSTM result invites the conclusion that the defence costs
five points of detection and is a net loss; the transformer result invites the opposite
conclusion with equal force. Neither is supportable, and the pair is a useful demonstration of
why §15.4's variance argument constrains this section too.

The honest summary is therefore: **constrained adversarial training reduces attack success on
every architecture tested and has no measurable clean-data cost on any of them.**

### 16.3 The benefit is not simply proportional to baseline fragility

The obvious hypothesis — that the defence helps most where the model was weakest — is only half
right. In absolute terms the CNN dominates, as expected. In relative terms the ranking is
CNN 85.2%, GNN 75.2%, LSTM 25.7%, Transformer 17.8%, which does not track baseline
vulnerability: the GNN and the transformer start from almost the same place (0.066 and 0.058)
and respond very differently.

The GNN result deserves weight because the GNN is by a wide margin the most stable architecture
in this study (false-positive-rate sd 0.002 across the twelve sub-saturation phase-4 cells,
against 0.021–0.038 for the others). Its 0.049 absolute reduction is therefore less likely to
be noise than the numerically similar movements seen elsewhere. Why a k-NN-graph model should
absorb constrained adversarial training so well is not established by these experiments, and it
is not sound to speculate from a single seed.

### 16.4 Two reasons not to over-read this

**The defence was evaluated against the attack it was trained on.** Training used constrained
PGD at ε=0.1; evaluation used constrained PGD at ε=0.1 with more steps. This is the standard
adversarial-training protocol and the standard criticism of it. A defence tuned to one attack
at one budget is not shown to be robust to an *adaptive* adversary who knows the defence is
present and optimises against it. Nothing here measures that.

**Reduced attack success is not proof of robustness.** §13.4 established gradient masking in
three of four models on this dataset — FGSM understated vulnerability, the GNN by 2.9×. A model
whose gradients become less useful after adversarial training will show a lower measured ASR
without being harder to evade by an attacker using a gradient-free or transfer-based method.
Evaluating with PGD-20 rather than FGSM guards against the weakest form of this, but does not
settle it. The claim supported by these numbers is "attack success under this attack falls",
not "the model is robust".

Establishing the stronger claim would need an adaptive-attack evaluation and a black-box
transfer evaluation against the defended models — neither of which is in this study, and both
of which are the natural next experiments.

---

## 17. Cross-domain transferability *(pending — requires UNSW-NB15 / CICIDS2017)*

Output: `outputs/metrics/cross_domain.json`.

### 17.1 The structural limit, measured

Before any model is trained, the feature intersection bounds what is achievable:

| Dataset pair | Shared canonical features |
|---|---:|
| NSL-KDD ∩ UNSW-NB15 | 10 |
| NSL-KDD ∩ CICIDS2017 | 5 |
| UNSW-NB15 ∩ CICIDS2017 | 7 |
| **All three** | **5** |

The three-way intersection is `duration`, `protocol_type`, `src_bytes`, `dst_bytes`,
`dst_host_count`.

This is a result in itself, and it is worth stating plainly: **a model trained on one public
NIDS dataset and moved to another sees five features in common.** Whatever cross-domain
accuracy is eventually measured, it is bounded above by what five flow statistics can
express — before any question of distribution shift arises.

### 17.2 Why the low number will be the honest one

Two standard workarounds would raise the reported figure and invalidate it, and both are
deliberately avoided (`experiments/cross_domain.py`):

* **Zero-filling absent features** lets the model infer "feature 47 is always zero,
  therefore I am in domain A" — a domain classifier wearing a transfer result's clothes.
* **Fine-tuning on target data** measures domain *adaptation*, a different and much easier
  question.

A shared encoder is fitted on pooled training data so that one scaled unit means the same
number of bytes in every domain; without this, transfer would fail for reasons of units
rather than of distribution shift.

---

## 18. Threats to validity

**Internal.**
* Single seed (42). Differences of a point or two between architectures should not be
  over-interpreted; the large effects (the 19-point generalisation gap, the
  constrained/unconstrained divergence) are far too big to be seed noise, but the
  architecture *ranking* is not established with confidence.
* Early stopping on validation loss means models may have trained for different numbers of
  epochs. This is the correct protocol — all four get identical *budget and criteria* — but
  it does mean compute is not held exactly constant.
* Constraint definitions encode one threat model. An adversary who controls both endpoints,
  or who can influence the victim's responses, faces fewer restrictions than modelled.

**External.**
* NSL-KDD is 25 years old, synthetic, and its content features assume unencrypted payloads.
  Results establish relative architecture behaviour under attack, not absolute deployability.
* Results from one dataset. The cross-domain experiments exist precisely because
  single-dataset conclusions are known not to generalise — which makes the pending status of
  §17 a genuine limitation of the present state of this work, not a formality.

**Construct.**
* The **GNN is not topology-aware.** NSL-KDD has no host identifiers, so the k-NN
  feature-similarity graph substitutes for a host-communication graph. It is a legitimate
  relational architecture but is *not* E-GraphSAGE-style structural detection, and its
  results say nothing about those systems.
* The **LSTM does not model time.** A single flow record has no time axis; this is the
  standard tabular-LSTM baseline, included as a recurrent datapoint.
* Dependency repair enforces achievable *granularity* for rate features but cannot
  reconstruct the underlying per-connection outcomes, so it is sound but incomplete.

---

## 19. Conclusions

Findings 1–7 rest on phases 1–3, findings 8–10 on phases 4–5. Cross-domain (§17) remains
outstanding and is the one substantive gap in the study as it stands.

1. **The zero-day generalisation gap on NSL-KDD is ~19 points** (>99% validation vs ~80%
   official test) and is the single largest effect measured. Any evaluation protocol that
   conceals it — by pooling and re-splitting the official files — misrepresents detector
   capability by roughly that margin.

2. **Deep models beat a linear control on novel attacks by ~5 points** (0.80 vs 0.74),
   a real but modest advantage relative to their complexity and opacity.

3. **Unconstrained adversarial evaluation overstates attack success by up to 17.5×, and the
   overstatement is budget-dependent.** For the CNN under PGD, unconstrained evaluation
   reports 0.999 at ε=0.01 where realizable perturbations achieve 0.057. The gap narrows as
   the budget grows — at ε=0.2 the same comparison is 1.000 against 0.962 — so a single
   headline "constraints halve the threat" figure would be wrong in both directions
   depending on where it was measured. Independently of the rate: **100% of unconstrained
   adversarial flows violate at least one physical constraint**, which is what licenses
   discarding them.

4. **The residual constrained vulnerability is real but concentrated in one architecture.**
   At ε=0.1 the CNN still yields 0.716 under fully realizable perturbations, whereas the
   LSTM, Transformer and GNN sit at 0.084, 0.058 and 0.066. The constraint correction is a
   correction to the *magnitude* of the published threat, not a refutation of it — but the
   surviving threat belongs to the CNN, and generalising the CNN's exposure to deep NIDS as
   a class is exactly the error this study was built to catch.

5. **Architecture changes adversarial exposure ~8× while changing clean accuracy by under 4
   points.** At ε=0.2 constrained PGD, the CNN yields 0.962 and the GNN 0.121, on test
   accuracies of 0.803 and 0.774. The CNN also has the best false-positive rate (0.026) of
   the four, so the architecture a conventional model-selection procedure would pick on
   clean metrics is the most fragile one under attack. Robustness is not readable from a
   clean-data leaderboard.

6. **FGSM understates vulnerability in 3 of 4 models** — the GNN by 2.9× at ε=0.2 — which is
   the signature of gradient masking. Robustness claims resting on FGSM alone are
   unsupported by their own evidence.

7. **Transfer is weak, and constraints weaken it further** (off-diagonal mean 0.289
   unconstrained → 0.044 constrained). The CNN's apparent "any surrogate defeats it"
   property — 1.000 from every surrogate unconstrained — collapses to 0.081–0.126 once the
   transferred samples must be transmittable. The CNN is reachable white-box; it is not
   meaningfully reachable black-box.

8. **Poisoning damage is a threshold, not a curve.** Across 64 retrains spanning four
   architectures, four strategies and four rates, no label-flip rate below saturation
   produces reliable damage on any architecture. At rate 0.35 — 75.2% of all attack labels
   wrong — every model still detects at or above its clean rate. At saturation all four
   collapse to a *constant classifier*, three predicting benign and the GNN predicting
   attack, each scoring exactly the class prior it settled on. The attacker requirement is
   not "corrupt more than N%" but "corrupt essentially all attack supervision", which is a
   far stronger assumption than the poisoning literature usually grants.

9. **Sub-saturation poisoning effects are not measurable at single-seed resolution.** The
   same manipulation produces four qualitatively different dose-response curves on four
   architectures, and the LSTM's false-positive rate takes one of two discrete values across
   all twelve of its sub-saturation cells — the signature of two optimisation basins rather
   than a response to poison rate. This is a limitation of the experimental design, not a
   property of poisoning, and §15.8 states what would fix it.

10. **Constrained adversarial training reduces attack success on every architecture with no
    measurable clean-data cost.** The CNN falls 0.716 → 0.106 and the GNN reaches 0.0163,
    the lowest vulnerability measured in this study, while false-positive rates move by at
    most 0.0046 anywhere. The anticipated robustness-versus-false-positives trade did not
    materialise. Two caveats bound this: the defence was evaluated against the attack it was
    trained on, and lower measured attack success is not proof of robustness given the
    gradient masking established in §13.4.

**What this study does not establish.** Cross-domain transferability (§17) is unrun, and it is
half of RO1. Every result here comes from one dataset, one seed, and one threat model, and the
architecture *ranking* in particular should not be treated as settled. The findings that
deserve confidence are the ones that were predicted from a mechanism and then reproduced on
every architecture — the constraint overstatement, the saturation cliff, symmetric-noise
invariance — rather than the ones read off a single model's curve.
