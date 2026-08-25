# Resume here

**Updated:** 2026-08-25, 12:00
**Project:** Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS
**Location:** `D:\NIDS\`

---

## One-line status

All five NSL-KDD phases are **complete**, and §§15, 16 and 19 of the report are written.
Phase 4 ran 64 retrains; phase 5 adversarially trained all four architectures. Total runtime
537 minutes. The only outstanding section is §17, cross-domain, still blocked on the two
dataset downloads.

**Phase 4 headline:** poisoning damage is a threshold, not a curve. No label-flip rate below
saturation produces reliable damage on any architecture — at rate 0.35, 75.2%% of attack
labels are wrong and every model still detects at or above its clean rate — and at saturation
all four collapse to a constant classifier (three predict benign, the GNN predicts attack).
Sub-saturation effects are **not measurable at single-seed resolution** (§15.4).

**Phase 5 headline:** constrained adversarial training reduces attack success on all four
architectures with no measurable clean-data cost. CNN 0.716 -> 0.106 (-85.2%%), GNN 0.066 ->
0.016 (-75.2%%, the lowest vulnerability measured anywhere in the study). False-positive rates
move by at most 0.0046. The anticipated robustness-vs-false-positives trade did not appear.

**Known gaps, both recorded in the report rather than hidden:** no seed-variance control
(§15.8), and the defence is evaluated only against the attack it was trained on (§16.4).

---

## Restart command

Everything needed is cached. From `D:\NIDS`:

```bash
PYTHONPATH=/d/NIDS/src /d/NIDS/.venv/Scripts/python.exe -u -m nids.experiments.run_all \
    --dataset nsl_kdd --data-dir data/raw/nsl_kdd --output-dir outputs \
    --epochs 20 --batch-size 512 --phases 4,5
```

This loads the four trained models from `outputs/models/` — **it does not retrain them** —
and runs only the outstanding phases.

**This command is now safe to re-issue after any interruption.** Phase 4 checkpoints
`outputs/metrics/nsl_kdd_phase4_poisoning.json` after every one of its 64
(architecture, condition, rate) cells, and phase 5 checkpoints after each architecture;
both reload the partial file on start and run only the missing entries, logging
`N/64 cells cached, M to run`. Re-running therefore resumes rather than restarting. Pass
`--force` only if you deliberately want to discard the checkpoint and recompute everything.

Expected cost from measured phase-1 training times (cnn 579 s, lstm 409 s, transformer
1212 s, gnn 118 s per fit): **~11.5 h for phase 4** (64 retrains, the transformer alone
~5.4 h) plus **~2 h for phase 5**.

To watch progress: `tail -f outputs/logs/nsl_kdd_run.log`

---

## What is done

| Phase | Status | Output |
|---|---|---|
| 1 — baselines | ✅ complete | `outputs/metrics/nsl_kdd_phase1_baseline.json` |
| 2 — white-box evasion | ✅ complete | `outputs/metrics/nsl_kdd_phase2_evasion.json` |
| 3 — black-box transfer | ✅ complete | `outputs/metrics/nsl_kdd_phase3_transfer.json` |
| 4 — poisoning | ✅ complete, 64/64 cells | `outputs/metrics/nsl_kdd_phase4_poisoning.json` |
| 5 — adversarial training | ✅ complete | `outputs/metrics/nsl_kdd_phase5_defence.json` |
| 6 — cross-domain | ⛔ **blocked** (see below) | — |

**Also saved:** 4 trained models (`outputs/models/`), 4 figures, 8 table files (CSV + MD),
16 adversarial-sample arrays, full run log.

**Codebase:** complete, `src/nids/`. Tests: `.venv/Scripts/python.exe tests/test_constraints.py`
→ 10/10 passing.

**Report:** `docs/NIDS_Adversarial_Robustness_Report.md` (~9,650 words, assembled from
`part1`–`part3`; rebuild with `scripts/build_report.py` after editing any part).

---

## Results established so far

### Baselines (NSL-KDD official split)

| Model | Params | Val acc | Test acc | Gap | DR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 23,298 | 0.9916 | 0.8026 | −0.189 | 0.673 | **0.026** |
| LSTM | 93,890 | 0.9941 | **0.8083** | −0.186 | **0.721** | 0.076 |
| Transformer | 101,890 | 0.9966 | 0.7720 | −0.225 | 0.654 | 0.072 |
| GNN | 82,690 | **0.9986** | 0.7740 | −0.225 | 0.662 | 0.077 |

Controls: logistic regression 0.7437 test / 0.9798 val; random forest 0.7762 test.

### Evasion — PGD, attack success rate

| Model | Regime | ε=0.01 | ε=0.05 | ε=0.1 | ε=0.2 |
|---|---|---:|---:|---:|---:|
| CNN | unconstrained | 0.999 | 1.000 | 1.000 | 1.000 |
| CNN | **constrained** | 0.057 | 0.227 | 0.716 | **0.962** |
| LSTM | unconstrained | 0.080 | 0.410 | 0.773 | 0.902 |
| LSTM | **constrained** | 0.004 | 0.036 | 0.084 | 0.173 |
| Transformer | unconstrained | 0.023 | 0.282 | 0.493 | 0.989 |
| Transformer | **constrained** | 0.004 | 0.022 | 0.058 | 0.187 |
| GNN | unconstrained | 0.069 | 0.210 | 0.461 | 0.824 |
| GNN | **constrained** | 0.008 | 0.039 | 0.066 | **0.121** |

### Transfer (PGD, ε=0.1) — off-diagonal mean

Unconstrained **0.289**; constrained **0.044**.

### Partial phase 4 result from the lost run (superseded, kept for reference)

CNN, attack→benign label flip, detection-rate drop at rates [0.05, 0.10, 0.20]:
`+0.008  −0.013  +0.037` — i.e. essentially **no damage**, even with ~43% of attack labels
flipped. This motivated the rate-grid change described below.

---

## The five findings

1. **Unconstrained adversarial evaluation overstates attack success by up to 17.5×**
   (CNN, ε=0.01: 0.999 → 0.057), and **100% of unconstrained adversarial flows violate at
   least one physical constraint**. The overstatement shrinks as ε grows.
2. **Architecture changes adversarial exposure ~8× while changing clean accuracy <4 points.**
   The CNN has the best FPR and is by far the most fragile (0.962 vs GNN 0.121 at ε=0.2).
3. **FGSM understates vulnerability in 3 of 4 models** (GNN by 2.9× at ε=0.2) — gradient
   masking. FGSM-only robustness claims are unsupported.
4. **Transfer is weak and constraints weaken it further** (0.289 → 0.044). The CNN's apparent
   "any surrogate defeats it" property (all 1.000 unconstrained) collapses to 0.081–0.126
   under constraints — it is reachable only white-box in reality.
5. **NSL-KDD's official split costs 19–23 accuracy points** versus an i.i.d. validation
   split, because `KDDTest+` withholds whole attack families. Published ~99% results require
   pooling and re-splitting, which destroys the zero-day test.

---

## Open question — settled, and now running

The lost partial run showed label-flip poisoning to be **ineffective** at rates ≤20%. Rather
than spend ~11 hours confirming a null result at rates already known to do nothing,
`POISON_RATES` in `src/nids/experiments/run_all.py` was changed to:

```python
POISON_RATES = [0.05, 0.20, 0.35, 0.50]
```

0.05 is the low anchor, 0.20 preserves continuity with the lost partial result and with the
literature's usual ceiling, and 0.35 / 0.50 search for the threshold at which damage begins.
The redundant 0.10 was dropped. The aim is to convert a null result into a quantified
attacker requirement: *poisoning a deep NIDS requires corrupting more than N% of training
labels* — a far stronger assumption than the literature usually grants.

Rates are fractions of the **whole** training set, so the maximum achievable rate is
class-bounded. Poisoning is applied to the *training split*, not the whole `KDDTrain+`
file: the 125,973 rows are divided into ~107,076 train and ~18,897 validation, and the
train split holds 49,835 attack rows. `attack_to_benign` therefore saturates at an
effective 0.465 and `benign_to_attack` at 0.535. `label_flip_poison` and `clean_label_poison` both clamp to
their eligible pool and record the achieved figure as `effective_rate` in
`PoisonResult.details`, which the run log prints per cell. A requested 0.50 under
`attack_to_benign` is therefore reported honestly as 0.465, and is a meaningful endpoint:
*every* attack label flipped.

## Changes made 2026-08-25 before launching

1. `POISON_RATES` changed as above (`src/nids/experiments/run_all.py`).
2. **Checkpointing added to phases 4 and 5.** Previously both wrote their JSON only after
   every retrain finished, which is why the earlier interruption cost the whole phase.
   Phase 4 now saves after each of its 64 cells and phase 5 after each architecture, and
   both resume from the partial file. `save_json` also writes via a temporary file and
   `os.replace`, so a crash mid-write cannot truncate a checkpoint.
3. **Bug fixed in `src/nids/attacks/poisoning.py`:** `clean_label_poison` did not record
   `effective_rate` / `requested_rate` in its `details` dict, unlike `label_flip_poison`.
   The new per-cell log line reads that key, so the run would have died with a `KeyError`
   at cell 13 — roughly two hours in. Caught by a subset smoke run before launch; both
   functions now report the same fields.
4. `docs/part3_results.md` §19 rewritten. It previously quoted preliminary evasion figures
   ("100% collapses to 66.7%") that the completed phase-2 sweep superseded, and it
   generalised the CNN's constrained vulnerability to all four architectures — the CNN is
   at 0.716 at ε=0.1 where the others are 0.058–0.084.

**Verification before launch:** `tests/test_constraints.py` 10/10 passing; a
`--subset 3000 --epochs 1` run of phases 1 and 4 exercised all 64 cells, the clamping, the
clean-label branch and the resume path against a scratch output directory.

---

## Blocked on you

Cross-domain transferability (half of RO1) needs two datasets behind registration forms:

* **UNSW-NB15** — https://research.unsw.edu.au/projects/unsw-nb15-dataset
  Download the *partitioned* CSVs: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
  → place in `D:\NIDS\data\raw\unsw_nb15\`
* **CICIDS2017** — https://www.unb.ca/cic/datasets/ids-2017.html
  Download `MachineLearningCSV.zip` (8 CSVs), **not** the raw PCAPs
  → extract to `D:\NIDS\data\raw\cicids2017\`

Then:

```bash
PYTHONPATH=/d/NIDS/src /d/NIDS/.venv/Scripts/python.exe -m nids.experiments.cross_domain \
    --nsl-kdd data/raw/nsl_kdd \
    --unsw-nb15 data/raw/unsw_nb15 \
    --cicids2017 data/raw/cicids2017 \
    --epochs 20
```

Note in advance: the genuine three-way shared feature space is **five columns**
(`duration`, `protocol_type`, `src_bytes`, `dst_bytes`, `dst_host_count`). Cross-domain
accuracy will be low, and that low number is the finding — do not be tempted to zero-fill
absent features or fine-tune on the target, both of which invalidate it (see
`experiments/cross_domain.py` for why).

CICIDS2017 is ~2.8M rows and slow on CPU; `notebooks/NIDS_GPU_Experiments.ipynb` runs the
same pipeline on Colab GPU.

---

## Remaining work after phases 4–5 finish

1. Write §15 (poisoning) and §16 (defence) of `docs/part3_results.md` — ~45 min.
2. `python -m nids.eval.figures --dataset nsl_kdd` → regenerates all figures/tables.
3. `python scripts/build_report.py` → reassembles the single report.
4. Complete §19 conclusions.

---

## Environment

* venv: `D:\NIDS\.venv` (torch 2.13.0+cpu, numpy 2.4.6, pandas 3.0.5, sklearn 1.9.0)
* Always run with `PYTHONPATH=/d/NIDS/src`
* `python3` on this machine is a Windows Store stub — use the venv python or `python`
* Data: `data/raw/nsl_kdd/` (KDDTrain+.txt 125,973 rows; KDDTest+.txt 22,544 rows)
