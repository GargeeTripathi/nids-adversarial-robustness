"""
Full experiment driver.

Runs the complete study for one dataset and writes every artefact to `outputs/`:

    Phase 1  baseline training        4 architectures, clean data
    Phase 2  white-box evasion        FGSM + PGD, constrained vs unconstrained, eps sweep
    Phase 3  black-box transfer       every (surrogate, target) architecture pair
    Phase 4  poisoning                label-flip (3 strategies) + clean-label, rate sweep
    Phase 5  adversarial training     the standard defence baseline

Designed to be resumable: each phase writes its results as soon as it finishes, and
completed phases are skipped on re-run unless `--force` is passed. On CPU the full suite
takes hours, so a crash in phase 4 must not cost the phase 1-3 work.

Usage:
    python -m nids.experiments.run_all --dataset nsl_kdd --data-dir data/raw/nsl_kdd
    python -m nids.experiments.run_all --phases 1,2 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..attacks.constraints import ConstraintConfig, ConstraintSet
from ..attacks.evasion import evaluate_evasion, fgsm_attack, pgd_attack, transfer_attack
from ..attacks.poisoning import clean_label_poison, evaluate_poisoning, label_flip_poison
from ..data.loaders import load_dataset
from ..data.preprocess import prepare
from ..eval.metrics import compute_metrics, per_attack_breakdown, robustness_score
from ..models.architectures import ARCHITECTURES, build_model, count_parameters
from ..models.trainer import TrainConfig, predict, train_model

ARCHITECTURE_NAMES = ["cnn", "lstm", "transformer", "gnn"]
EPSILONS = [0.01, 0.05, 0.1, 0.2]
POISON_RATES = [0.05, 0.20, 0.35, 0.50]
# 0.05 low anchor / 0.20 continuity with the literature's usual ceiling /
# 0.35, 0.50 threshold search. Rates are fractions of the whole training set, so
# `attack_to_benign` saturates at 0.465 (every attack label flipped) and
# label_flip_poison clamps to that, recording the achieved `effective_rate`.


class ExperimentRunner:
    def __init__(self, dataset_name: str, data_dir: str, output_dir: str = "outputs",
                 epochs: int = 20, batch_size: int = 512, seed: int = 42,
                 device: str = "cpu", subset: int | None = None):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.subset = subset

        for sub in ("models", "metrics", "figures", "logs", "adversarial"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        self.bundle: dict | None = None
        self.models: dict[str, torch.nn.Module] = {}
        self.constraints: dict[str, ConstraintSet] = {}

    # ---- infrastructure ---------------------------------------------------------------

    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with (self.output_dir / "logs" / f"{self.dataset_name}_run.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_json(self, obj, name: str, quiet: bool = False) -> Path:
        """Write atomically: a crash mid-write must not truncate a checkpoint."""
        path = self.output_dir / "metrics" / f"{self.dataset_name}_{name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=_json_safe), encoding="utf-8")
        tmp.replace(path)
        if not quiet:
            self.log(f"  wrote {path.name}")
        return path

    def load_json(self, name: str):
        path = self.output_dir / "metrics" / f"{self.dataset_name}_{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def train_config(self, verbose: bool = False) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs, batch_size=self.batch_size,
            device=self.device, seed=self.seed, verbose=verbose,
        )

    # ---- setup -------------------------------------------------------------------------

    def setup(self) -> None:
        self.log(f"Loading {self.dataset_name} from {self.data_dir}")
        dataset = load_dataset(self.dataset_name, self.data_dir)
        self.log(dataset.summary().replace("\n", "\n           "))

        self.bundle = prepare(dataset, seed=self.seed)

        if self.subset:
            for key in ("x_train", "y_train"):
                self.bundle[key] = self.bundle[key][: self.subset]
            self.log(f"  SUBSET MODE: training on {len(self.bundle['y_train'])} rows")

        self.constraints = {
            "unconstrained": ConstraintSet(self.bundle["preprocessor"],
                                           ConstraintConfig.unconstrained()),
            "constrained": ConstraintSet(self.bundle["preprocessor"],
                                         ConstraintConfig.realistic()),
        }
        self.log(f"  encoded features: {self.bundle['n_features']}")
        self.log(f"  dependency rules: "
                 f"{[r['name'] for r in self.constraints['constrained'].dependency_rules]}")

    def model_path(self, architecture: str, tag: str = "baseline") -> Path:
        return self.output_dir / "models" / f"{self.dataset_name}_{architecture}_{tag}.pt"

    # ---- phase 1: baselines --------------------------------------------------------------

    def phase1_baselines(self, force: bool = False) -> dict:
        self.log("=" * 70)
        self.log("PHASE 1  baseline training")
        self.log("=" * 70)

        existing = self.load_json("phase1_baseline")
        results = existing or {}

        for architecture in ARCHITECTURE_NAMES:
            path = self.model_path(architecture)
            model = build_model(architecture, self.bundle["n_features"])
            cached = path.exists() and not force

            if cached:
                model.load_state_dict(torch.load(path, map_location=self.device))
                self.models[architecture] = model
                self.log(f"  {architecture}: loaded cached weights")
                train_seconds = results.get(architecture, {}).get("train_seconds", 0.0)
                epochs_run = results.get(architecture, {}).get("epochs_run", 0)
                best_epoch = results.get(architecture, {}).get("best_epoch", 0)
            else:
                self.log(f"  {architecture}: training "
                         f"({count_parameters(model):,} parameters)")
                started = time.time()
                model, history = train_model(
                    model,
                    self.bundle["x_train"], self.bundle["y_train"],
                    self.bundle["x_val"], self.bundle["y_val"],
                    self.train_config(),
                )
                torch.save(model.state_dict(), path)
                self.models[architecture] = model
                train_seconds = round(time.time() - started, 1)
                epochs_run, best_epoch = history.epochs_run, history.best_epoch

            # Evaluation runs for cached models too - it costs seconds, and skipping it
            # left the metrics JSON missing every architecture that was resumed from disk.
            y_pred, y_score = predict(model, self.bundle["x_test"], self.device)
            metrics = compute_metrics(self.bundle["y_test"], y_pred, y_score)
            y_pred_val, y_score_val = predict(model, self.bundle["x_val"], self.device)
            val_metrics = compute_metrics(self.bundle["y_val"], y_pred_val, y_score_val)

            results[architecture] = {
                "parameters": count_parameters(model),
                "epochs_run": epochs_run,
                "best_epoch": best_epoch,
                "train_seconds": train_seconds,
                "resumed_from_cache": cached,
                "test": metrics,
                "validation": val_metrics,
                "per_attack": per_attack_breakdown(
                    self.bundle["y_test"], y_pred, self.bundle["attack_category_test"]
                ),
            }
            self.log(f"    val acc {val_metrics['accuracy']:.4f} | "
                     f"test acc {metrics['accuracy']:.4f} "
                     f"DR {metrics['detection_rate']:.4f} "
                     f"FPR {metrics['false_positive_rate']:.4f}"
                     + (f" ({train_seconds}s)" if not cached else " (cached)"))

            # Written after every architecture, not once at the end: a kill during the
            # Transformer previously discarded the CNN and LSTM results entirely.
            self.save_json(results, "phase1_baseline")

        return results

    # ---- phase 2: white-box evasion -------------------------------------------------------

    def phase2_evasion(self, force: bool = False) -> dict:
        self.log("=" * 70)
        self.log("PHASE 2  white-box evasion (FGSM, PGD) x (constrained, unconstrained)")
        self.log("=" * 70)

        if (existing := self.load_json("phase2_evasion")) and not force:
            self.log("  cached, skipping")
            return existing

        results: dict = {}
        x_test, y_test = self.bundle["x_test"], self.bundle["y_test"]

        for architecture in ARCHITECTURE_NAMES:
            model = self.models[architecture]
            results[architecture] = {}

            for attack_name, attack_fn, extra in (
                ("fgsm", fgsm_attack, {}),
                ("pgd", pgd_attack, {"steps": 20}),
            ):
                for regime, constraint_set in self.constraints.items():
                    key = f"{attack_name}_{regime}"
                    results[architecture][key] = {}

                    for epsilon in EPSILONS:
                        outcome = evaluate_evasion(
                            model, x_test, y_test, attack_fn,
                            {"epsilon": epsilon, "constraints": constraint_set, **extra},
                            batch_size=self.batch_size, device=self.device,
                        )
                        x_adv = outcome.pop("x_adv")

                        # Feasibility is measured against the *realistic* rule set no
                        # matter which regime produced the examples - that is what makes
                        # the two regimes comparable.
                        attack_mask = y_test == 1
                        report = self.constraints["constrained"].feasibility_report(
                            torch.tensor(x_adv),
                            torch.tensor(x_test[attack_mask]),
                        )
                        outcome["feasibility"] = report
                        results[architecture][key][str(epsilon)] = outcome

                        if epsilon == 0.1:
                            np.save(
                                self.output_dir / "adversarial"
                                / f"{self.dataset_name}_{architecture}_{key}_eps0.1.npy",
                                x_adv[:1000],
                            )

                    rates = [
                        results[architecture][key][str(e)]["attack_success_rate"]
                        for e in EPSILONS
                    ]
                    self.log(f"  {architecture:12s} {key:22s} ASR by eps: "
                             + " ".join(f"{r:.3f}" for r in rates))

        self.save_json(results, "phase2_evasion")
        return results

    # ---- phase 3: black-box transfer -------------------------------------------------------

    def phase3_transfer(self, force: bool = False, epsilon: float = 0.1) -> dict:
        self.log("=" * 70)
        self.log("PHASE 3  black-box transfer across architectures")
        self.log("=" * 70)

        if (existing := self.load_json("phase3_transfer")) and not force:
            self.log("  cached, skipping")
            return existing

        # Restrict to genuine attack samples and cap the count: the transfer matrix is
        # 16 pairs x 2 regimes, and PGD on the full test set for each is needlessly slow
        # for a quantity that is stable at this sample size.
        mask = self.bundle["y_test"] == 1
        x = torch.tensor(self.bundle["x_test"][mask][:2000], dtype=torch.float32)
        y = torch.tensor(self.bundle["y_test"][mask][:2000], dtype=torch.long)

        results: dict = {}
        for regime, constraint_set in self.constraints.items():
            results[regime] = {}
            for surrogate in ARCHITECTURE_NAMES:
                results[regime][surrogate] = {}
                for target in ARCHITECTURE_NAMES:
                    _, info = transfer_attack(
                        self.models[surrogate], self.models[target], x, y,
                        epsilon=epsilon, steps=20,
                        constraints=constraint_set, device=self.device,
                    )
                    results[regime][surrogate][target] = info

                row = " ".join(
                    f"{results[regime][surrogate][t]['target_success_rate']:.3f}"
                    for t in ARCHITECTURE_NAMES
                )
                self.log(f"  [{regime:13s}] {surrogate:12s} -> {row}")

        results["_meta"] = {
            "epsilon": epsilon, "n_samples": int(len(x)),
            "column_order": ARCHITECTURE_NAMES,
            "note": "rows = surrogate (attacker's model), columns = target (deployed model)",
        }
        self.save_json(results, "phase3_transfer")
        return results

    # ---- phase 4: poisoning ------------------------------------------------------------------

    def phase4_poisoning(self, force: bool = False) -> dict:
        self.log("=" * 70)
        self.log("PHASE 4  poisoning (label-flip + clean-label)")
        self.log("=" * 70)

        baseline = self.load_json("phase1_baseline")
        conditions = [f"label_flip_{s}" for s in
                      ("attack_to_benign", "benign_to_attack", "symmetric")] + ["clean_label"]

        # Resume support. Each (architecture, condition, rate) cell costs one full retrain -
        # up to 20 minutes for the transformer - so the partial result is checkpointed after
        # every cell and reloaded here. A crash or interrupt therefore costs one cell, not
        # the whole phase. `--force` discards the checkpoint and recomputes everything.
        results: dict = {} if force else (self.load_json("phase4_poisoning") or {})
        todo = sum(1 for a in ARCHITECTURE_NAMES for c in conditions for r in POISON_RATES
                   if str(r) not in results.get(a, {}).get(c, {}))
        total = len(ARCHITECTURE_NAMES) * len(conditions) * len(POISON_RATES)
        if todo == 0:
            self.log("  cached, skipping")
            return results
        self.log(f"  {total - todo}/{total} cells cached, {todo} to run")

        done = 0
        for architecture in ARCHITECTURE_NAMES:
            cells = results.setdefault(architecture, {})
            clean_metrics = baseline[architecture]["test"]

            for condition in conditions:
                cells.setdefault(condition, {})
                for rate in POISON_RATES:
                    if str(rate) in cells[condition]:
                        continue
                    started = time.time()
                    if condition == "clean_label":
                        poison = clean_label_poison(
                            self.models[architecture],
                            self.bundle["x_train"], self.bundle["y_train"],
                            poison_rate=rate, epsilon=0.1, steps=30,
                            constraints=self.constraints["constrained"],
                            device=self.device, seed=self.seed,
                        )
                    else:
                        poison = label_flip_poison(
                            self.bundle["x_train"], self.bundle["y_train"],
                            poison_rate=rate,
                            strategy=condition[len("label_flip_"):], seed=self.seed,
                        )
                    metrics = self._train_and_evaluate(architecture, poison)
                    cells[condition][str(rate)] = (
                        evaluate_poisoning(clean_metrics, metrics, poison)
                    )
                    self.save_json(results, "phase4_poisoning", quiet=True)
                    done += 1
                    self.log(f"    [{done}/{todo}] {architecture:12s} {condition:24s} "
                             f"rate {rate:.2f} (eff {poison.details['effective_rate']:.3f}) "
                             f"DR drop {cells[condition][str(rate)]['detection_rate_drop']:+.3f} "
                             f"({time.time() - started:.0f}s)")

                drops = [cells[condition][str(r)]["detection_rate_drop"] for r in POISON_RATES]
                self.log(f"  {architecture:12s} {condition:24s} DR drop: "
                         + " ".join(f"{d:+.3f}" for d in drops))

        self.save_json(results, "phase4_poisoning")
        return results

    def _train_and_evaluate(self, architecture: str, poison) -> dict:
        """Retrain from scratch on poisoned data, evaluate on the clean test set."""
        model = build_model(architecture, self.bundle["n_features"])
        model, _ = train_model(
            model,
            poison.x_poisoned, poison.y_poisoned,
            self.bundle["x_val"], self.bundle["y_val"],   # clean validation, as a defender would have
            self.train_config(),
        )
        y_pred, y_score = predict(model, self.bundle["x_test"], self.device)
        return compute_metrics(self.bundle["y_test"], y_pred, y_score)

    # ---- phase 5: adversarial training -------------------------------------------------------

    def phase5_defence(self, force: bool = False, epsilon: float = 0.1) -> dict:
        self.log("=" * 70)
        self.log("PHASE 5  adversarial training defence")
        self.log("=" * 70)

        # Same resume contract as phase 4, at architecture granularity: adversarial
        # training is one long run per architecture, so there is no finer checkpoint.
        results: dict = {} if force else (self.load_json("phase5_defence") or {})
        if all(a in results for a in ARCHITECTURE_NAMES):
            self.log("  cached, skipping")
            return results

        constraint_set = self.constraints["constrained"]

        def attack(model, xb, yb):
            return pgd_attack(model, xb, yb, epsilon=epsilon, steps=7,
                              constraints=constraint_set, device=self.device)

        for architecture in ARCHITECTURE_NAMES:
            if architecture in results:
                self.log(f"  {architecture}: cached, skipping")
                continue
            self.log(f"  {architecture}: adversarial training")
            model = build_model(architecture, self.bundle["n_features"])
            model, _ = train_model(
                model,
                self.bundle["x_train"], self.bundle["y_train"],
                self.bundle["x_val"], self.bundle["y_val"],
                self.train_config(),
                adversarial_config={"attack": attack, "ratio": 0.5},
            )
            torch.save(model.state_dict(), self.model_path(architecture, "advtrain"))

            y_pred, y_score = predict(model, self.bundle["x_test"], self.device)
            clean_metrics = compute_metrics(self.bundle["y_test"], y_pred, y_score)

            robust = evaluate_evasion(
                model, self.bundle["x_test"], self.bundle["y_test"], pgd_attack,
                {"epsilon": epsilon, "steps": 20, "constraints": constraint_set},
                batch_size=self.batch_size, device=self.device,
            )
            robust.pop("x_adv")

            undefended = self.load_json("phase2_evasion")[architecture]["pgd_constrained"][str(epsilon)]
            results[architecture] = {
                "defended_clean": clean_metrics,
                "defended_attack_success_rate": robust["attack_success_rate"],
                "undefended_attack_success_rate": undefended["attack_success_rate"],
                "attack_success_reduction": (
                    undefended["attack_success_rate"] - robust["attack_success_rate"]
                ),
                "clean_accuracy_cost": (
                    self.load_json("phase1_baseline")[architecture]["test"]["accuracy"]
                    - clean_metrics["accuracy"]
                ),
            }
            self.log(f"    ASR {undefended['attack_success_rate']:.3f} -> "
                     f"{robust['attack_success_rate']:.3f} | "
                     f"clean accuracy cost {results[architecture]['clean_accuracy_cost']:+.4f}")
            self.save_json(results, "phase5_defence", quiet=True)

        self.save_json(results, "phase5_defence")
        return results

    # ---- orchestration --------------------------------------------------------------------

    def run(self, phases: list[int], force: bool = False) -> dict:
        started = time.time()
        self.setup()

        all_results = {}
        if 1 in phases:
            all_results["baseline"] = self.phase1_baselines(force)
        else:
            for architecture in ARCHITECTURE_NAMES:      # later phases need loaded models
                model = build_model(architecture, self.bundle["n_features"])
                model.load_state_dict(
                    torch.load(self.model_path(architecture), map_location=self.device)
                )
                self.models[architecture] = model

        if 2 in phases:
            all_results["evasion"] = self.phase2_evasion(force)
        if 3 in phases:
            all_results["transfer"] = self.phase3_transfer(force)
        if 4 in phases:
            all_results["poisoning"] = self.phase4_poisoning(force)
        if 5 in phases:
            all_results["defence"] = self.phase5_defence(force)

        self.log(f"COMPLETE in {(time.time() - started)/60:.1f} minutes")
        return all_results


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIDS adversarial robustness experiments")
    parser.add_argument("--dataset", default="nsl_kdd", choices=sorted(ARCHITECTURES) and
                        ["nsl_kdd", "unsw_nb15", "cicids2017"])
    parser.add_argument("--data-dir", default="data/raw/nsl_kdd")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subset", type=int, default=None,
                        help="train on only the first N rows (for quick checks)")
    parser.add_argument("--phases", default="1,2,3,4,5")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runner = ExperimentRunner(
        dataset_name=args.dataset, data_dir=args.data_dir, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=args.device, subset=args.subset,
    )
    runner.run([int(p) for p in args.phases.split(",")], force=args.force)


if __name__ == "__main__":
    main()
