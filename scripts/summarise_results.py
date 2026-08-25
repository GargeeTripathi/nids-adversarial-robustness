"""
Condense the experiment JSON into the handful of numbers the write-up actually needs.

`run_all.py` emits a lot of detail (every architecture x attack x regime x epsilon). This
extracts the headline findings and prints them as markdown tables ready to paste into the
report.

    .venv/Scripts/python.exe scripts/summarise_results.py --dataset nsl_kdd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHITECTURES = ["cnn", "lstm", "transformer", "gnn"]
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "gnn": "GNN"}


def load(metric_dir: Path, dataset: str, phase: str):
    path = metric_dir / f"{dataset}_{phase}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list]) -> str:
    def fmt(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def summarise(dataset: str, output_dir: str) -> None:
    metric_dir = Path(output_dir) / "metrics"
    print(f"# Results summary — {dataset}\n")

    # ---- phase 1 --------------------------------------------------------------------
    baseline = load(metric_dir, dataset, "phase1_baseline")
    if baseline:
        print("## Baseline performance\n")
        rows = []
        for a in ARCHITECTURES:
            if a not in baseline:
                continue
            entry = baseline[a]
            rows.append([
                DISPLAY[a], f"{entry['parameters']:,}", entry["epochs_run"],
                entry["validation"]["accuracy"], entry["test"]["accuracy"],
                entry["validation"]["accuracy"] - entry["test"]["accuracy"],
                entry["test"]["detection_rate"], entry["test"]["false_positive_rate"],
                entry["test"].get("roc_auc", float("nan")),
            ])
        print(table(["Model", "Params", "Epochs", "Val acc", "Test acc",
                     "Gap", "DR", "FPR", "AUC"], rows))
        print()

        gaps = [r[5] for r in rows]
        print(f"*Mean validation→test gap: {sum(gaps)/len(gaps):.4f}* — KDDTest+ contains "
              f"attack families absent from training.\n")

    # ---- phase 2 --------------------------------------------------------------------
    evasion = load(metric_dir, dataset, "phase2_evasion")
    if evasion:
        available = [a for a in ARCHITECTURES if a in evasion]
        epsilons = sorted(evasion[available[0]]["pgd_constrained"], key=float)
        reference = "0.1" if "0.1" in epsilons else epsilons[-1]

        print(f"## Evasion — constrained vs unconstrained (PGD, ε={reference})\n")
        rows = []
        for a in available:
            loose = evasion[a]["pgd_unconstrained"][reference]
            tight = evasion[a]["pgd_constrained"][reference]
            feas = loose.get("feasibility", {})
            rows.append([
                DISPLAY[a],
                loose["attack_success_rate"],
                tight["attack_success_rate"],
                loose["attack_success_rate"] - tight["attack_success_rate"],
                loose["attack_success_rate"] / max(tight["attack_success_rate"], 1e-9),
                feas.get("any_violation_rate_max", float("nan")),
            ])
        print(table(["Model", "ASR unconstrained", "ASR constrained",
                     "Absolute drop", "Overstatement ×", "Infeasible rate"], rows))
        print()

        print(f"## Attack success vs ε (PGD, constrained)\n")
        rows = []
        for a in available:
            rows.append([DISPLAY[a]] +
                        [evasion[a]["pgd_constrained"][e]["attack_success_rate"]
                         for e in epsilons])
        print(table(["Model"] + [f"ε={e}" for e in epsilons], rows))
        print()

        print("## Why unconstrained adversarial flows are unrealizable\n")
        rows = []
        for a in available:
            feas = evasion[a]["pgd_unconstrained"][reference].get("feasibility", {})
            rows.append([
                DISPLAY[a],
                feas.get("immutable_violation_rate", float("nan")),
                feas.get("direction_violation_rate", float("nan")),
                feas.get("onehot_violation_rate", float("nan")),
                feas.get("integrality_violation_rate", float("nan")),
            ])
        print(table(["Model", "Changed immutable", "Decreased increase-only",
                     "Broke one-hot", "Non-integer count"], rows))
        print()

    # ---- phase 3 --------------------------------------------------------------------
    transfer = load(metric_dir, dataset, "phase3_transfer")
    if transfer:
        for regime in ("unconstrained", "constrained"):
            if regime not in transfer:
                continue
            print(f"## Black-box transfer — {regime}\n")
            print("Rows = surrogate (attacker's model), columns = target (deployed).\n")
            rows = []
            for s in ARCHITECTURES:
                if s not in transfer[regime]:
                    continue
                rows.append([DISPLAY[s]] +
                            [transfer[regime][s][t]["target_success_rate"]
                             for t in ARCHITECTURES if t in transfer[regime][s]])
            print(table(["Surrogate ↓ / Target →"] +
                        [DISPLAY[t] for t in ARCHITECTURES if t in transfer[regime]], rows))
            print()

            diagonal, off_diagonal = [], []
            for s in ARCHITECTURES:
                if s not in transfer[regime]:
                    continue
                for t in ARCHITECTURES:
                    if t not in transfer[regime][s]:
                        continue
                    value = transfer[regime][s][t]["target_success_rate"]
                    (diagonal if s == t else off_diagonal).append(value)
            if diagonal and off_diagonal:
                print(f"*White-box (diagonal) mean: {sum(diagonal)/len(diagonal):.4f}; "
                      f"transfer (off-diagonal) mean: "
                      f"{sum(off_diagonal)/len(off_diagonal):.4f}*\n")

    # ---- phase 4 --------------------------------------------------------------------
    poisoning = load(metric_dir, dataset, "phase4_poisoning")
    if poisoning:
        available = [a for a in ARCHITECTURES if a in poisoning]
        strategies = [s for s in ("label_flip_attack_to_benign", "label_flip_benign_to_attack",
                                  "label_flip_symmetric", "clean_label")
                      if s in poisoning[available[0]]]
        rates = sorted(poisoning[available[0]][strategies[0]], key=float)

        print("## Poisoning — detection-rate drop on clean test data\n")
        for strategy in strategies:
            print(f"**{strategy.replace('_', ' ')}**\n")
            rows = []
            for a in available:
                rows.append([DISPLAY[a]] +
                            [poisoning[a][strategy][r]["detection_rate_drop"] for r in rates])
            print(table(["Model"] + [f"rate={r}" for r in rates], rows))
            print()

    # ---- phase 5 --------------------------------------------------------------------
    defence = load(metric_dir, dataset, "phase5_defence")
    if defence:
        print("## Adversarial training defence\n")
        rows = []
        for a in ARCHITECTURES:
            if a not in defence:
                continue
            entry = defence[a]
            rows.append([
                DISPLAY[a],
                entry["undefended_attack_success_rate"],
                entry["defended_attack_success_rate"],
                entry["attack_success_reduction"],
                entry["clean_accuracy_cost"],
            ])
        print(table(["Model", "ASR undefended", "ASR defended",
                     "Reduction", "Clean accuracy cost"], rows))
        print()

    # ---- cross-domain ----------------------------------------------------------------
    cross = None
    cross_path = metric_dir / "cross_domain.json"
    if cross_path.exists():
        cross = json.loads(cross_path.read_text(encoding="utf-8"))
    if cross:
        meta = cross.get("_meta", {})
        print("## Cross-domain transfer\n")
        print(f"Shared feature space ({len(meta.get('shared_features', []))}): "
              f"{meta.get('shared_features')}\n")
        datasets = meta.get("datasets", [])
        for architecture in ARCHITECTURES:
            if architecture not in cross:
                continue
            print(f"**{DISPLAY[architecture]}**\n")
            rows = []
            for source in datasets:
                if source not in cross[architecture]:
                    continue
                rows.append([source] +
                            [cross[architecture][source][t]["accuracy"] for t in datasets])
            print(table(["Train ↓ / Test →"] + datasets, rows))
            print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nsl_kdd")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    summarise(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
