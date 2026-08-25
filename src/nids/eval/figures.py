"""
Turn experiment JSON into report-ready tables and figures.

Kept separate from the experiment runner so that plots can be regenerated and restyled
without re-running hours of training.

    python -m nids.eval.figures --dataset nsl_kdd --output-dir outputs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # no display in a background/CI context
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHITECTURES = ["cnn", "lstm", "transformer", "gnn"]
DISPLAY_NAME = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "gnn": "GNN"}
PALETTE = {"cnn": "#4C72B0", "lstm": "#DD8452", "transformer": "#55A868", "gnn": "#C44E52"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class FigureBuilder:
    def __init__(self, dataset: str, output_dir: str = "outputs"):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.metric_dir = self.output_dir / "metrics"
        self.figure_dir = self.output_dir / "figures"
        self.table_dir = self.output_dir / "tables"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)

    def load(self, phase: str):
        path = self.metric_dir / f"{self.dataset}_{phase}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_table(self, frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(self.table_dir / f"{self.dataset}_{name}.csv")
        (self.table_dir / f"{self.dataset}_{name}.md").write_text(
            frame.to_markdown(), encoding="utf-8"
        )
        print(f"  table: {name}")

    def save_figure(self, name: str) -> None:
        plt.savefig(self.figure_dir / f"{self.dataset}_{name}.png")
        plt.close()
        print(f"  figure: {name}")

    # ---- phase 1 ----------------------------------------------------------------------

    def baseline(self) -> pd.DataFrame | None:
        data = self.load("phase1_baseline")
        if not data:
            return None

        frame = pd.DataFrame({
            DISPLAY_NAME[a]: {
                "Parameters": data[a]["parameters"],
                "Epochs": data[a]["epochs_run"],
                "Train (s)": data[a]["train_seconds"],
                "Val accuracy": data[a]["validation"]["accuracy"],
                "Test accuracy": data[a]["test"]["accuracy"],
                "Detection rate": data[a]["test"]["detection_rate"],
                "False positive rate": data[a]["test"]["false_positive_rate"],
                "F1": data[a]["test"]["f1"],
                "ROC-AUC": data[a]["test"].get("roc_auc", float("nan")),
            }
            for a in ARCHITECTURES if a in data
        }).T
        self.save_table(frame.round(4), "table1_baseline")

        # Validation-vs-test gap: the zero-day generalisation story.
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        names = [DISPLAY_NAME[a] for a in ARCHITECTURES if a in data]
        val = [data[a]["validation"]["accuracy"] for a in ARCHITECTURES if a in data]
        test = [data[a]["test"]["accuracy"] for a in ARCHITECTURES if a in data]

        x = np.arange(len(names))
        axes[0].bar(x - 0.2, val, 0.4, label="Validation (same distribution)", color="#4C72B0")
        axes[0].bar(x + 0.2, test, 0.4, label="Official KDDTest+ (novel attacks)", color="#C44E52")
        axes[0].set_xticks(x); axes[0].set_xticklabels(names)
        axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Generalisation gap to unseen attack families")
        axes[0].legend(fontsize=8)
        for i, (v, t) in enumerate(zip(val, test)):
            axes[0].text(i, max(v, t) + 0.02, f"−{v-t:.3f}", ha="center", fontsize=8)

        dr = [data[a]["test"]["detection_rate"] for a in ARCHITECTURES if a in data]
        fpr = [data[a]["test"]["false_positive_rate"] for a in ARCHITECTURES if a in data]
        axes[1].scatter(fpr, dr, s=140,
                        c=[PALETTE[a] for a in ARCHITECTURES if a in data], zorder=3)
        for i, name in enumerate(names):
            axes[1].annotate(name, (fpr[i], dr[i]),
                             textcoords="offset points", xytext=(8, 4), fontsize=9)
        axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("Detection rate")
        axes[1].set_title("Operating points (upper-left is better)")

        plt.tight_layout()
        self.save_figure("fig1_baseline")
        return frame

    # ---- phase 2 ----------------------------------------------------------------------

    def evasion(self) -> pd.DataFrame | None:
        data = self.load("phase2_evasion")
        if not data:
            return None

        architectures = [a for a in ARCHITECTURES if a in data]
        epsilons = sorted(
            (float(e) for e in data[architectures[0]]["pgd_constrained"]),
        )

        # --- ASR vs epsilon, constrained vs unconstrained -------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for attack, ax in zip(("fgsm", "pgd"), axes):
            for architecture in architectures:
                for regime, style in (("unconstrained", "--"), ("constrained", "-")):
                    entry = data[architecture][f"{attack}_{regime}"]
                    rates = [entry[str(e)]["attack_success_rate"] for e in epsilons]
                    ax.plot(epsilons, rates, style, marker="o", markersize=4,
                            color=PALETTE[architecture],
                            label=f"{DISPLAY_NAME[architecture]} ({regime[:5]})")
            ax.set_xlabel("Perturbation budget ε"); ax.set_title(attack.upper())
            ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("Attack success rate")
        axes[1].legend(fontsize=7, ncol=2, loc="upper left")
        plt.suptitle("Evasion: dashed = unconstrained (unrealizable), solid = realistic constraints")
        plt.tight_layout()
        self.save_figure("fig2_evasion_epsilon")

        # --- headline comparison table at eps=0.1 ---------------------------------------
        reference = "0.1" if "0.1" in data[architectures[0]]["pgd_constrained"] else str(epsilons[-1])
        rows = {}
        for architecture in architectures:
            unconstrained = data[architecture]["pgd_unconstrained"][reference]
            constrained = data[architecture]["pgd_constrained"][reference]
            feasibility = unconstrained.get("feasibility", {})
            rows[DISPLAY_NAME[architecture]] = {
                "ASR unconstrained": unconstrained["attack_success_rate"],
                "ASR constrained": constrained["attack_success_rate"],
                "Absolute reduction": (unconstrained["attack_success_rate"]
                                       - constrained["attack_success_rate"]),
                "Overstatement factor": (unconstrained["attack_success_rate"]
                                         / max(constrained["attack_success_rate"], 1e-9)),
                "Immutable violated": feasibility.get("immutable_violation_rate", float("nan")),
                "Direction violated": feasibility.get("direction_violation_rate", float("nan")),
                "One-hot violated": feasibility.get("onehot_violation_rate", float("nan")),
            }
        frame = pd.DataFrame(rows).T
        self.save_table(frame.round(4), f"table2_evasion_eps{reference}")

        # --- feasibility violations of unconstrained attacks -----------------------------
        violation_types = ["immutable_violation_rate", "direction_violation_rate",
                           "onehot_violation_rate", "integrality_violation_rate"]
        labels = ["Immutable\nfeature changed", "Decreased an\nincrease-only feature",
                  "Broke one-hot\nencoding", "Non-integer\ncount"]

        values = np.array([
            [data[a]["pgd_unconstrained"][reference].get("feasibility", {}).get(v, 0.0)
             for v in violation_types]
            for a in architectures
        ])

        plt.figure(figsize=(9, 4.5))
        width = 0.8 / len(architectures)
        x = np.arange(len(violation_types))
        for i, architecture in enumerate(architectures):
            plt.bar(x + i * width - 0.4 + width / 2, values[i], width,
                    label=DISPLAY_NAME[architecture], color=PALETTE[architecture])
        plt.xticks(x, labels, fontsize=8)
        plt.ylabel("Fraction of adversarial examples violating")
        plt.title(f"Why unconstrained adversarial flows cannot exist (PGD, ε={reference})")
        plt.ylim(0, 1.05)
        plt.legend(fontsize=8)
        plt.tight_layout()
        self.save_figure("fig3_feasibility_violations")
        return frame

    # ---- phase 3 ----------------------------------------------------------------------

    def transfer(self) -> pd.DataFrame | None:
        data = self.load("phase3_transfer")
        if not data:
            return None

        import seaborn as sns

        regimes = [r for r in ("unconstrained", "constrained") if r in data]
        fig, axes = plt.subplots(1, len(regimes), figsize=(6.5 * len(regimes), 5))
        if len(regimes) == 1:
            axes = [axes]

        frames = {}
        for ax, regime in zip(axes, regimes):
            matrix = np.array([
                [data[regime][s][t]["target_success_rate"] for t in ARCHITECTURES]
                for s in ARCHITECTURES
            ])
            frame = pd.DataFrame(
                matrix,
                index=[DISPLAY_NAME[a] for a in ARCHITECTURES],
                columns=[DISPLAY_NAME[a] for a in ARCHITECTURES],
            )
            frames[regime] = frame
            sns.heatmap(frame, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1,
                        ax=ax, cbar=regime == regimes[-1])
            ax.set_title(f"{regime.capitalize()}")
            ax.set_xlabel("Target (deployed model)")
            ax.set_ylabel("Surrogate (attacker's model)")

        plt.suptitle("Black-box transfer: attack success on an unseen target")
        plt.tight_layout()
        self.save_figure("fig4_transfer_matrix")

        for regime, frame in frames.items():
            self.save_table(frame.round(4), f"table3_transfer_{regime}")
        return frames.get("constrained")

    # ---- phase 4 ----------------------------------------------------------------------

    def poisoning(self) -> pd.DataFrame | None:
        data = self.load("phase4_poisoning")
        if not data:
            return None

        architectures = [a for a in ARCHITECTURES if a in data]
        strategies = ["label_flip_attack_to_benign", "label_flip_benign_to_attack",
                      "label_flip_symmetric", "clean_label"]
        strategy_labels = ["Label flip\n(attack→benign)", "Label flip\n(benign→attack)",
                           "Label flip\n(symmetric)", "Clean-label"]

        available = [s for s in strategies if s in data[architectures[0]]]
        rates = sorted({float(r) for a in architectures for s in available
                        for r in data[a].get(s, {})})

        # Detection-rate drop alone is not a sufficient damage measure here, and plotting it
        # alone would actively mislead. Measured on the CNN, attack->benign flipping at rate
        # 0.35 yields a detection-rate drop of -0.147 - which reads as "the poisoned model
        # detects better" - while the false-positive rate rises 0.026 -> 0.161. Poisoning
        # moved the operating point rather than degrading it, and for a NIDS a 6x
        # false-positive increase is severe operational damage that the headline metric
        # scores as a benefit. Both rows are therefore always plotted together.
        metrics = [("detection_rate_drop", "Detection-rate drop\n(higher = more damage)"),
                   ("false_positive_rate_change", "False-positive-rate change\n(higher = more alerts)")]

        fig, axes = plt.subplots(len(metrics), len(available),
                                 figsize=(4.2 * len(available), 3.6 * len(metrics)),
                                 # Share y within each row so magnitudes are comparable
                                 # across strategies: on independent axes a -0.015 change
                                 # and a +0.135 spike look equally dramatic, which is the
                                 # exact misreading this figure exists to prevent.
                                 sharex=True, sharey='row', squeeze=False)

        for row, (metric, ylabel) in enumerate(metrics):
            for col, strategy in enumerate(available):
                ax = axes[row][col]
                for architecture in architectures:
                    cells = data[architecture].get(strategy, {})
                    # The checkpoint may be partial while phase 4 is still running.
                    xs = [r for r in rates if str(r) in cells]
                    if not xs:
                        continue
                    ax.plot(xs, [cells[str(r)][metric] for r in xs], marker="o", markersize=4,
                            color=PALETTE[architecture], label=DISPLAY_NAME[architecture])
                ax.axhline(0, color="grey", linewidth=0.8)
                if row == 0:
                    ax.set_title(strategy_labels[strategies.index(strategy)], fontsize=9)
                if row == len(metrics) - 1:
                    ax.set_xlabel("Poison rate")
            axes[row][0].set_ylabel(ylabel)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][-1].legend(handles, labels, fontsize=8)
        plt.suptitle("Poisoning damage measured on clean test data")
        plt.tight_layout()
        self.save_figure("fig5_poisoning")

        worst = max(rates)
        rows = {}
        for architecture in architectures:
            row = {}
            for strategy in available:
                entry = data[architecture].get(strategy, {}).get(str(worst))
                if entry is None:
                    continue
                key = strategy.replace("label_flip_", "flip ").replace("_", " ")
                row[f"{key} ΔDR"] = entry["detection_rate_drop"]
                row[f"{key} ΔFPR"] = entry["false_positive_rate_change"]
            if row:
                rows[DISPLAY_NAME[architecture]] = row
        frame = pd.DataFrame(rows).T
        self.save_table(frame.round(4), f"table4_poisoning_rate{worst}")
        return frame

    # ---- phase 5 ----------------------------------------------------------------------

    def defence(self) -> pd.DataFrame | None:
        data = self.load("phase5_defence")
        if not data:
            return None

        frame = pd.DataFrame({
            DISPLAY_NAME[a]: {
                "ASR undefended": data[a]["undefended_attack_success_rate"],
                "ASR adversarially trained": data[a]["defended_attack_success_rate"],
                "ASR reduction": data[a]["attack_success_reduction"],
                "Clean accuracy cost": data[a]["clean_accuracy_cost"],
                "Defended accuracy": data[a]["defended_clean"]["accuracy"],
                "Defended detection rate": data[a]["defended_clean"]["detection_rate"],
            }
            for a in ARCHITECTURES if a in data
        }).T
        self.save_table(frame.round(4), "table5_defence")

        architectures = [a for a in ARCHITECTURES if a in data]
        x = np.arange(len(architectures))
        plt.figure(figsize=(9, 4.5))
        plt.bar(x - 0.2, [data[a]["undefended_attack_success_rate"] for a in architectures],
                0.4, label="Undefended", color="#C44E52")
        plt.bar(x + 0.2, [data[a]["defended_attack_success_rate"] for a in architectures],
                0.4, label="Adversarially trained", color="#4C72B0")
        plt.xticks(x, [DISPLAY_NAME[a] for a in architectures])
        plt.ylabel("Attack success rate")
        plt.title("Adversarial training as a defence (constrained PGD)")
        plt.legend()
        plt.tight_layout()
        self.save_figure("fig6_defence")
        return frame

    # ---- orchestration -----------------------------------------------------------------

    def build_all(self) -> None:
        print(f"Building figures for {self.dataset}")
        for name, builder in (
            ("phase 1 baseline", self.baseline),
            ("phase 2 evasion", self.evasion),
            ("phase 3 transfer", self.transfer),
            ("phase 4 poisoning", self.poisoning),
            ("phase 5 defence", self.defence),
        ):
            try:
                if builder() is None:
                    print(f"  {name}: no data, skipped")
            except Exception as exc:                      # keep going; partial results are useful
                print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
        print(f"Output: {self.figure_dir} and {self.table_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build figures and tables from results")
    parser.add_argument("--dataset", default="nsl_kdd")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    FigureBuilder(args.dataset, args.output_dir).build_all()


if __name__ == "__main__":
    main()
