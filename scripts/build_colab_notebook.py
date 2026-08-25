"""
Generate the Colab notebook for GPU experiments on UNSW-NB15 / CICIDS2017 and cross-domain
transfer.

Generated from a script rather than hand-edited so the pipeline stays reviewable as plain
Python and diffable in version control.

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    })


# ======================================================================================
md("""# Adversarial Robustness & Cross-Domain Transferability of Deep-Learning NIDS

GPU companion notebook to the `nids` library.

**What runs where.** NSL-KDD is small enough to run on CPU and its results are produced
locally by `python -m nids.experiments.run_all`. This notebook exists for the two datasets
that are too large for that, and for the cross-domain experiments that need all three:

| Dataset | Rows | Where |
|---|---:|---|
| NSL-KDD | 148k | local CPU (already done) |
| UNSW-NB15 | 257k | here, GPU |
| CICIDS2017 | ~2.8M | here, GPU |

**Before running:** both datasets are gated behind registration forms and cannot be
downloaded automatically.

* UNSW-NB15 — https://research.unsw.edu.au/projects/unsw-nb15-dataset
  Get the *partitioned* CSVs: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
* CICIDS2017 — https://www.unb.ca/cic/datasets/ids-2017.html
  Get the `MachineLearningCSV.zip` archive (8 CSVs), not the raw PCAPs

Upload both to Drive, then set the paths in Section 2.

**Runtime → Change runtime type → GPU** before running anything.
""")

# --------------------------------------------------------------------------------------
md("## 1. Environment")

code("""import torch
print("torch:", torch.__version__)
print("CUDA :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "- enable GPU runtime!")
""")

code("""# Upload the `nids` package. Two options:
#   (a) clone from your repo, if you have pushed it
#   (b) zip D:/NIDS/src and upload it here
from google.colab import files
import os, sys, zipfile

if not os.path.exists("/content/src/nids"):
    print("Upload src.zip (zip of D:/NIDS/src):")
    uploaded = files.upload()
    with zipfile.ZipFile(next(iter(uploaded))) as archive:
        archive.extractall("/content")

sys.path.insert(0, "/content/src")
from nids.data.schema import CANONICAL_NAMES, shared_features
print("nids package loaded;", len(CANONICAL_NAMES), "canonical features")
""")

# --------------------------------------------------------------------------------------
md("""## 2. Data paths

Mount Drive and point these at the uploaded datasets. If `drive.mount` hangs (common on
institutional Google accounts and restrictive networks), use the `gdown` fallback in the
next cell instead.""")

code("""from google.colab import drive
drive.mount('/content/drive')

NSL_KDD_DIR    = "/content/drive/MyDrive/NIDS/data/nsl_kdd"
UNSW_NB15_DIR  = "/content/drive/MyDrive/NIDS/data/unsw_nb15"
CICIDS2017_DIR = "/content/drive/MyDrive/NIDS/data/cicids2017"

OUTPUT_DIR = "/content/outputs"   # local: Drive writes are slow for many small files
""")

code("""# FALLBACK if drive.mount() times out: download a zip shared as "Anyone with the link".
# !pip -q install gdown
# !gdown --id YOUR_FILE_ID -O /content/data.zip
# !unzip -q /content/data.zip -d /content/data
""")

code("""import os
for name, path in [("NSL-KDD", NSL_KDD_DIR), ("UNSW-NB15", UNSW_NB15_DIR),
                   ("CICIDS2017", CICIDS2017_DIR)]:
    exists = os.path.isdir(path)
    contents = sorted(os.listdir(path))[:4] if exists else []
    print(f"{name:12s} {'OK ' if exists else 'MISSING'} {path}")
    if contents:
        print(f"             {contents}")
""")

# --------------------------------------------------------------------------------------
md("""## 3. Single-dataset experiments

Runs the identical pipeline used for NSL-KDD: baselines, white-box evasion (constrained vs
unconstrained), black-box transfer, poisoning, and adversarial-training defence.

CICIDS2017 has ~2.8M rows. Start with `--subset` to confirm everything runs before
committing to a full pass.""")

code("""# UNSW-NB15 - full run
!cd /content && python -u -m nids.experiments.run_all \\
    --dataset unsw_nb15 \\
    --data-dir "{UNSW_NB15_DIR}" \\
    --output-dir "{OUTPUT_DIR}" \\
    --device cuda --epochs 20 --batch-size 1024 --phases 1,2,3,4,5
""".replace("{UNSW_NB15_DIR}", "$UNSW_NB15_DIR").replace("{OUTPUT_DIR}", "$OUTPUT_DIR"))

code("""import os
os.environ["UNSW_NB15_DIR"] = UNSW_NB15_DIR
os.environ["CICIDS2017_DIR"] = CICIDS2017_DIR
os.environ["NSL_KDD_DIR"] = NSL_KDD_DIR
os.environ["OUTPUT_DIR"] = OUTPUT_DIR
print("environment variables set for the shell cells")
""")

code("""# CICIDS2017 - smoke test first (200k rows), then remove --subset for the full run
!cd /content && python -u -m nids.experiments.run_all \\
    --dataset cicids2017 \\
    --data-dir "$CICIDS2017_DIR" \\
    --output-dir "$OUTPUT_DIR" \\
    --device cuda --epochs 20 --batch-size 1024 \\
    --subset 200000 --phases 1,2,3
""")

# --------------------------------------------------------------------------------------
md("""## 4. Cross-domain transferability

Trains on one dataset and evaluates on the others, using only the features genuinely
common to all of them.

**Expect low numbers, and report them.** The full three-way intersection is just five
features (duration, protocol_type, src_bytes, dst_bytes, dst_host_count). A model
restricted to five features and then moved to a different capture environment, a different
decade, and a different attack taxonomy should not be expected to transfer well. The
measurement is the contribution: it quantifies how badly published-dataset NIDS results
generalise, which is the practical question for anyone deploying one.

Do **not** be tempted to zero-fill missing features or fine-tune on the target - the first
lets the model identify its domain from the padding, the second measures domain adaptation
rather than transferability.""")

code("""from nids.data.schema import shared_features

for pair in [("nsl_kdd","unsw_nb15"), ("nsl_kdd","cicids2017"), ("unsw_nb15","cicids2017")]:
    print(f"{pair[0]:12s} & {pair[1]:12s}: {shared_features(*pair)}")
print()
print("all three:", shared_features("nsl_kdd","unsw_nb15","cicids2017"))
""")

code("""!cd /content && python -u -m nids.experiments.cross_domain \\
    --nsl-kdd "$NSL_KDD_DIR" \\
    --unsw-nb15 "$UNSW_NB15_DIR" \\
    --cicids2017 "$CICIDS2017_DIR" \\
    --output-dir "$OUTPUT_DIR" \\
    --epochs 20 --device cuda
""")

# --------------------------------------------------------------------------------------
md("## 5. Results summary")

code("""import json, glob, pandas as pd

for path in sorted(glob.glob(f"{OUTPUT_DIR}/metrics/*.json")):
    print(path.split("/")[-1])

def baseline_table(dataset):
    with open(f"{OUTPUT_DIR}/metrics/{dataset}_phase1_baseline.json") as fh:
        data = json.load(fh)
    return pd.DataFrame({
        arch: {
            "parameters": row["parameters"],
            "test_accuracy": row["test"]["accuracy"],
            "detection_rate": row["test"]["detection_rate"],
            "false_positive_rate": row["test"]["false_positive_rate"],
            "roc_auc": row["test"].get("roc_auc"),
        } for arch, row in data.items()
    }).T.round(4)

for dataset in ("unsw_nb15", "cicids2017"):
    try:
        print(f"\\n=== {dataset} baselines ===")
        display(baseline_table(dataset))
    except FileNotFoundError:
        print(f"\\n{dataset}: not run yet")
""")

code("""# Constrained vs unconstrained attack success - the RO2 headline
import json, pandas as pd

def constraint_table(dataset, attack="pgd", epsilon="0.1"):
    with open(f"{OUTPUT_DIR}/metrics/{dataset}_phase2_evasion.json") as fh:
        data = json.load(fh)
    rows = {}
    for arch, entry in data.items():
        unconstrained = entry[f"{attack}_unconstrained"][epsilon]
        constrained = entry[f"{attack}_constrained"][epsilon]
        rows[arch] = {
            "ASR_unconstrained": unconstrained["attack_success_rate"],
            "ASR_constrained": constrained["attack_success_rate"],
            "overstatement_factor": (
                unconstrained["attack_success_rate"]
                / max(constrained["attack_success_rate"], 1e-9)
            ),
            "unconstrained_infeasible_rate":
                unconstrained["feasibility"].get("any_violation_rate_max"),
        }
    return pd.DataFrame(rows).T.round(4)

for dataset in ("unsw_nb15", "cicids2017"):
    try:
        print(f"\\n=== {dataset}: PGD eps=0.1 ===")
        display(constraint_table(dataset))
    except FileNotFoundError:
        print(f"\\n{dataset}: not run yet")
""")

# --------------------------------------------------------------------------------------
md("""## 6. Download results

Colab wipes `/content` on disconnect. Save this zip before closing the tab, and send it
back for the analysis write-up.""")

code("""import shutil
from google.colab import files
archive = shutil.make_archive("/content/nids_outputs", "zip", OUTPUT_DIR)
print("archived:", archive)
files.download(archive)
""")

# ======================================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "NIDS_GPU_Experiments.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {out_path}")
print(f"  {len(cells)} cells ({n_code} code, {len(cells)-n_code} markdown)")
