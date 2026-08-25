"""
Cross-domain transferability: train on dataset A, evaluate on dataset B.

Why this is hard, and why the honest answer matters
---------------------------------------------------
The three datasets do not share a feature space. NSL-KDD (1999-era, connection-oriented)
records 41 features including host-based counters; UNSW-NB15 (2015) records 49 flow
features from a different generator; CICIDS2017 (2017) records 78 purely statistical flow
features with no service or connection-state fields at all.

`schema.shared_features()` computes the genuine intersection. For all three datasets
together that intersection is only **five** features - duration, protocol_type, src_bytes,
dst_bytes, dst_host_count. Pairwise intersections are larger but still small.

This is itself a finding, and it should be stated rather than engineered around. Two common
workarounds in the literature both invalidate the result:

* **Zero-filling absent features.** A model then learns "feature 47 is always zero,
  therefore I am in domain A", and cross-domain numbers become uninterpretable.
* **Re-training the classifier head on target data.** That is domain *adaptation*, not
  transferability, and it silently answers a much easier question.

Neither is used here. Models are trained on the shared feature space only, and evaluated on
the target domain with frozen weights. The resulting numbers are low - and that low number
is the scientific content: flow-level NIDS features are not portable across capture
environments, which has direct consequences for anyone deploying a model trained on public
data onto a real network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..data.loaders import load_dataset
from ..data.preprocess import Preprocessor
from ..data.schema import shared_features
from ..eval.metrics import compute_metrics
from ..models.architectures import build_model
from ..models.trainer import TrainConfig, predict, train_model

ARCHITECTURE_NAMES = ["cnn", "lstm", "transformer", "gnn"]


def build_aligned_bundles(
    dataset_dirs: dict[str, str],
    seed: int = 42,
) -> tuple[dict[str, dict], list[str]]:
    """
    Load every dataset restricted to their common feature space, with a **shared encoder**.

    The shared preprocessor is essential and easy to get wrong. If each dataset were scaled
    by its own min/max, the value 0.5 would mean a different number of bytes in each domain
    and the transfer would fail for reasons of units rather than of distribution. Here the
    encoder is fitted on the datasets' pooled training data so that one scaled unit means
    the same physical quantity everywhere.
    """
    import pandas as pd

    names = list(dataset_dirs)
    features = shared_features(*names)
    if not features:
        raise ValueError(f"No shared features across {names}")

    raw = {}
    for name, directory in dataset_dirs.items():
        raw[name] = load_dataset(name, directory, feature_subset=features)

    pooled = pd.concat([d.x_train for d in raw.values()], ignore_index=True)
    categorical = [c for c in pooled.columns if not pd.api.types.is_numeric_dtype(pooled[c])]
    numeric = [c for c in pooled.columns if c not in categorical]

    encoder = Preprocessor(categorical_features=categorical, numeric_features=numeric)
    encoder.fit(pooled)

    bundles = {}
    for name, dataset in raw.items():
        bundles[name] = {
            "x_train": encoder.transform(dataset.x_train),
            "y_train": dataset.y_train,
            "x_test": encoder.transform(dataset.x_test),
            "y_test": dataset.y_test,
            "encoder": encoder,
            "n_features": encoder.n_features_out_,
        }
    return bundles, features


def run_cross_domain(
    dataset_dirs: dict[str, str],
    output_dir: str = "outputs",
    epochs: int = 20,
    batch_size: int = 512,
    seed: int = 42,
    device: str = "cpu",
    architectures: list[str] | None = None,
) -> dict:
    """
    Train each architecture on each source domain, evaluate on every other domain.

    Reports both the in-domain and out-of-domain score, because the *gap* is the quantity
    of interest - an out-of-domain accuracy of 0.62 means something quite different when
    in-domain is 0.99 than when it is 0.68.
    """
    from sklearn.model_selection import train_test_split

    architectures = architectures or ARCHITECTURE_NAMES
    output_path = Path(output_dir) / "metrics"
    output_path.mkdir(parents=True, exist_ok=True)

    bundles, features = build_aligned_bundles(dataset_dirs, seed=seed)
    names = list(bundles)

    print(f"Shared feature space ({len(features)}): {features}")
    print(f"Encoded width: {bundles[names[0]]['n_features']}")

    results: dict = {
        "_meta": {
            "shared_features": features,
            "n_encoded_features": bundles[names[0]]["n_features"],
            "datasets": names,
            "epochs": epochs,
            "seed": seed,
        }
    }

    for architecture in architectures:
        results[architecture] = {}

        for source in names:
            bundle = bundles[source]
            x_tr, x_val, y_tr, y_val = train_test_split(
                bundle["x_train"], bundle["y_train"],
                test_size=0.15, stratify=bundle["y_train"], random_state=seed,
            )

            print(f"\n[{architecture}] training on {source} "
                  f"({len(y_tr)} rows, {bundle['n_features']} features)")
            started = time.time()
            model = build_model(architecture, bundle["n_features"])
            model, _ = train_model(
                model, x_tr, y_tr, x_val, y_val,
                TrainConfig(epochs=epochs, batch_size=batch_size,
                            device=device, seed=seed, verbose=False),
            )
            results[architecture][source] = {"train_seconds": round(time.time() - started, 1)}

            for target in names:
                y_pred, y_score = predict(model, bundles[target]["x_test"], device)
                metrics = compute_metrics(bundles[target]["y_test"], y_pred, y_score)
                relation = "in_domain" if target == source else "cross_domain"
                results[architecture][source][target] = {"relation": relation, **metrics}

                print(f"    -> {target:12s} [{relation:12s}] "
                      f"acc {metrics['accuracy']:.4f} "
                      f"DR {metrics['detection_rate']:.4f} "
                      f"FPR {metrics['false_positive_rate']:.4f} "
                      f"AUC {metrics.get('roc_auc', float('nan')):.4f}")

            in_domain = results[architecture][source][source]["accuracy"]
            gaps = [
                in_domain - results[architecture][source][t]["accuracy"]
                for t in names if t != source
            ]
            results[architecture][source]["mean_generalisation_gap"] = (
                float(np.mean(gaps)) if gaps else 0.0
            )

    path = output_path / "cross_domain.json"
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nWrote {path}")
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cross-domain NIDS transferability")
    parser.add_argument("--nsl-kdd", default=None, help="path to NSL-KDD directory")
    parser.add_argument("--unsw-nb15", default=None, help="path to UNSW-NB15 directory")
    parser.add_argument("--cicids2017", default=None, help="path to CICIDS2017 directory")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dataset_dirs = {
        name: path
        for name, path in (
            ("nsl_kdd", args.nsl_kdd),
            ("unsw_nb15", args.unsw_nb15),
            ("cicids2017", args.cicids2017),
        )
        if path
    }
    if len(dataset_dirs) < 2:
        parser.error("cross-domain transfer needs at least two datasets")

    run_cross_domain(dataset_dirs, output_dir=args.output_dir,
                     epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
