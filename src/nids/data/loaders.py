"""
Dataset loaders producing a uniform `NIDSDataset` regardless of source.

Each loader is responsible for the messy, dataset-specific work - column naming, label
vocabularies, known data-quality defects - and returns the same structure, so that
everything downstream (models, attacks, evaluation) is dataset-agnostic.

Design decisions worth knowing
------------------------------
* **Labels are binarised to benign=0 / attack=1.** The three datasets have incompatible
  attack taxonomies (NSL-KDD's 39 attack names, UNSW-NB15's 9 categories, CICIDS2017's 14),
  so a multi-class cross-domain experiment would require an arbitrary mapping. Binary
  detection is the task all three genuinely share, and is what an operational NIDS alarm
  actually does. The original fine-grained label is retained in `attack_category` for
  per-attack-type analysis.

* **The official NSL-KDD train/test split is preserved.** KDDTest+ deliberately contains
  attack types absent from KDDTrain+, making it a built-in generalisation test. Reshuffling
  the two together - common in published work - destroys that property and inflates
  reported accuracy substantially.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import CANONICAL_NAMES, DATASET_MAPPINGS, SPEC_BY_NAME, FeatureType


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    """
    Identify non-numeric columns.

    Checked via `is_numeric_dtype` rather than `dtype == object`: pandas 3.0 represents
    string columns with a dedicated `str` dtype, so the older object-dtype test silently
    reports zero categorical columns and the one-hot encoding is skipped entirely.
    """
    return [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]


# --------------------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------------------


@dataclass
class NIDSDataset:
    """A loaded, canonicalised dataset split."""

    name: str
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    feature_names: list[str]
    categorical_features: list[str]
    attack_category_train: np.ndarray
    attack_category_test: np.ndarray

    def summary(self) -> str:
        def dist(y: np.ndarray) -> str:
            benign = int((y == 0).sum())
            attack = int((y == 1).sum())
            total = len(y)
            return f"{total} rows (benign {benign} / {benign/total:.1%}, attack {attack} / {attack/total:.1%})"

        return (
            f"Dataset: {self.name}\n"
            f"  train: {dist(self.y_train)}\n"
            f"  test : {dist(self.y_test)}\n"
            f"  features: {len(self.feature_names)} "
            f"({len(self.categorical_features)} categorical)\n"
            f"  missing canonical features: "
            f"{sorted(set(CANONICAL_NAMES) - set(self.feature_names)) or 'none'}"
        )


# --------------------------------------------------------------------------------------
# NSL-KDD
# --------------------------------------------------------------------------------------

# The NSL-KDD files ship without a header row. This is the canonical 41-feature ordering
# from the KDD Cup '99 task description, plus the label and the NSL-KDD difficulty score.
NSL_KDD_COLUMNS: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]


def load_nsl_kdd(
    data_dir: str | Path,
    canonical_only: bool = False,
    feature_subset: list[str] | None = None,
) -> NIDSDataset:
    """
    Load NSL-KDD from `KDDTrain+.txt` / `KDDTest+.txt`.

    Args:
        data_dir: directory containing the two .txt files.
        canonical_only: restrict to the cross-domain canonical feature space. Use for
            transfer experiments; leave False for single-domain work, where discarding
            NSL-KDD's other 23 features would needlessly weaken the baseline.
        feature_subset: explicit canonical features to keep (overrides canonical_only).
            Used by cross-domain runs via `schema.shared_features(...)`.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "KDDTrain+.txt"
    test_path = data_dir / "KDDTest+.txt"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"NSL-KDD file missing: {path}")

    train = pd.read_csv(train_path, names=NSL_KDD_COLUMNS, header=None)
    test = pd.read_csv(test_path, names=NSL_KDD_COLUMNS, header=None)

    # `difficulty` is NSL-KDD metadata (how many baseline learners classified the row
    # correctly), not a network measurement. Leaving it in leaks label information.
    train = train.drop(columns=["difficulty"])
    test = test.drop(columns=["difficulty"])

    attack_train = train["label"].to_numpy()
    attack_test = test["label"].to_numpy()
    y_train = (train["label"] != "normal").astype(np.int64).to_numpy()
    y_test = (test["label"] != "normal").astype(np.int64).to_numpy()
    x_train = train.drop(columns=["label"])
    x_test = test.drop(columns=["label"])

    if feature_subset is not None or canonical_only:
        mapping = DATASET_MAPPINGS["nsl_kdd"]
        wanted = feature_subset if feature_subset is not None else list(CANONICAL_NAMES)
        keep = [mapping[c] for c in wanted if c in mapping]
        rename = {mapping[c]: c for c in wanted if c in mapping}
        x_train = x_train[keep].rename(columns=rename)
        x_test = x_test[keep].rename(columns=rename)

    # `num_outbound_cmds` is identically zero across the whole dataset - a well-known
    # NSL-KDD defect. A constant column contributes nothing and produces a zero-variance
    # division during standardisation.
    constant = [c for c in x_train.columns if x_train[c].nunique() <= 1]
    if constant:
        x_train = x_train.drop(columns=constant)
        x_test = x_test.drop(columns=constant)

    categorical = _categorical_columns(x_train)

    # The official test set contains service values unseen in training (part of its
    # deliberate difficulty). Map them to a sentinel so encoders do not fail at inference.
    for column in categorical:
        unseen = set(x_test[column].unique()) - set(x_train[column].unique())
        if unseen:
            x_test[column] = x_test[column].where(~x_test[column].isin(unseen), "__unseen__")

    return NIDSDataset(
        name="nsl_kdd",
        x_train=x_train.reset_index(drop=True),
        y_train=y_train,
        x_test=x_test.reset_index(drop=True),
        y_test=y_test,
        feature_names=list(x_train.columns),
        categorical_features=categorical,
        attack_category_train=attack_train,
        attack_category_test=attack_test,
    )


# --------------------------------------------------------------------------------------
# UNSW-NB15
# --------------------------------------------------------------------------------------


def load_unsw_nb15(
    data_dir: str | Path,
    canonical_only: bool = True,
    feature_subset: list[str] | None = None,
) -> NIDSDataset:
    """
    Load UNSW-NB15 from its official train/test CSVs.

    Expects `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` (the partitioned
    release). Requires manual download - UNSW Canberra gates the files behind a form.
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "UNSW_NB15_training-set.csv"
    test_path = data_dir / "UNSW_NB15_testing-set.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"UNSW-NB15 file missing: {path}\n"
                "Download from https://research.unsw.edu.au/projects/unsw-nb15-dataset "
                "(registration required) and place the partitioned CSVs here."
            )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # `label` is the binary target; `attack_cat` the fine-grained category; `id` is a row
    # index that would leak ordering information.
    attack_train = train.get("attack_cat", pd.Series(["unknown"] * len(train))).to_numpy()
    attack_test = test.get("attack_cat", pd.Series(["unknown"] * len(test))).to_numpy()
    y_train = train["label"].astype(np.int64).to_numpy()
    y_test = test["label"].astype(np.int64).to_numpy()

    drop = [c for c in ("id", "label", "attack_cat") if c in train.columns]
    x_train = train.drop(columns=drop)
    x_test = test.drop(columns=drop)

    if feature_subset is not None or canonical_only:
        mapping = DATASET_MAPPINGS["unsw_nb15"]
        wanted = feature_subset if feature_subset is not None else list(CANONICAL_NAMES)
        keep = [mapping[c] for c in wanted if c in mapping and mapping[c] in x_train.columns]
        rename = {mapping[c]: c for c in wanted if c in mapping and mapping[c] in x_train.columns}
        x_train = x_train[keep].rename(columns=rename)
        x_test = x_test[keep].rename(columns=rename)

    categorical = _categorical_columns(x_train)
    for column in categorical:
        unseen = set(x_test[column].unique()) - set(x_train[column].unique())
        if unseen:
            x_test[column] = x_test[column].where(~x_test[column].isin(unseen), "__unseen__")

    return NIDSDataset(
        name="unsw_nb15",
        x_train=x_train.reset_index(drop=True),
        y_train=y_train,
        x_test=x_test.reset_index(drop=True),
        y_test=y_test,
        feature_names=list(x_train.columns),
        categorical_features=categorical,
        attack_category_train=attack_train,
        attack_category_test=attack_test,
    )


# --------------------------------------------------------------------------------------
# CICIDS2017
# --------------------------------------------------------------------------------------


def load_cicids2017(
    data_dir: str | Path,
    canonical_only: bool = True,
    feature_subset: list[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> NIDSDataset:
    """
    Load CICIDS2017 from the MachineLearningCSV release (8 CSVs, one per capture period).

    CICIDS2017 ships no official train/test split, so one is made here - stratified, with a
    fixed seed for reproducibility.

    The raw CSVs have several well-documented defects that are corrected here:
      * leading spaces in almost every column name
      * infinite values in `Flow Bytes/s` and `Flow Packets/s` (division by zero duration)
      * a handful of NaN rows
    """
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSVs found in {data_dir}\n"
            "Download the CICIDS2017 MachineLearningCSV archive from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and extract it here."
        )

    frames = [pd.read_csv(path, low_memory=False) for path in csv_paths]
    data = pd.concat(frames, ignore_index=True)
    data.columns = [c.strip() for c in data.columns]

    label_column = "Label" if "Label" in data.columns else data.columns[-1]
    attack_category = data[label_column].astype(str).str.strip().to_numpy()
    y = (attack_category != "BENIGN").astype(np.int64)

    x = data.drop(columns=[label_column])
    x = x.replace([np.inf, -np.inf], np.nan)
    valid = ~x.isna().any(axis=1)
    x, y, attack_category = x[valid], y[valid.to_numpy()], attack_category[valid.to_numpy()]

    if feature_subset is not None or canonical_only:
        mapping = DATASET_MAPPINGS["cicids2017"]
        wanted = feature_subset if feature_subset is not None else list(CANONICAL_NAMES)
        keep = [mapping[c] for c in wanted if c in mapping and mapping[c] in x.columns]
        rename = {mapping[c]: c for c in wanted if c in mapping and mapping[c] in x.columns}
        x = x[keep].rename(columns=rename)

    x_train, x_test, y_train, y_test, cat_train, cat_test = train_test_split(
        x, y, attack_category, test_size=test_size, stratify=y, random_state=seed
    )

    categorical = _categorical_columns(x_train)

    return NIDSDataset(
        name="cicids2017",
        x_train=x_train.reset_index(drop=True),
        y_train=np.asarray(y_train),
        x_test=x_test.reset_index(drop=True),
        y_test=np.asarray(y_test),
        feature_names=list(x_train.columns),
        categorical_features=categorical,
        attack_category_train=np.asarray(cat_train),
        attack_category_test=np.asarray(cat_test),
    )


LOADERS = {
    "nsl_kdd": load_nsl_kdd,
    "unsw_nb15": load_unsw_nb15,
    "cicids2017": load_cicids2017,
}


def load_dataset(name: str, data_dir: str | Path, **kwargs) -> NIDSDataset:
    """Dispatch to the appropriate loader by dataset name."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(LOADERS)}")
    return LOADERS[name](data_dir, **kwargs)
