"""
Preprocessing: encoding, scaling, and the feature-metadata bookkeeping the attacks need.

The central object is `Preprocessor`. Beyond the usual encode-and-scale duties it tracks
*where each original feature ended up* in the final numeric matrix. That bookkeeping is
what makes RO2 possible: to forbid perturbation of `protocol_type`, the constraint layer
must know which columns of the model's input tensor the one-hot expansion of
`protocol_type` occupies.

Scaling choice
--------------
Network flow features are extremely heavy-tailed - `src_bytes` in NSL-KDD spans 0 to
~1.4 billion, with a median of 44. Standardisation (z-score) on such a distribution leaves
almost all mass in a sliver near zero while a handful of outliers dominate the range, and
a fixed epsilon perturbation budget then means wildly different things for different
features.

`log1p` followed by min-max to [0, 1] is used instead for count-like and continuous
features. This has three benefits:
  1. it compresses the heavy tail into a usable range;
  2. a perturbation budget in the scaled space corresponds to a *multiplicative* change in
     bytes/packets, which is the physically meaningful notion for traffic volume;
  3. all features share the [0, 1] range, so an L-infinity budget is comparable across
     them - important when reporting a single epsilon.

Rates are already in [0, 1] and binaries in {0, 1}, so both are passed through unscaled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .loaders import NIDSDataset
from .schema import SPEC_BY_NAME, FeatureType, Mutability


@dataclass
class ColumnGroup:
    """Where one original feature lives in the encoded matrix."""

    source_name: str
    indices: tuple[int, ...]
    is_onehot: bool
    dtype: FeatureType
    mutability: Mutability
    scale_min: float = 0.0
    scale_max: float = 1.0
    log_scaled: bool = False


@dataclass
class Preprocessor:
    """Fit on training data, then applied identically to every other split."""

    categorical_features: list[str]
    numeric_features: list[str]
    groups: dict[str, ColumnGroup] = field(default_factory=dict)
    categories_: dict[str, list] = field(default_factory=dict)
    n_features_out_: int = 0
    output_names_: list[str] = field(default_factory=list)

    # ---- fitting ---------------------------------------------------------------------

    def fit(self, x: pd.DataFrame) -> "Preprocessor":
        cursor = 0
        self.groups.clear()
        self.output_names_ = []

        for name in self.numeric_features:
            values = x[name].to_numpy(dtype=np.float64)
            spec = SPEC_BY_NAME.get(name)
            dtype = spec.dtype if spec else _infer_type(values)
            mutability = spec.mutability if spec else Mutability.FREE

            # Rates and binaries are already normalised; scaling them again would distort
            # their semantics (a rate of 1.0 means "all connections", not "the maximum
            # value observed in this particular training set").
            passthrough = dtype in (FeatureType.RATE, FeatureType.BINARY)
            log_scaled = not passthrough

            if passthrough:
                lo, hi = 0.0, 1.0
            else:
                transformed = np.log1p(np.clip(values, 0, None))
                lo, hi = float(transformed.min()), float(transformed.max())
                if hi <= lo:
                    hi = lo + 1.0            # guard against a degenerate constant column

            self.groups[name] = ColumnGroup(
                source_name=name,
                indices=(cursor,),
                is_onehot=False,
                dtype=dtype,
                mutability=mutability,
                scale_min=lo,
                scale_max=hi,
                log_scaled=log_scaled,
            )
            self.output_names_.append(name)
            cursor += 1

        for name in self.categorical_features:
            categories = sorted(x[name].astype(str).unique().tolist())
            if "__unseen__" not in categories:
                categories.append("__unseen__")   # reserve a slot for inference-time novelties
            self.categories_[name] = categories

            indices = tuple(range(cursor, cursor + len(categories)))
            spec = SPEC_BY_NAME.get(name)
            self.groups[name] = ColumnGroup(
                source_name=name,
                indices=indices,
                is_onehot=True,
                dtype=FeatureType.CATEGORICAL,
                mutability=spec.mutability if spec else Mutability.IMMUTABLE,
            )
            self.output_names_.extend(f"{name}={c}" for c in categories)
            cursor += len(categories)

        self.n_features_out_ = cursor
        return self

    # ---- transforming ----------------------------------------------------------------

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(x), self.n_features_out_), dtype=np.float32)

        for name in self.numeric_features:
            group = self.groups[name]
            values = x[name].to_numpy(dtype=np.float64)
            if group.log_scaled:
                values = np.log1p(np.clip(values, 0, None))
                values = (values - group.scale_min) / (group.scale_max - group.scale_min)
            out[:, group.indices[0]] = np.clip(values, 0.0, 1.0)

        for name in self.categorical_features:
            group = self.groups[name]
            categories = self.categories_[name]
            lookup = {c: i for i, c in enumerate(categories)}
            unseen_slot = lookup["__unseen__"]
            column = x[name].astype(str).to_numpy()
            for row, value in enumerate(column):
                out[row, group.indices[lookup.get(value, unseen_slot)]] = 1.0

        return out

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.fit(x).transform(x)

    # ---- inverse, for inspecting adversarial examples --------------------------------

    def inverse_numeric(self, matrix: np.ndarray, name: str) -> np.ndarray:
        """
        Map a scaled column back to its original physical units.

        Needed to answer "what does this adversarial perturbation actually *mean*?" -
        e.g. that an epsilon of 0.05 on `src_bytes` corresponds to adding ~N bytes.
        """
        group = self.groups[name]
        values = matrix[:, group.indices[0]].astype(np.float64)
        if not group.log_scaled:
            return values
        values = values * (group.scale_max - group.scale_min) + group.scale_min
        return np.expm1(values)

    # ---- metadata for the attack layer -----------------------------------------------

    def mutability_masks(self) -> dict[str, np.ndarray]:
        """
        Boolean masks over output columns, one per mutability class.

        `nids.attacks.constraints` uses these to zero out gradient components that a
        realistic attacker could not act on.
        """
        masks = {
            m.value: np.zeros(self.n_features_out_, dtype=bool) for m in Mutability
        }
        for group in self.groups.values():
            masks[group.mutability.value][list(group.indices)] = True
        return masks

    def onehot_groups(self) -> list[tuple[int, ...]]:
        """Index tuples of one-hot blocks, which must stay exactly-one-hot after attack."""
        return [g.indices for g in self.groups.values() if g.is_onehot]

    def integer_columns(self) -> np.ndarray:
        """Columns whose underlying feature is integer-valued (counts, binaries)."""
        mask = np.zeros(self.n_features_out_, dtype=bool)
        for group in self.groups.values():
            if group.dtype in (FeatureType.COUNT, FeatureType.BINARY) and not group.is_onehot:
                mask[list(group.indices)] = True
        return mask


def _infer_type(values: np.ndarray) -> FeatureType:
    """Fallback typing for features absent from the canonical schema."""
    unique = np.unique(values)
    if set(unique.tolist()) <= {0.0, 1.0}:
        return FeatureType.BINARY
    if np.all(unique >= 0) and np.all(unique <= 1):
        return FeatureType.RATE
    if np.allclose(values, np.round(values)):
        return FeatureType.COUNT
    return FeatureType.CONTINUOUS


def prepare(
    dataset: NIDSDataset,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Turn a loaded dataset into model-ready arrays plus the metadata attacks need.

    A validation split is carved out of *training* data only; the official test split is
    never touched during model selection.
    """
    from sklearn.model_selection import train_test_split

    numeric = [c for c in dataset.feature_names if c not in dataset.categorical_features]
    preprocessor = Preprocessor(
        categorical_features=list(dataset.categorical_features),
        numeric_features=numeric,
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        dataset.x_train,
        dataset.y_train,
        test_size=validation_fraction,
        stratify=dataset.y_train,
        random_state=seed,
    )

    x_train = preprocessor.fit_transform(x_train_raw)
    x_val = preprocessor.transform(x_val_raw)
    x_test = preprocessor.transform(dataset.x_test)

    return {
        "x_train": x_train,
        "y_train": np.asarray(y_train),
        "x_val": x_val,
        "y_val": np.asarray(y_val),
        "x_test": x_test,
        "y_test": dataset.y_test,
        "preprocessor": preprocessor,
        "n_features": preprocessor.n_features_out_,
        "feature_names": preprocessor.output_names_,
        "attack_category_test": dataset.attack_category_test,
    }
