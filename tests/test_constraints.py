"""
Tests for the realistic-traffic constraint projection.

These matter more than typical unit tests: the projection operator *is* Research Objective 2.
If it silently fails to enforce a constraint, the headline "constrained attacks are much
weaker" result would be measuring nothing, and the error would be invisible in the metrics
(a weaker attack looks like a successful defence).

Each test therefore checks a property that must hold for **every** output of `project()`,
not just that the code runs.

    .venv/Scripts/python.exe -m pytest tests/ -v
    .venv/Scripts/python.exe tests/test_constraints.py     # runs without pytest too
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nids.attacks.constraints import ConstraintConfig, ConstraintSet
from nids.data.preprocess import Preprocessor
from nids.data.schema import FeatureType, Mutability


def build_fixture(n_rows: int = 64, seed: int = 0):
    """
    A small frame exercising every mutability class and feature type.

    Rates are derived from their parent counts rather than drawn independently, so the
    fixture represents *internally consistent* records. Drawing them independently would
    produce impossible rows (e.g. serror_rate=0.37 over 2 connections), and the dependency
    repair would then legitimately move them - making `test_feasible_point_is_unchanged`
    fail for a reason that says nothing about the projection's correctness.
    """
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 500, n_rows)
    srv_counts = rng.integers(1, 500, n_rows)

    frame = pd.DataFrame({
        # increase-only counts / continuous
        "duration": rng.integers(0, 5000, n_rows).astype(float),
        "src_bytes": rng.integers(0, 100_000, n_rows).astype(float),
        "count": counts.astype(float),
        "srv_count": srv_counts.astype(float),
        "dst_host_count": rng.integers(1, 255, n_rows).astype(float),
        # immutable (victim-generated)
        "dst_bytes": rng.integers(0, 100_000, n_rows).astype(float),
        # derived rates - integer numerator over the parent count, hence achievable
        "serror_rate": (rng.integers(0, counts + 1) / counts).astype(float),
        "same_srv_rate": (rng.integers(0, counts + 1) / counts).astype(float),
        # free binary
        "land": rng.integers(0, 2, n_rows).astype(float),
        # immutable categoricals
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n_rows),
        "service": rng.choice(["http", "ftp", "dns", "smtp"], n_rows),
    })

    categorical = ["protocol_type", "service"]
    numeric = [c for c in frame.columns if c not in categorical]
    preprocessor = Preprocessor(categorical_features=categorical, numeric_features=numeric)
    x = preprocessor.fit_transform(frame)
    return preprocessor, torch.tensor(x, dtype=torch.float32), frame


def random_perturbation(x: torch.Tensor, scale: float = 0.3, seed: int = 1) -> torch.Tensor:
    """Large unconstrained noise, deliberately violating everything."""
    generator = torch.Generator().manual_seed(seed)
    noise = (torch.rand(x.shape, generator=generator) - 0.5) * 2 * scale
    return x + noise


# ======================================================================================
# Individual constraint families
# ======================================================================================


def test_immutable_features_are_restored():
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x), x)

    mask = constraints.immutable_mask
    assert torch.allclose(x_adv[:, mask], x[:, mask], atol=1e-6), \
        "immutable columns were modified by the projection"
    print("PASS immutable features restored")


def test_increase_only_features_never_decrease():
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    # Perturb strongly downward - the projection must undo every decrease.
    x_adv = constraints.project(x - 0.3, x)

    mask = constraints.increase_only_mask
    decreases = (x_adv[:, mask] - x[:, mask]) < -1e-6
    assert not decreases.any(), \
        f"{int(decreases.sum())} increase-only values fell below their original"
    print("PASS increase-only features never decrease")


def test_onehot_blocks_stay_onehot():
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x, scale=0.5), x)

    for group in constraints.onehot_groups:
        block = x_adv[:, group]
        assert torch.allclose(block.sum(dim=1), torch.ones(len(block)), atol=1e-5), \
            "one-hot block does not sum to 1"
        is_binary = ((block < 1e-6) | (block > 1 - 1e-6)).all()
        assert is_binary, "one-hot block contains fractional values"
    print("PASS one-hot blocks remain exactly one-hot")


def test_bounds_respected():
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x, scale=2.0), x)

    assert x_adv.min() >= -1e-6, f"value below 0: {float(x_adv.min())}"
    assert x_adv.max() <= 1 + 1e-6, f"value above 1: {float(x_adv.max())}"
    print("PASS values stay within [0, 1]")


def test_counts_are_integers_in_original_units():
    """
    Integrality must hold in *physical* units, not scaled ones.

    This is the subtle one. Under log1p scaling, rounding the scaled value is meaningless -
    near zero one scaled unit spans a fraction of a byte, near one it spans millions. The
    projection must un-scale, round, and re-scale.
    """
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x), x)

    for name, group in preprocessor.groups.items():
        if group.is_onehot or group.dtype not in (FeatureType.COUNT, FeatureType.BINARY):
            continue
        original_units = preprocessor.inverse_numeric(x_adv.numpy(), name)
        residual = np.abs(original_units - np.round(original_units))
        # Relative tolerance: float32 storage of a value near 1e6 cannot resolve better
        # than ~0.1 absolute, so an absolute threshold would fail on precision alone.
        tolerance = np.maximum(np.abs(original_units) * 1e-5, 1e-3)
        assert (residual <= tolerance).all(), (
            f"{name}: non-integer count after projection "
            f"(worst residual {residual.max():.4g} vs tolerance {tolerance.max():.4g})"
        )
    print("PASS counts are integers in original units")


def test_derived_rates_respect_granularity():
    """A rate over n connections must be a multiple of 1/n."""
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x), x)

    assert constraints.dependency_rules, "no dependency rules were discovered"
    for rule in constraints.dependency_rules:
        counts = preprocessor.inverse_numeric(x_adv.numpy(), rule["parent_names"][0])
        counts = np.maximum(np.round(counts), 1)
        rates = x_adv[:, rule["index"]].numpy()
        scaled = rates * counts
        residual = np.abs(scaled - np.round(scaled))
        assert residual.max() < 1e-3, \
            f"{rule['name']}: rate not a multiple of 1/count (residual {residual.max():.4g})"
    print(f"PASS derived rates respect 1/n granularity "
          f"({len(constraints.dependency_rules)} rules)")


# ======================================================================================
# Comparative behaviour
# ======================================================================================


def test_unconstrained_config_permits_violations():
    """
    The unconstrained baseline must genuinely be unconstrained.

    If `ConstraintConfig.unconstrained()` accidentally enforced something, the headline
    constrained-vs-unconstrained comparison would understate the difference, and the
    project's main claim would be wrong in the *conservative* direction - easy to miss.
    """
    preprocessor, x, _ = build_fixture()
    loose = ConstraintSet(preprocessor, ConstraintConfig.unconstrained())
    strict = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_loose = loose.project(random_perturbation(x), x)
    report = strict.feasibility_report(x_loose, x)

    assert report["immutable_violation_rate"] > 0.5, \
        "unconstrained projection unexpectedly preserved immutable features"
    assert report["onehot_violation_rate"] > 0.5, \
        "unconstrained projection unexpectedly preserved one-hot structure"
    print(f"PASS unconstrained permits violations "
          f"(immutable {report['immutable_violation_rate']:.2f}, "
          f"one-hot {report['onehot_violation_rate']:.2f})")


def test_projection_is_idempotent():
    """Projecting an already-feasible point must not move it."""
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    once = constraints.project(random_perturbation(x), x)
    twice = constraints.project(once, x)

    assert torch.allclose(once, twice, atol=1e-5), \
        "projection is not idempotent - the feasible set is inconsistent"
    print("PASS projection is idempotent")


def test_feasible_point_is_unchanged():
    """The original, real data must itself be feasible."""
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    projected = constraints.project(x.clone(), x)

    max_shift = float((projected - x).abs().max())
    assert max_shift < 1e-3, \
        f"projection moved genuine data by {max_shift:.4g} - the constraint set is too tight"
    print(f"PASS real data is feasible (max shift {max_shift:.2e})")


def test_feasibility_report_clean_on_projected_output():
    """After projection, the feasibility report must show ~zero violations."""
    preprocessor, x, _ = build_fixture()
    constraints = ConstraintSet(preprocessor, ConstraintConfig.realistic())

    x_adv = constraints.project(random_perturbation(x), x)
    report = constraints.feasibility_report(x_adv, x)

    for key, value in report.items():
        if key.endswith("_violation_rate"):
            assert value < 1e-6, f"{key} = {value} after projection"
    print("PASS projected output reports zero violations")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    print(f"Running {len(tests)} constraint tests\n" + "-" * 60)
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
