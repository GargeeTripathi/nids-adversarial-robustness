"""
Realistic perturbation constraints for network traffic (Research Objective 2).

The problem
-----------
Adversarial ML was developed on images, where the threat model is simple: any pixel may
change by any amount up to a budget, and the result is still a valid image. Network flows
do not work that way. A gradient-based attack run unmodified on flow features will happily
produce records such as:

    duration      = -0.4 s          (time does not run backwards)
    src_bytes     = 1_204.7         (bytes are integers)
    protocol_type = 0.3·tcp + 0.7·udp   (a packet has exactly one protocol)
    src_bytes     = 12_000_000  while  duration = 0.001 s   (≈ 12 GB/ms)
    serror_rate   = 0.83  while  count = 1   (a rate over one connection is 0 or 1)

Every one of these is unrealizable: no attacker can emit such a flow, so a detector's
"failure" on it is a mathematical artefact, not a security finding. Papers that omit these
constraints systematically overstate attack success rates.

The solution
------------
A **projection operator** `Π` that maps any perturbed point back to the nearest point in
the realizable set `Ω`. Attacks call `project()` after each gradient step, so the iterate
is feasible at all times - the standard projected-gradient formulation, with `Ω` defined by
network semantics rather than by an epsilon-ball alone.

Six constraint families are enforced, in a deliberate order (later ones can invalidate
earlier ones, so ordering matters):

1. **Immutability** - features the attacker does not control are restored to their
   original values (`protocol_type`, `dst_bytes`, `flag`, ...).
2. **Directionality** - INCREASE_ONLY features may not fall below their original value.
   This is the most consequential constraint: an attacker can pad, delay and add, but
   cannot un-send.
3. **Box bounds** - every feature clipped to its valid range, all [0,1] post-scaling.
4. **One-hot integrity** - each categorical block is re-projected to exactly one active
   category.
5. **Integrality** - count-valued features rounded, in the *original* unit space rather
   than the scaled space, since rounding a log-scaled value is not the same operation.
6. **Dependency repair** - derived features (rates, ratios) recomputed from their parents
   so the record is internally arithmetically consistent.

`ConstraintSet.project()` is differentiable-friendly in the sense that it is applied
*outside* the gradient computation (as in standard PGD), so no gradient flows through the
rounding or the argmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.preprocess import Preprocessor
from ..data.schema import FeatureType, Mutability


@dataclass
class ConstraintConfig:
    """Which constraint families to enforce. Disabling all reproduces the naive attack."""

    enforce_immutability: bool = True
    enforce_direction: bool = True
    enforce_bounds: bool = True
    enforce_onehot: bool = True
    enforce_integrality: bool = True
    enforce_dependencies: bool = True

    @classmethod
    def unconstrained(cls) -> "ConstraintConfig":
        """The naive image-style threat model, used as the comparison baseline."""
        return cls(False, False, True, False, False, False)
        # NB: bounds stay on even when "unconstrained". Without them the attack escapes
        # the [0,1] scaled domain entirely and the comparison becomes meaningless - the
        # model would be evaluated far outside its training distribution for reasons that
        # have nothing to do with adversarial robustness.

    @classmethod
    def realistic(cls) -> "ConstraintConfig":
        return cls(True, True, True, True, True, True)


class ConstraintSet:
    """
    Projection operator onto the realizable-traffic set for one preprocessor.

    Built once per dataset and reused across attacks.
    """

    def __init__(self, preprocessor: Preprocessor, config: ConstraintConfig | None = None):
        self.preprocessor = preprocessor
        self.config = config or ConstraintConfig.realistic()

        masks = preprocessor.mutability_masks()

        # Truly immutable and derived are tracked SEPARATELY, though both are reset to
        # their original values before dependency repair. The distinction matters when
        # reporting feasibility: a derived rate that changed because the attacker legally
        # raised its parent `count` is *correct* behaviour, not a violation, whereas a
        # changed `protocol_type` always is. Conflating them made the feasibility report
        # flag every projected batch as violating immutability.
        self.immutable_mask = torch.tensor(masks[Mutability.IMMUTABLE.value])
        self.derived_mask = torch.tensor(masks[Mutability.DERIVED.value])
        self.reset_mask = self.immutable_mask | self.derived_mask
        self.increase_only_mask = torch.tensor(masks[Mutability.INCREASE_ONLY.value])
        self.onehot_groups = [torch.tensor(g) for g in preprocessor.onehot_groups()]
        self.integer_mask = torch.tensor(preprocessor.integer_columns())

        # Derived features and the parents they must be recomputed from.
        self.dependency_rules = self._build_dependency_rules()

    # ---- construction -----------------------------------------------------------------

    def _build_dependency_rules(self) -> list[dict]:
        """
        Locate derived rate features and their parent counts in the encoded matrix.

        Only rules whose parents are all present are kept; a rate whose denominator is not
        in the feature set cannot be repaired and is instead treated as immutable (it is
        already in `immutable_mask` via the DERIVED mutability class).
        """
        from ..data.schema import SPEC_BY_NAME

        rules = []
        groups = self.preprocessor.groups
        for name, group in groups.items():
            spec = SPEC_BY_NAME.get(name)
            if spec is None or spec.mutability != Mutability.DERIVED:
                continue
            parents = [p for p in spec.depends_on if p in groups]
            if len(parents) != len(spec.depends_on) or not parents:
                continue
            rules.append({
                "name": name,
                "index": group.indices[0],
                "parent_indices": [groups[p].indices[0] for p in parents],
                "parent_names": parents,
            })
        return rules

    # ---- projection --------------------------------------------------------------------

    def project(self, x_adv: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Project `x_adv` onto the realizable set, given the unperturbed `x_orig`.

        Both tensors are in the *scaled* [0,1] encoded space.
        """
        x = x_adv.clone()
        device = x.device

        if self.config.enforce_immutability:
            # Derived features are reset here too: the gradient step moved them directly,
            # which an attacker cannot do. Their correct values are recomputed from the
            # (possibly perturbed) parents in the dependency step below.
            mask = self.reset_mask.to(device)
            x[:, mask] = x_orig[:, mask]

        if self.config.enforce_direction:
            mask = self.increase_only_mask.to(device)
            # Monotone (log1p + min-max) scaling preserves order, so "no smaller than the
            # original" is the same test in scaled space as in byte/second space.
            x[:, mask] = torch.maximum(x[:, mask], x_orig[:, mask])

        if self.config.enforce_bounds:
            x = x.clamp(0.0, 1.0)

        if self.config.enforce_onehot:
            for group in self.onehot_groups:
                group = group.to(device)
                block = x[:, group]
                hard = torch.zeros_like(block)
                hard[torch.arange(block.shape[0], device=device), block.argmax(dim=1)] = 1.0
                x[:, group] = hard

        if self.config.enforce_integrality:
            x = self._round_counts(x)

        if self.config.enforce_dependencies:
            x = self._repair_dependencies(x)
            if self.config.enforce_bounds:
                x = x.clamp(0.0, 1.0)

        # Rounding and dependency repair both write to columns, so immutability is
        # re-asserted last. Without this the float round-trip in _round_counts drifts
        # immutable counts (e.g. dst_bytes) a few ULPs off their original values, and the
        # projection silently violates the constraint it just enforced.
        if self.config.enforce_immutability:
            mask = self.immutable_mask.to(device)
            x[:, mask] = x_orig[:, mask]

        return x

    def _round_counts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Round count features to integers **in their original units**.

        Rounding the scaled value would be wrong: near zero, one scaled unit spans a
        fraction of a byte, while near one it spans millions. The value is therefore
        un-scaled, rounded, and re-scaled.
        """
        x = x.clone()
        for name, group in self.preprocessor.groups.items():
            if group.is_onehot:
                continue
            if group.dtype not in (FeatureType.COUNT, FeatureType.BINARY):
                continue
            # Immutable counts already hold exact original values; running them through the
            # log/exp round-trip would only introduce drift.
            if group.mutability == Mutability.IMMUTABLE:
                continue

            index = group.indices[0]
            column = x[:, index]

            if not group.log_scaled:                       # binaries: already 0/1 space
                x[:, index] = column.round()
                continue

            # float64 for the round-trip: src_bytes reaches ~1e6, where float32's ~7
            # significant digits leave a residual large enough to break integrality after
            # expm1/log1p. The result is cast back to the tensor's own dtype.
            span = group.scale_max - group.scale_min
            wide = column.double()
            original = torch.expm1(wide * span + group.scale_min).round().clamp(min=0)
            rescaled = (torch.log1p(original) - group.scale_min) / span
            x[:, index] = rescaled.to(x.dtype)
        return x

    def _repair_dependencies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recompute derived rate features so the record is arithmetically self-consistent.

        The exact windowed computation behind NSL-KDD's rate features cannot be inverted
        from a single record - the underlying per-connection outcomes are not in the data.
        What *is* enforceable is the granularity bound: a rate over `n` connections must be
        a multiple of `1/n`. A `serror_rate` of 0.37 with `count = 2` is impossible, and
        this snaps it to the nearest achievable value (0.5).

        This is a weaker repair than a full generative reconstruction would be, but it is
        sound - it never accepts a record it should reject - and it is exactly the kind of
        integrity check a defender could implement at ingest.
        """
        x = x.clone()
        for rule in self.dependency_rules:
            rate_index = rule["index"]
            denominator_index = rule["parent_indices"][0]

            group = self.preprocessor.groups[rule["parent_names"][0]]
            span = group.scale_max - group.scale_min
            counts = torch.expm1(
                x[:, denominator_index] * span + group.scale_min
            ).round().clamp(min=1)

            rate = x[:, rate_index].clamp(0.0, 1.0)
            x[:, rate_index] = torch.round(rate * counts) / counts
        return x

    # ---- reporting ---------------------------------------------------------------------

    def feasibility_report(self, x_adv: torch.Tensor, x_orig: torch.Tensor) -> dict:
        """
        Measure how far an *unprojected* adversarial batch violates realism.

        This is what quantifies the paper's central claim: it converts "the unconstrained
        attack is unrealistic" from an assertion into a number.
        """
        device = x_adv.device
        report: dict[str, float] = {}

        immutable = self.immutable_mask.to(device)
        if immutable.any():
            changed = (x_adv[:, immutable] - x_orig[:, immutable]).abs() > 1e-6
            report["immutable_violation_rate"] = float(changed.any(dim=1).float().mean())
            report["immutable_features_changed_mean"] = float(changed.float().sum(dim=1).mean())

        # Derived features are checked for *consistency with their parents*, not for
        # equality with the original. A rate legitimately changes when the attacker raises
        # the count it is computed over; what it may not do is take a value unachievable
        # for that count (e.g. 0.37 over 2 connections).
        if self.dependency_rules:
            inconsistent = []
            for rule in self.dependency_rules:
                parent = self.preprocessor.groups[rule["parent_names"][0]]
                span = parent.scale_max - parent.scale_min
                counts = torch.expm1(
                    x_adv[:, rule["parent_indices"][0]].double() * span + parent.scale_min
                ).round().clamp(min=1)
                rate = x_adv[:, rule["index"]].double().clamp(0.0, 1.0)
                scaled = rate * counts
                inconsistent.append((scaled - scaled.round()).abs() > 1e-3)
            report["derived_inconsistency_rate"] = float(
                torch.stack(inconsistent).any(dim=0).float().mean()
            )

        increase = self.increase_only_mask.to(device)
        if increase.any():
            decreased = (x_adv[:, increase] - x_orig[:, increase]) < -1e-6
            report["direction_violation_rate"] = float(decreased.any(dim=1).float().mean())
            report["direction_features_decreased_mean"] = float(decreased.float().sum(dim=1).mean())

        if self.onehot_groups:
            violations = []
            for group in self.onehot_groups:
                block = x_adv[:, group.to(device)]
                is_onehot = (
                    ((block - 1.0).abs() < 1e-6).sum(dim=1) == 1
                ) & (((block).abs() < 1e-6).sum(dim=1) == block.shape[1] - 1)
                violations.append(~is_onehot)
            report["onehot_violation_rate"] = float(
                torch.stack(violations).any(dim=0).float().mean()
            )

        out_of_bounds = (x_adv < -1e-6) | (x_adv > 1.0 + 1e-6)
        report["bounds_violation_rate"] = float(out_of_bounds.any(dim=1).float().mean())

        integer_mask = self.integer_mask.to(device)
        if integer_mask.any():
            fractional = []
            for name, group in self.preprocessor.groups.items():
                if group.is_onehot or group.dtype not in (FeatureType.COUNT, FeatureType.BINARY):
                    continue
                index = group.indices[0]
                if group.log_scaled:
                    span = group.scale_max - group.scale_min
                    original = torch.expm1(x_adv[:, index].double() * span + group.scale_min)
                else:
                    original = x_adv[:, index].double()
                # Relative tolerance: a residual of 0.01 is meaningless on a value of 1e6
                # but significant on a value of 3.
                tolerance = torch.clamp(original.abs() * 1e-6, min=1e-3)
                fractional.append((original - original.round()).abs() > tolerance)
            if fractional:
                report["integrality_violation_rate"] = float(
                    torch.stack(fractional).any(dim=0).float().mean()
                )

        any_violation = max(
            (v for k, v in report.items() if k.endswith("_violation_rate")), default=0.0
        )
        report["any_violation_rate_max"] = any_violation
        return report

    def perturbation_in_original_units(
        self, x_adv: torch.Tensor, x_orig: torch.Tensor, feature: str
    ) -> np.ndarray:
        """
        Translate a perturbation back into bytes / seconds / packets.

        Essential for interpreting results: "epsilon = 0.05" is meaningless to a network
        engineer, whereas "the attack adds a median of 340 bytes" is actionable.
        """
        adv = self.preprocessor.inverse_numeric(x_adv.detach().cpu().numpy(), feature)
        orig = self.preprocessor.inverse_numeric(x_orig.detach().cpu().numpy(), feature)
        return adv - orig
