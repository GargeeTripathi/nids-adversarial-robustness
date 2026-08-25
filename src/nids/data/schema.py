"""
Feature schema and semantics for network-flow data.

This module is the backbone of both research objectives:

RO1 (cross-domain transferability) needs a *canonical feature space*. NSL-KDD has 41
features, UNSW-NB15 has 49, CICIDS2017 has 78, and they disagree on names, units and
direction conventions. A model trained on one cannot be evaluated on another unless the
features are first mapped into a shared space. `CANONICAL_FEATURES` defines that space and
the per-dataset `*_MAPPING` dicts define the translation.

RO2 (realistic perturbations) needs to know what each feature *means* physically. An
unconstrained gradient attack will happily propose a flow with -3.7 packets, a duration of
0.2 seconds carrying 900 MB, or a "protocol type" of 1.4. Such adversarial examples inflate
reported attack success rates but cannot be transmitted onto a real network. `FeatureSpec`
records the type, bounds, mutability and attacker-controllability of every feature so that
`nids.attacks.constraints` can project perturbations back onto the realizable set.

Terminology note for readers new to the area
--------------------------------------------
A *flow* is a sequence of packets sharing a 5-tuple (source IP, source port, destination
IP, destination port, protocol). Flow-based NIDS do not inspect payload bytes; they work
from summary statistics of the flow such as its duration, byte counts, and packet rates.
Every dataset used here is flow-based (or connection-based, in NSL-KDD's older
terminology).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FeatureType(str, Enum):
    """Physical type of a feature, which determines how it may be perturbed."""

    CONTINUOUS = "continuous"      # real-valued, e.g. duration in seconds
    COUNT = "count"                # non-negative integer, e.g. number of packets
    BINARY = "binary"              # 0/1 indicator, e.g. "logged in successfully"
    CATEGORICAL = "categorical"    # unordered discrete, e.g. protocol (tcp/udp/icmp)
    RATE = "rate"                  # ratio in [0, 1], e.g. % of connections with errors


class Mutability(str, Enum):
    """
    How much control a realistic attacker has over a feature.

    This is the crux of RO2. Standard adversarial ML assumes the attacker can nudge every
    input dimension freely, which is true for image pixels but false for network flows.
    An attacker who controls a botnet can pad packets, insert delays, or open extra
    connections. They cannot reach into the victim's NIC and decrement a counter, and they
    cannot change what the receiving host observed.
    """

    IMMUTABLE = "immutable"
    """Cannot be changed without changing the attack itself, or is set by the victim/
    network rather than the attacker. Example: `protocol_type` - a TCP SYN flood is not a
    TCP SYN flood if you switch it to UDP. Also covers server-side observations such as
    `num_failed_logins`, which the server counts, not the client."""

    INCREASE_ONLY = "increase_only"
    """The attacker can add but not remove. This is the single most important realistic
    constraint. An attacker can send *more* packets, pad payloads to make flows *larger*,
    or wait to make a flow *longer*. They cannot un-send a packet or retroactively shorten
    a duration. Example: `duration`, `src_bytes`, `count`."""

    FREE = "free"
    """Genuinely attacker-controlled in both directions within the feature's valid range,
    typically because the attacker can choose the value up-front when crafting traffic."""

    DERIVED = "derived"
    """Not independently settable at all: a deterministic function of other features.
    Example: `srv_error_rate` is a ratio computed from counts. Perturbing it directly
    produces an arithmetically impossible flow record. These must be *recomputed* after
    perturbing their parents, never perturbed themselves."""


@dataclass(frozen=True)
class FeatureSpec:
    """Complete description of one canonical feature."""

    name: str
    dtype: FeatureType
    mutability: Mutability
    description: str
    minimum: float = 0.0
    maximum: float | None = None          # None => unbounded above
    unit: str = ""
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    """For DERIVED features: the canonical features this one is computed from."""

    def is_perturbable(self) -> bool:
        """Whether a gradient attack is allowed to modify this feature directly."""
        return self.mutability in (Mutability.FREE, Mutability.INCREASE_ONLY)


# ======================================================================================
# Canonical feature space
# ======================================================================================
# Chosen as the intersection of concepts genuinely present in all three datasets. Keeping
# this deliberately small (rather than unioning everything and zero-filling) matters: a
# cross-domain result is only meaningful if the model sees the *same physical quantity* in
# both domains. Zero-filled absent features would let a model learn "this feature is always
# zero => I must be in domain A", which inflates apparent transfer performance.

CANONICAL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        name="duration",
        dtype=FeatureType.CONTINUOUS,
        mutability=Mutability.INCREASE_ONLY,
        description="Wall-clock length of the flow. An attacker can stall to lengthen a "
                    "flow but cannot shorten one already observed.",
        unit="seconds",
    ),
    FeatureSpec(
        name="protocol_type",
        dtype=FeatureType.CATEGORICAL,
        mutability=Mutability.IMMUTABLE,
        description="Transport protocol (tcp/udp/icmp). Immutable because changing it "
                    "changes the nature of the attack rather than evading detection of it.",
    ),
    FeatureSpec(
        name="service",
        dtype=FeatureType.CATEGORICAL,
        mutability=Mutability.IMMUTABLE,
        description="Destination application service (http, ftp, dns...). Determined by "
                    "the target the attacker wants to reach.",
    ),
    FeatureSpec(
        name="flag",
        dtype=FeatureType.CATEGORICAL,
        mutability=Mutability.IMMUTABLE,
        description="Connection state/status summary (SF, REJ, S0...). Reflects how the "
                    "victim and network responded, not what the attacker chose.",
    ),
    FeatureSpec(
        name="src_bytes",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Bytes sent source->destination. Attacker may pad with junk to "
                    "increase; cannot un-send bytes.",
        unit="bytes",
    ),
    FeatureSpec(
        name="dst_bytes",
        dtype=FeatureType.COUNT,
        mutability=Mutability.IMMUTABLE,
        description="Bytes sent destination->source. Produced by the victim host, so it "
                    "is outside the attacker's direct control.",
        unit="bytes",
    ),
    FeatureSpec(
        name="src_packets",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Packets sent source->destination. Attacker may inject additional "
                    "packets.",
        unit="packets",
    ),
    FeatureSpec(
        name="dst_packets",
        dtype=FeatureType.COUNT,
        mutability=Mutability.IMMUTABLE,
        description="Packets sent destination->source; victim-generated.",
        unit="packets",
    ),
    FeatureSpec(
        name="count",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Connections to the same destination host in the recent time window. "
                    "Attacker can open more connections.",
        unit="connections",
    ),
    FeatureSpec(
        name="srv_count",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Connections to the same service in the recent time window.",
        unit="connections",
    ),
    FeatureSpec(
        name="serror_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="Fraction of connections in the window with SYN errors. A ratio over "
                    "`count`, so it must be recomputed rather than perturbed.",
        maximum=1.0,
        depends_on=("count",),
    ),
    FeatureSpec(
        name="rerror_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="Fraction of connections in the window with REJ errors.",
        maximum=1.0,
        depends_on=("count",),
    ),
    FeatureSpec(
        name="same_srv_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="Fraction of connections in the window to the same service.",
        maximum=1.0,
        depends_on=("count", "srv_count"),
    ),
    FeatureSpec(
        name="diff_srv_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="Fraction of connections in the window to different services. "
                    "Complement-linked to same_srv_rate.",
        maximum=1.0,
        depends_on=("count", "srv_count"),
    ),
    FeatureSpec(
        name="dst_host_count",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Connections seen to the destination host (longer window).",
        unit="connections",
    ),
    FeatureSpec(
        name="dst_host_srv_count",
        dtype=FeatureType.COUNT,
        mutability=Mutability.INCREASE_ONLY,
        description="Connections seen to the destination host's same service.",
        unit="connections",
    ),
    FeatureSpec(
        name="dst_host_same_srv_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="Ratio of dst_host_srv_count to dst_host_count.",
        maximum=1.0,
        depends_on=("dst_host_count", "dst_host_srv_count"),
    ),
    FeatureSpec(
        name="dst_host_serror_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="SYN-error fraction over the destination-host window.",
        maximum=1.0,
        depends_on=("dst_host_count",),
    ),
    FeatureSpec(
        name="dst_host_rerror_rate",
        dtype=FeatureType.RATE,
        mutability=Mutability.DERIVED,
        description="REJ-error fraction over the destination-host window.",
        maximum=1.0,
        depends_on=("dst_host_count",),
    ),
    FeatureSpec(
        name="land",
        dtype=FeatureType.BINARY,
        mutability=Mutability.FREE,
        description="1 if source and destination address/port are identical (a LAND "
                    "attack). Attacker-chosen when crafting packets.",
        maximum=1.0,
    ),
)

CANONICAL_NAMES: tuple[str, ...] = tuple(spec.name for spec in CANONICAL_FEATURES)
SPEC_BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in CANONICAL_FEATURES}


# ======================================================================================
# Per-dataset mappings into the canonical space
# ======================================================================================
# Each maps canonical name -> source column name. A canonical feature absent from a dataset
# is simply omitted here; loaders report the coverage so that gaps are visible rather than
# silently zero-filled.

NSL_KDD_MAPPING: dict[str, str] = {
    "duration": "duration",
    "protocol_type": "protocol_type",
    "service": "service",
    "flag": "flag",
    "src_bytes": "src_bytes",
    "dst_bytes": "dst_bytes",
    "count": "count",
    "srv_count": "srv_count",
    "serror_rate": "serror_rate",
    "rerror_rate": "rerror_rate",
    "same_srv_rate": "same_srv_rate",
    "diff_srv_rate": "diff_srv_rate",
    "dst_host_count": "dst_host_count",
    "dst_host_srv_count": "dst_host_srv_count",
    "dst_host_same_srv_rate": "dst_host_same_srv_rate",
    "dst_host_serror_rate": "dst_host_serror_rate",
    "dst_host_rerror_rate": "dst_host_rerror_rate",
    "land": "land",
    # NSL-KDD has no per-direction packet counts; src_packets/dst_packets are unavailable.
}

UNSW_NB15_MAPPING: dict[str, str] = {
    "duration": "dur",
    "protocol_type": "proto",
    "service": "service",
    "flag": "state",
    "src_bytes": "sbytes",
    "dst_bytes": "dbytes",
    "src_packets": "spkts",
    "dst_packets": "dpkts",
    "count": "ct_dst_ltm",
    "srv_count": "ct_srv_dst",
    "dst_host_count": "ct_dst_sport_ltm",
    "dst_host_srv_count": "ct_srv_src",
    # UNSW-NB15 reports connection-state counts rather than error *rates*; the rate
    # features are recomputed downstream where possible.
}

CICIDS2017_MAPPING: dict[str, str] = {
    "duration": "Flow Duration",
    "protocol_type": "Protocol",
    "dst_host_count": "Destination Port",
    "src_bytes": "Total Length of Fwd Packets",
    "dst_bytes": "Total Length of Bwd Packets",
    "src_packets": "Total Fwd Packets",
    "dst_packets": "Total Backward Packets",
    # CICIDS2017 is purely flow-statistical: it has no service/flag equivalents and no
    # time-window connection counts, so those canonical features are unavailable.
}

DATASET_MAPPINGS: dict[str, dict[str, str]] = {
    "nsl_kdd": NSL_KDD_MAPPING,
    "unsw_nb15": UNSW_NB15_MAPPING,
    "cicids2017": CICIDS2017_MAPPING,
}


def shared_features(*dataset_names: str) -> list[str]:
    """
    Canonical features available in *every* named dataset.

    This is the feature space a cross-domain experiment must use. Training on the full
    NSL-KDD canonical set and testing on CICIDS2017 would be meaningless, because the
    columns would not correspond to the same physical quantities.
    """
    if not dataset_names:
        return list(CANONICAL_NAMES)
    common = set(DATASET_MAPPINGS[dataset_names[0]])
    for name in dataset_names[1:]:
        common &= set(DATASET_MAPPINGS[name])
    return [n for n in CANONICAL_NAMES if n in common]   # keep canonical ordering


def coverage_report(dataset_name: str) -> dict[str, bool]:
    """Which canonical features the dataset provides. Used by loaders for logging."""
    mapping = DATASET_MAPPINGS[dataset_name]
    return {name: name in mapping for name in CANONICAL_NAMES}
