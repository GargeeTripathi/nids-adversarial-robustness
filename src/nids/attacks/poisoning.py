"""
Poisoning attacks: corrupting the *training* data so the deployed model is born defective.

Threat model
------------
Different from evasion, and often more realistic for NIDS specifically. Intrusion detectors
are routinely retrained on traffic harvested from the network they protect, and the labels
come from imperfect sources - analyst triage, signature engines, honeypots, or simply
"whatever was flowing on a day we assume was clean". An adversary who can influence that
pipeline shapes the model itself.

Two families are implemented:

* **Label flipping** - the adversary changes labels on a fraction of training data. Loud
  but powerful, and realistic wherever labelling is crowdsourced or semi-automated. The
  targeted variant (attack -> benign only) is the operationally sensible one: it teaches
  the model that malicious patterns are normal, which is what an intruder wants. It is also
  stealthier against class-balance audits than symmetric flipping.

* **Clean-label** - the adversary perturbs *features* while leaving labels correct. Far
  stealthier: every poisoned record survives human review, because its label genuinely
  matches its (stated) class. Implemented here as feature-collision poisoning: benign
  training points are nudged toward the attack manifold, blurring the boundary so that at
  test time real attacks fall on the benign side.

The measured quantity in both cases is the *damage* to clean-data detection, not attack
success on the poisoned points themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoisonResult:
    """Poisoned training data plus a record of exactly what was altered."""

    x_poisoned: np.ndarray
    y_poisoned: np.ndarray
    poisoned_indices: np.ndarray
    poison_rate: float
    strategy: str
    details: dict


# ======================================================================================
# Label flipping
# ======================================================================================


def label_flip_poison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    poison_rate: float = 0.1,
    strategy: str = "attack_to_benign",
    seed: int = 42,
) -> PoisonResult:
    """
    Flip the labels of a fraction of training samples.

    Args:
        poison_rate: fraction of the *whole* training set to poison. Reported this way (not
            as a fraction of the affected class) so that rates are comparable across
            datasets with different class balances.
        strategy:
            - `attack_to_benign`: relabel attacks as benign. The adversary's goal - induces
              false negatives, the costly error for a NIDS.
            - `benign_to_attack`: relabel benign as attack. Induces false positives; a
              denial-of-service on the analyst team rather than on the network.
            - `symmetric`: flip both directions at random. The classic academic setting,
              included as a reference point.
    """
    rng = np.random.default_rng(seed)
    y_poisoned = y_train.copy()
    n_poison = int(len(y_train) * poison_rate)

    if strategy == "attack_to_benign":
        candidates = np.flatnonzero(y_train == 1)
    elif strategy == "benign_to_attack":
        candidates = np.flatnonzero(y_train == 0)
    elif strategy == "symmetric":
        candidates = np.arange(len(y_train))
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    n_poison = min(n_poison, len(candidates))
    chosen = rng.choice(candidates, size=n_poison, replace=False)
    y_poisoned[chosen] = 1 - y_poisoned[chosen]

    return PoisonResult(
        x_poisoned=x_train.copy(),
        y_poisoned=y_poisoned,
        poisoned_indices=chosen,
        poison_rate=poison_rate,
        strategy=f"label_flip_{strategy}",
        details={
            "n_poisoned": int(n_poison),
            "requested_rate": poison_rate,
            "effective_rate": n_poison / len(y_train),
            "candidate_pool": int(len(candidates)),
        },
    )


# ======================================================================================
# Clean-label
# ======================================================================================


def clean_label_poison(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    poison_rate: float = 0.1,
    epsilon: float = 0.1,
    steps: int = 40,
    constraints=None,
    device: str = "cpu",
    seed: int = 42,
) -> PoisonResult:
    """
    Feature-collision clean-label poisoning.

    Benign training points are perturbed - within a budget, and within realizable-traffic
    constraints if supplied - so that their *feature representation* drifts toward attack
    traffic while their benign label is retained. A model fitted to this data learns a
    boundary shifted into attack territory, so genuine attacks land on the benign side.

    Why this is the dangerous one: nothing about a poisoned record is detectably mislabelled.
    An analyst auditing the training set sees benign traffic labelled benign. Only the
    features are subtly off, and "subtly off" is indistinguishable from natural variation in
    a dataset with millions of flows.

    Implementation: gradient *descent* on the loss w.r.t. the attack class - the opposite
    sign to evasion, since here we want the point to look more like the target class while
    keeping its original label.
    """
    rng = np.random.default_rng(seed)
    model = model.to(device).eval()

    benign_indices = np.flatnonzero(y_train == 0)
    n_poison = min(int(len(y_train) * poison_rate), len(benign_indices))
    chosen = rng.choice(benign_indices, size=n_poison, replace=False)

    x_poisoned = x_train.copy()
    x_subset = torch.tensor(x_train[chosen], dtype=torch.float32, device=device)
    x_original = x_subset.clone()

    # Target: the attack class. We pull benign features toward it.
    target = torch.ones(len(x_subset), dtype=torch.long, device=device)
    alpha = max(2.5 * epsilon / max(steps, 1), 1e-4)

    x_adv = x_subset.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), target)
        grad = torch.autograd.grad(loss, x_adv)[0]

        # Descent (minus) - reduce loss w.r.t. the attack class, i.e. look more like it.
        x_adv = x_adv.detach() - alpha * grad.sign()

        delta = (x_adv - x_original).clamp(-epsilon, epsilon)
        x_adv = x_original + delta

        if constraints is not None:
            x_adv = constraints.project(x_adv, x_original)
        else:
            x_adv = x_adv.clamp(0.0, 1.0)

    x_poisoned[chosen] = x_adv.detach().cpu().numpy()

    with torch.no_grad():
        drift = float(
            (torch.softmax(model(x_adv), dim=1)[:, 1]
             - torch.softmax(model(x_original), dim=1)[:, 1]).mean()
        )

    return PoisonResult(
        x_poisoned=x_poisoned,
        y_poisoned=y_train.copy(),          # labels deliberately untouched
        poisoned_indices=chosen,
        poison_rate=poison_rate,
        strategy="clean_label_feature_collision",
        details={
            "n_poisoned": int(n_poison),
            "requested_rate": poison_rate,
            # Clamped to the benign pool, exactly as label flipping is clamped to the
            # attack pool: only benign records can be nudged toward the attack manifold
            # while keeping a truthful label, so the achievable rate saturates.
            "effective_rate": n_poison / len(y_train),
            "epsilon": epsilon,
            "steps": steps,
            "mean_attack_probability_drift": drift,
            "constrained": constraints is not None,
        },
    )


# ======================================================================================
# Evaluation
# ======================================================================================


def evaluate_poisoning(
    clean_metrics: dict,
    poisoned_metrics: dict,
    poison_result: PoisonResult,
) -> dict:
    """
    Quantify poisoning damage by comparing a clean-trained and poison-trained model, both
    evaluated on the same *clean* test set.

    Evaluating on clean data is the point: a successful poisoning attack degrades the
    model's real-world behaviour permanently, regardless of what the attacker sends later.
    """
    return {
        "strategy": poison_result.strategy,
        "poison_rate": poison_result.poison_rate,
        "n_poisoned": poison_result.details.get("n_poisoned"),
        "clean_accuracy": clean_metrics["accuracy"],
        "poisoned_accuracy": poisoned_metrics["accuracy"],
        "accuracy_drop": clean_metrics["accuracy"] - poisoned_metrics["accuracy"],
        "clean_detection_rate": clean_metrics["detection_rate"],
        "poisoned_detection_rate": poisoned_metrics["detection_rate"],
        # The headline number: how much attack traffic now slips through undetected.
        "detection_rate_drop": (
            clean_metrics["detection_rate"] - poisoned_metrics["detection_rate"]
        ),
        "clean_false_positive_rate": clean_metrics["false_positive_rate"],
        "poisoned_false_positive_rate": poisoned_metrics["false_positive_rate"],
        "false_positive_rate_change": (
            poisoned_metrics["false_positive_rate"] - clean_metrics["false_positive_rate"]
        ),
        "details": poison_result.details,
    }
