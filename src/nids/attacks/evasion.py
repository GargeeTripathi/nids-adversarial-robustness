"""
Evasion attacks: test-time perturbations that make an attack flow look benign.

Threat model
------------
The adversary has already compromised something and is generating malicious traffic. They
want the NIDS to label that traffic benign. They may perturb only their *own* flows, only
in ways they can physically produce (enforced by `ConstraintSet`), and they do not get to
modify the model or its training data - that is the poisoning threat model, handled in
`poisoning.py`.

Three attacks are implemented, in increasing order of realism-of-assumption:

* **FGSM** (Goodfellow et al., 2015) - single gradient step. White-box: needs the model's
  gradients. Cheap, and a useful lower bound on vulnerability.
* **PGD** (Madry et al., 2018) - iterative FGSM with projection back into the epsilon-ball
  each step. Still white-box, but a far stronger attack, and the standard benchmark for
  robustness claims.
* **Black-box transfer** - the realistic one. The adversary cannot see the deployed model,
  so they train their own *surrogate*, attack that, and hope the adversarial examples
  transfer. This is the threat an operational NIDS actually faces, and its success rate is
  the headline number for RO1.

The `constraints` argument threads RO2 through all of them: when supplied, every iterate is
projected onto the realizable-traffic set, so the resulting adversarial examples correspond
to flows an attacker could actually emit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraints import ConstraintSet


def _as_tensor(x, device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    return x.to(device)


# ======================================================================================
# FGSM
# ======================================================================================


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    epsilon: float = 0.05,
    constraints: ConstraintSet | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

        x_adv = x + eps · sign(∇_x L(f(x), y))

    Moves one step of size `epsilon` in the direction that most increases the loss of the
    *true* label - i.e. away from being classified correctly.
    """
    model = model.to(device).eval()
    x = _as_tensor(x, device)
    y = _as_tensor(y, device).long() if not isinstance(y, torch.Tensor) else y.to(device).long()

    x_original = x.clone().detach()
    x_adv = x.clone().detach().requires_grad_(True)

    loss = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]

    x_adv = x_adv.detach() + epsilon * grad.sign()

    if constraints is not None:
        x_adv = constraints.project(x_adv, x_original)
    else:
        x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


# ======================================================================================
# PGD
# ======================================================================================


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    epsilon: float = 0.05,
    alpha: float | None = None,
    steps: int = 20,
    random_start: bool = True,
    constraints: ConstraintSet | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Projected Gradient Descent (L-infinity).

    Iterates FGSM-sized steps, projecting back into the epsilon-ball around `x` after each,
    and - when constraints are supplied - onto the realizable set as well. The double
    projection is the technically interesting part of applying PGD to network traffic: the
    iterate must satisfy both the perturbation budget and the traffic semantics, and the
    intersection can be much smaller than either alone.

    Args:
        alpha: per-step size. Defaults to `2.5 * epsilon / steps`, the standard heuristic
            giving enough total movement to traverse the ball a couple of times.
    """
    model = model.to(device).eval()
    x = _as_tensor(x, device)
    y = _as_tensor(y, device).long() if not isinstance(y, torch.Tensor) else y.to(device).long()
    alpha = alpha if alpha is not None else max(2.5 * epsilon / max(steps, 1), 1e-4)

    x_original = x.clone().detach()

    if random_start:
        # Random init helps escape the flat-gradient regions that make single-step FGSM
        # underestimate vulnerability.
        noise = torch.empty_like(x_original).uniform_(-epsilon, epsilon)
        x_adv = (x_original + noise).clamp(0.0, 1.0)
        if constraints is not None:
            x_adv = constraints.project(x_adv, x_original)
    else:
        x_adv = x_original.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * grad.sign()

        # Projection 1: back into the L-inf epsilon-ball around the original point.
        delta = (x_adv - x_original).clamp(-epsilon, epsilon)
        x_adv = x_original + delta

        # Projection 2: onto the realizable-traffic set.
        if constraints is not None:
            x_adv = constraints.project(x_adv, x_original)
        else:
            x_adv = x_adv.clamp(0.0, 1.0)

    return x_adv.detach()


# ======================================================================================
# Black-box transfer
# ======================================================================================


def transfer_attack(
    surrogate_model: nn.Module,
    target_model: nn.Module,
    x: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    epsilon: float = 0.05,
    steps: int = 20,
    constraints: ConstraintSet | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict]:
    """
    Craft adversarial examples on a surrogate and evaluate them on an unseen target.

    Returns `(x_adv, info)` where `info` records success on both models. The gap between
    them is the transfer gap: high surrogate success with low target success means the
    adversarial examples are model-specific rather than capturing a shared blind spot.
    """
    x = _as_tensor(x, device)
    y = _as_tensor(y, device).long() if not isinstance(y, torch.Tensor) else y.to(device).long()

    x_adv = pgd_attack(
        surrogate_model, x, y,
        epsilon=epsilon, steps=steps, constraints=constraints, device=device,
    )

    with torch.no_grad():
        surrogate_pred = surrogate_model.to(device).eval()(x_adv).argmax(1)
        target_clean_pred = target_model.to(device).eval()(x).argmax(1)
        target_adv_pred = target_model(x_adv).argmax(1)

    # Success is only meaningful on samples the target originally got right; flipping an
    # already-misclassified sample is not an attack achievement.
    correct_before = target_clean_pred == y
    flipped = correct_before & (target_adv_pred != y)

    info = {
        "surrogate_success_rate": float((surrogate_pred != y).float().mean()),
        "target_success_rate": float(
            flipped.sum() / correct_before.sum().clamp(min=1)
        ),
        "target_accuracy_clean": float(correct_before.float().mean()),
        "target_accuracy_adv": float((target_adv_pred == y).float().mean()),
    }
    return x_adv.detach(), info


# ======================================================================================
# Evaluation helper
# ======================================================================================


def evaluate_evasion(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    attack_fn,
    attack_kwargs: dict | None = None,
    batch_size: int = 512,
    device: str = "cpu",
    attack_class_only: bool = True,
) -> dict:
    """
    Run an evasion attack over a dataset and measure its effect.

    Args:
        attack_class_only: restrict to true-attack samples (y == 1). This matters. An
            evasion adversary wants malicious traffic classified benign; perturbing benign
            traffic into the attack class is a *false-positive* injection, a different
            threat with different consequences. Mixing both into one "attack success rate"
            is a common reporting error that makes numbers incomparable across papers.
    """
    attack_kwargs = attack_kwargs or {}
    model = model.to(device).eval()

    if attack_class_only:
        mask = y == 1
        x, y = x[mask], y[mask]
        if len(x) == 0:
            return {"error": "no attack-class samples present"}

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    clean_correct, adv_correct, flipped, evaluated = 0, 0, 0, 0
    perturbation_linf, perturbation_l2 = [], []
    adversarial_batches = []

    for start in range(0, len(x_tensor), batch_size):
        xb = x_tensor[start:start + batch_size].to(device)
        yb = y_tensor[start:start + batch_size].to(device)

        x_adv = attack_fn(model, xb, yb, device=device, **attack_kwargs)

        with torch.no_grad():
            clean_pred = model(xb).argmax(1)
            adv_pred = model(x_adv).argmax(1)

        was_correct = clean_pred == yb
        clean_correct += int(was_correct.sum())
        adv_correct += int((adv_pred == yb).sum())
        flipped += int((was_correct & (adv_pred != yb)).sum())
        evaluated += len(xb)

        delta = (x_adv - xb)
        perturbation_linf.append(delta.abs().amax(dim=1).cpu().numpy())
        perturbation_l2.append(delta.norm(dim=1).cpu().numpy())
        adversarial_batches.append(x_adv.cpu().numpy())

    linf = np.concatenate(perturbation_linf)
    l2 = np.concatenate(perturbation_l2)

    return {
        "n_samples": evaluated,
        "clean_accuracy": clean_correct / evaluated,
        "adversarial_accuracy": adv_correct / evaluated,
        "attack_success_rate": flipped / max(clean_correct, 1),
        "accuracy_drop": (clean_correct - adv_correct) / evaluated,
        "perturbation_linf_mean": float(linf.mean()),
        "perturbation_linf_max": float(linf.max()),
        "perturbation_l2_mean": float(l2.mean()),
        "x_adv": np.concatenate(adversarial_batches),
    }
