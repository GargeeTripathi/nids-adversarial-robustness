"""
Training loop shared by all four architectures.

Deliberately one implementation rather than per-model scripts: if the CNN were trained for
more epochs or with a different learning-rate schedule than the Transformer, any robustness
difference between them would be confounded by training setup rather than architecture.
Identical optimisation for every model is what licenses the RO1 comparison.

Adversarial training is supported here too (`adversarial_config`), since it is the standard
defence baseline any robustness study needs for comparison.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6                 # early-stopping patience on validation loss
    device: str = "cpu"
    seed: int = 42
    class_weighting: bool = True
    """NSL-KDD is near-balanced but CICIDS2017 is heavily benign-skewed; weighting keeps the
    same config usable across datasets without silently collapsing to majority prediction."""
    verbose: bool = True


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epochs_run: int = 0
    best_epoch: int = 0
    seconds: float = 0.0


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    # drop_last on the training loader only: BatchNorm (in the CNN) fails on a batch of 1,
    # which can occur on the final partial batch.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
    adversarial_config: dict | None = None,
) -> tuple[nn.Module, TrainHistory]:
    """
    Train one model, returning the best-validation-loss weights.

    Args:
        adversarial_config: if given, enables adversarial training. Expected keys:
            `attack` (callable(model, x, y) -> x_adv), `ratio` (fraction of each batch
            replaced by adversarial examples, default 0.5).
    """
    config = config or TrainConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    if config.class_weighting:
        counts = np.bincount(y_train, minlength=2).astype(np.float64)
        weights = counts.sum() / (2.0 * np.maximum(counts, 1))
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    train_loader = make_loader(x_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, config.batch_size, shuffle=False)

    history = TrainHistory()
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(config.epochs):
        model.train()
        running_loss, correct, seen = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            if adversarial_config is not None:
                attack = adversarial_config["attack"]
                ratio = adversarial_config.get("ratio", 0.5)
                n_adv = int(len(xb) * ratio)
                if n_adv > 0:
                    # Generate under eval() so BatchNorm/Dropout do not make the attack
                    # gradient depend on other samples in the batch.
                    model.eval()
                    x_adv = attack(model, xb[:n_adv], yb[:n_adv])
                    model.train()
                    xb = torch.cat([x_adv.detach(), xb[n_adv:]], dim=0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(xb)

        train_loss = running_loss / max(seen, 1)
        train_acc = correct / max(seen, 1)

        model.eval()
        val_loss, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_seen += len(xb)
        val_loss /= max(val_seen, 1)
        val_acc = val_correct / max(val_seen, 1)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.epochs_run = epoch + 1
        scheduler.step(val_loss)

        if config.verbose:
            print(f"  epoch {epoch+1:3d}/{config.epochs}  "
                  f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if config.verbose:
                    print(f"  early stopping at epoch {epoch+1} "
                          f"(best was {history.best_epoch})")
                break

    model.load_state_dict(best_state)
    history.seconds = time.time() - started
    return model, history


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray | torch.Tensor, device: str = "cpu",
            batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_labels, attack_probabilities)."""
    model = model.to(device).eval()
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    predictions, probabilities = [], []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size].to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        predictions.append(logits.argmax(1).cpu().numpy())
        probabilities.append(probs[:, 1].cpu().numpy())

    return np.concatenate(predictions), np.concatenate(probabilities)
