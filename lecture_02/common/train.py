"""Один цикл обучения для frozen backbone и partial fine-tuning."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def configure_training(model: nn.Module, learning_rate: float, backbone_lr: float | None = None) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    groups = [{"params": model.fc.parameters(), "lr": learning_rate}]
    if backbone_lr is not None:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        groups.insert(0, {"params": model.layer4.parameters(), "lr": backbone_lr})
    return groups


def _validation_metrics(model: nn.Module, loader: DataLoader, device, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    if not total:
        raise ValueError("Validation-датасет пуст")
    return loss_sum / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    backbone_lr: float | None = None,
    history: list[dict] | None = None,
    validation_loader: DataLoader | None = None,
    early_stopping_patience: int | None = None,
    scheduler_patience: int | None = None,
    scheduler_factor: float = 0.3,
    min_delta: float = 0.0,
) -> nn.Module:
    model = model.to(device)
    groups = configure_training(model, learning_rate, backbone_lr)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(groups, momentum=momentum)
    scheduler = None
    if validation_loader is not None and scheduler_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Сохраняем прежнее поведение BatchNorm: running statistics обновляются.
        model.train()
        running_loss = correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        if not total:
            raise ValueError("Обучающий датасет пуст")
        values = {
            "epoch": epoch + 1,
            "loss": running_loss / total,
            "train_accuracy": correct / total,
        }
        status = (
            f"epoch={epoch + 1}/{epochs} loss={values['loss']:.4f} " f"train_accuracy={values['train_accuracy']:.2%}"
        )
        if validation_loader is not None:
            validation_loss, validation_accuracy = _validation_metrics(model, validation_loader, device, criterion)
            improved = validation_loss < best_loss - min_delta
            if improved:
                best_loss = validation_loss
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step(validation_loss)
            values.update(
                {
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "best": improved,
                }
            )
            status += f" val_loss={validation_loss:.4f} " f"val_accuracy={validation_accuracy:.2%}"
        print(status)
        if history is not None:
            history.append(values)
        if (
            validation_loader is not None
            and early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"early stopping: epoch={epoch + 1}, best_val_loss={best_loss:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save_model(model: nn.Module, classes: tuple[str, ...] | list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": list(classes)}, path)


def load_model(
    model: nn.Module,
    path: str | Path,
    device,
    expected_classes: tuple[str, ...] | list[str] | None = None,
) -> tuple[nn.Module, list[str]]:
    print(f"Loading model: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    if expected_classes is not None and list(classes) != list(expected_classes):
        raise ValueError(
            "Классы или их порядок в checkpoint не совпадают с экспериментом. "
            "Выберите другой model_path или переобучите модель."
        )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device), classes
