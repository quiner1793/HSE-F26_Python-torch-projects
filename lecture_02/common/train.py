"""Один цикл обучения для frozen backbone и partial fine-tuning."""

from pathlib import Path

import torch


def configure_training(model, learning_rate, backbone_lr=None):
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


def train_model(model, train_loader, device, epochs, learning_rate, momentum, backbone_lr=None,
                history=None):
    model = model.to(device)
    groups = configure_training(model, learning_rate, backbone_lr)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(groups, momentum=momentum)

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
        print(f"epoch={epoch + 1}/{epochs} loss={running_loss / total:.4f} " f"train_accuracy={correct / total:.2%}")
        if history is not None:
            history.append({"epoch": epoch + 1, "loss": running_loss / total,
                            "train_accuracy": correct / total})
    return model


def save_model(model, classes, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": list(classes)}, path)


def load_model(model, path, device, expected_classes=None):
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
