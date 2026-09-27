import os

import torch

from lecture_02.common.config import (
    DEVICE,
    DATA_ROOT,
    MODELS_DIR,
    RESULTS_DIR,
)

from lecture_02.common.dataset import create_datasets, create_loaders
from lecture_02.common.model import create_model
from lecture_02.common.evaluate import evaluate
from lecture_02.common.train import save_model

# ============================================================
# EXPERIMENT 02.2
# Partial Fine-Tuning ResNet18
#
# Цель:
# проверить гипотезу, что падение accuracy связано с тем,
# что полностью замороженный ImageNet backbone недостаточно
# хорошо разделяет визуально похожие породы.
#
# В отличие от 02_25_classes.py:
# - layer1-layer3 заморожены
# - layer4 обучается
# - fc обучается
# ============================================================


SELECTED_CLASSES = [
    "Abyssinian",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British Shorthair",
    "Chihuahua",
    "Egyptian Mau",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Maine Coon",
    "Miniature Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
]


BATCH_SIZE = 32
EPOCHS = 5
MOMENTUM = 0.9

# Для pretrained layer4 нужен маленький learning rate.
BACKBONE_LR = 0.0001

# Новый классификатор можно обучать быстрее.
FC_LR = 0.01


MODEL_PATH = f"{MODELS_DIR}/model_25_classes_partial_finetune.pth"
EXPERIMENT_RESULTS_DIR = f"{RESULTS_DIR}/02_2_partial_finetune"


def configure_partial_finetuning(model):
    """
    Замораживаем всю сеть,
    затем размораживаем только layer4 и fc.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def train_partial_finetuning(
    model,
    train_loader,
    device,
    epochs,
):
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Разные learning rate:
    #
    # layer4 уже pretrained -> меняем веса осторожно
    # fc новый -> можно обучать значительно быстрее
    optimizer = torch.optim.SGD(
        [
            {
                "params": model.layer4.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.fc.parameters(),
                "lr": FC_LR,
            },
        ],
        momentum=MOMENTUM,
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"loss={epoch_loss:.4f} "
            f"train_accuracy={epoch_accuracy:.2%}"
        )

    return model


def main():

    print("device:", DEVICE)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, weights = create_model(len(SELECTED_CLASSES))

    transform = weights.transforms()

    model = configure_partial_finetuning(model)

    print("\nОбучаемые части модели:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(" -", name)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_ds, test_ds = create_datasets(
        DATA_ROOT,
        SELECTED_CLASSES,
        transform,
    )

    train_loader, test_loader = create_loaders(
        train_ds,
        test_ds,
        BATCH_SIZE,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        print(
            f"\nМодель уже существует: {MODEL_PATH}"
        )

        print("Загружаем сохранённую модель...")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model = model.to(DEVICE)

    else:

        print("\nЗапускаем partial fine-tuning...")

        print(
            f"layer4 learning rate = {BACKBONE_LR}"
        )

        print(
            f"fc learning rate = {FC_LR}"
        )

        model = train_partial_finetuning(
            model,
            train_loader,
            DEVICE,
            EPOCHS,
        )

        save_model(
            model,
            SELECTED_CLASSES,
            MODEL_PATH,
        )

        print(
            f"\nМодель сохранена: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\nTEST:")

    accuracy = evaluate(
        model,
        test_loader,
        DEVICE,
        classes=SELECTED_CLASSES,
        results_dir=EXPERIMENT_RESULTS_DIR,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("EXPERIMENT 02.2 — PARTIAL FINE-TUNING")

    print("=" * 60)

    print(f"Accuracy: {accuracy:.2%}")

    print("\nBaseline 02.1:")
    print("Accuracy: 90.65%")

    difference = accuracy - 0.9065

    print(
        f"\nИзменение: {difference:+.2%}"
    )

    print(
        f"\nРезультаты сохранены в: "
        f"{EXPERIMENT_RESULTS_DIR}"
    )

    print(
        f"Модель сохранена в: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
