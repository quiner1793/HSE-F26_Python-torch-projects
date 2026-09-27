import os

import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights

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
# EXPERIMENT 02.3
# Data Augmentation + Partial Fine-Tuning
#
# Цель:
# проверить гипотезу, что после partial fine-tuning
# главным ограничением становится generalization.
#
# Отличие от предыдущего эксперимента:
# - layer4 + fc по-прежнему обучаются
# - learning rates те же
# - epochs те же
# - добавляем augmentation ТОЛЬКО для train
# - test preprocessing не изменяем
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

BACKBONE_LR = 0.0001
FC_LR = 0.01


MODEL_PATH = (
    f"{MODELS_DIR}/model_25_classes_partial_finetune_aug.pth"
)

EXPERIMENT_RESULTS_DIR = (
    f"{RESULTS_DIR}/02_3_augmentation"
)


# ============================================================
# TRANSFORMS
# ============================================================

def create_transforms():

    weights = ResNet18_Weights.DEFAULT

    # ImageNet normalization used by pretrained ResNet18.
    #
    # Train получает случайные преобразования,
    # чтобы модель не запоминала конкретное положение,
    # crop, освещение и т.д.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.8, 1.0),
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Test оставляем стандартным.
    # Это тот же preprocessing, который использовался
    # в предыдущих экспериментах.
    test_transform = weights.transforms()

    return train_transform, test_transform


# ============================================================
# DATASETS
# ============================================================

def create_augmented_datasets(
    data_root,
    selected_classes,
    train_transform,
    test_transform,
):
    """
    Используем существующий create_datasets два раза,
    чтобы не менять common/dataset.py.

    Первый вызов нужен для train с augmentation.
    Второй — для test со стандартным preprocessing.
    """

    train_ds, _ = create_datasets(
        data_root,
        selected_classes,
        train_transform,
    )

    _, test_ds = create_datasets(
        data_root,
        selected_classes,
        test_transform,
    )

    return train_ds, test_ds


# ============================================================
# MODEL
# ============================================================

def configure_partial_finetuning(model):

    # Сначала замораживаем всю ResNet.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Последний residual block адаптируем
    # под fine-grained classification пород.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Классификатор также обучается.
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    train_loader,
    device,
    epochs,
):

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

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

            running_loss += (
                loss.item() * X.size(0)
            )

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        print(
            f"epoch={epoch + 1}/{epochs} "
            f"loss={epoch_loss:.4f} "
            f"train_accuracy={epoch_accuracy:.2%}"
        )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("device:", DEVICE)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, _ = create_model(
        len(SELECTED_CLASSES)
    )

    model = configure_partial_finetuning(model)

    print("\nОбучаемые части модели:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(" -", name)

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    train_transform, test_transform = (
        create_transforms()
    )

    print("\nTRAIN augmentation:")
    print(train_transform)

    print("\nTEST transform:")
    print(test_transform)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_ds, test_ds = (
        create_augmented_datasets(
            DATA_ROOT,
            SELECTED_CLASSES,
            train_transform,
            test_transform,
        )
    )

    train_loader, test_loader = (
        create_loaders(
            train_ds,
            test_ds,
            BATCH_SIZE,
        )
    )

    print(
        f"\nTrain images: {len(train_ds)}"
    )

    print(
        f"Test images: {len(test_ds)}"
    )

    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    os.makedirs(
        MODELS_DIR,
        exist_ok=True,
    )

    os.makedirs(
        EXPERIMENT_RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRAIN / LOAD
    # --------------------------------------------------------

    if os.path.exists(MODEL_PATH):

        print(
            f"\nМодель уже существует: "
            f"{MODEL_PATH}"
        )

        print(
            "Загружаем сохранённую модель..."
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        model = model.to(DEVICE)

    else:

        print(
            "\nЗапускаем обучение "
            "с augmentation..."
        )

        print(
            f"layer4 learning rate = "
            f"{BACKBONE_LR}"
        )

        print(
            f"fc learning rate = "
            f"{FC_LR}"
        )

        model = train_model(
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
            f"\nМодель сохранена: "
            f"{MODEL_PATH}"
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
    # COMPARISON
    # --------------------------------------------------------

    baseline_accuracy = 0.9065
    finetune_accuracy = 0.9113

    print("\n" + "=" * 60)

    print(
        "EXPERIMENT 02.3 — "
        "AUGMENTATION + PARTIAL FINE-TUNING"
    )

    print("=" * 60)

    print(
        f"02.1 Frozen backbone: "
        f"{baseline_accuracy:.2%}"
    )

    print(
        f"02.2 Partial fine-tuning: "
        f"{finetune_accuracy:.2%}"
    )

    print(
        f"02.3 Augmentation + fine-tuning: "
        f"{accuracy:.2%}"
    )

    print(
        "\nИзменение относительно 02.1: "
        f"{accuracy - baseline_accuracy:+.2%}"
    )

    print(
        "Изменение относительно 02.2: "
        f"{accuracy - finetune_accuracy:+.2%}"
    )

    print(
        f"\nРезультаты: "
        f"{EXPERIMENT_RESULTS_DIR}"
    )

    print(
        f"Модель: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
