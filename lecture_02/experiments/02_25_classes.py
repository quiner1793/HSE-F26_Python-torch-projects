import os

from lecture_02.common.config import (
    DEVICE,
    DATA_ROOT,
    MODELS_DIR
)

from lecture_02.common.dataset import (
    create_datasets,
    create_loaders
)

from lecture_02.common.model import create_model

from lecture_02.common.train import (
    train_model,
    save_model,
    load_model
)

from lecture_02.common.evaluate import evaluate


# 25 классов
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
    "Pomeranian"
]


MODEL_PATH = (
    f"{MODELS_DIR}/model_25_classes.pth"
)

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
MOMENTUM = 0.9


def main():

    print("device:", DEVICE)

    print("\nКлассы:")
    for i, name in enumerate(SELECTED_CLASSES):
        print(f"{i}: {name}")

    model, weights = create_model(
        len(SELECTED_CLASSES)
    )

    transform = weights.transforms()

    train_ds, test_ds = create_datasets(
        DATA_ROOT,
        SELECTED_CLASSES,
        transform
    )

    train_loader, test_loader = create_loaders(
        train_ds,
        test_ds,
        BATCH_SIZE
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    if os.path.exists(MODEL_PATH):

        print(
            f"\nЗагружаем модель из "
            f"{MODEL_PATH}..."
        )

        model, classes = load_model(
            model,
            MODEL_PATH,
            DEVICE
        )

    else:

        print("\nМодель не найдена.")
        print("Запускаем обучение...")

        model = train_model(
            model,
            train_loader,
            DEVICE,
            EPOCHS,
            LEARNING_RATE,
            MOMENTUM
        )

        save_model(
            model,
            SELECTED_CLASSES,
            MODEL_PATH
        )

    print("\nTEST:")

    accuracy = evaluate(
        model,
        test_loader,
        DEVICE
    )

    print(
        f"\nИтоговая accuracy: "
        f"{accuracy:.2%}"
    )


if __name__ == "__main__":
    main()