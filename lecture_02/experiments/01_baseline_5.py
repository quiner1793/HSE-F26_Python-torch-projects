import os

from ..common.config import (
    DEVICE,
    SELECTED_CLASSES,
    DATA_ROOT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    MODELS_DIR,
    BASELINE_MODEL_PATH,
)

from ..common.dataset import create_datasets, create_loaders

from ..common.model import create_model

from ..common.train import train_model, save_model, load_model

from ..common.evaluate import evaluate

from ..common.predict import predict_image


def main():
    print("device:", DEVICE)

    model, weights = create_model(len(SELECTED_CLASSES))

    transform = weights.transforms()

    train_ds, test_ds = create_datasets(DATA_ROOT, SELECTED_CLASSES, transform)

    train_loader, test_loader = create_loaders(train_ds, test_ds, BATCH_SIZE)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(BASELINE_MODEL_PATH):

        print(f"Загружаем модель из " f"{BASELINE_MODEL_PATH}...")

        model, classes = load_model(model, BASELINE_MODEL_PATH, DEVICE)

    else:

        print("Модель не найдена. " "Запускаем обучение...")

        model = train_model(
            model, train_loader, DEVICE, EPOCHS, LEARNING_RATE, MOMENTUM
        )

        save_model(model, SELECTED_CLASSES, BASELINE_MODEL_PATH)

    print("\nTEST:")

    evaluate(model, test_loader, DEVICE)

    print("\nREAL CASE:")

    image_path = "data_real/praire_dog.jpg"

    class_name, confidence = predict_image(
        model, image_path, transform, SELECTED_CLASSES, DEVICE
    )

    print(class_name, f"{confidence:.1%}")


if __name__ == "__main__":
    main()
