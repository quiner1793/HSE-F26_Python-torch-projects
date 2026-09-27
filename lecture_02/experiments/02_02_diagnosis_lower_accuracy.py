import os
from collections import Counter

import torch
import matplotlib.pyplot as plt

from lecture_02.common.config import (
    DEVICE,
    DATA_ROOT,
    MODELS_DIR,
    RESULTS_DIR
)

from lecture_02.common.dataset import (
    create_datasets,
    create_loaders,
)

from lecture_02.common.model import create_model
from lecture_02.common.train import load_model


# ============================================================
# EXPERIMENT 02.1
# Диагностика причин снижения Accuracy
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

MODEL_PATH = f"{MODELS_DIR}/model_25_classes.pth"
RESULTS_DIR = f"{RESULTS_DIR}/02_1_diagnosis"
BATCH_SIZE = 32


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("device:", DEVICE)

    # ---------- DATA ----------
    model, weights = create_model(len(SELECTED_CLASSES))
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

    # ---------- MODEL ----------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}\n"
            "Сначала запусти experiments/02_25_classes.py"
        )

    model, classes = load_model(
        model,
        MODEL_PATH,
        DEVICE
    )

    model.eval()

    # ---------- PREDICTIONS ----------
    all_true = []
    all_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            pred = logits.argmax(dim=1)

            all_true.extend(y.tolist())
            all_pred.extend(pred.cpu().tolist())

    total = len(all_true)
    correct = sum(
        true == pred
        for true, pred in zip(all_true, all_pred)
    )
    accuracy = correct / total

    # ---------- CLASS ACCURACY ----------
    class_total = Counter(all_true)
    class_correct = Counter()

    for true, pred in zip(all_true, all_pred):
        if true == pred:
            class_correct[true] += 1

    class_accuracy = {}

    for i, name in enumerate(classes):
        total_i = class_total[i]
        correct_i = class_correct[i]
        class_accuracy[name] = correct_i / total_i if total_i else 0.0

    # ---------- CONFUSION MATRIX ----------
    n = len(classes)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for true, pred in zip(all_true, all_pred):
        matrix[true][pred] += 1

    # ---------- TOP CONFUSIONS ----------
    confusions = []

    for true in range(n):
        for pred in range(n):
            if true != pred and matrix[true][pred] > 0:
                confusions.append(
                    (
                        matrix[true][pred],
                        classes[true],
                        classes[pred],
                    )
                )

    confusions.sort(reverse=True)

    # ---------- GRAPH 1: CLASS ACCURACY ----------
    sorted_classes = sorted(
        class_accuracy.items(),
        key=lambda x: x[1]
    )

    names = [x[0] for x in sorted_classes]
    values = [x[1] * 100 for x in sorted_classes]

    plt.figure(figsize=(10, 8))
    plt.barh(names, values)
    plt.xlabel("Accuracy, %")
    plt.title("Accuracy по классам — 25 классов")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/class_accuracy.png",
        dpi=200
    )
    plt.close()

    # ---------- GRAPH 2: TOP CONFUSIONS ----------
    top = confusions[:15]

    labels = [
        f"{true} → {pred}"
        for _, true, pred in top
    ]
    counts = [count for count, _, _ in top]

    plt.figure(figsize=(11, 7))
    plt.barh(labels[::-1], counts[::-1])
    plt.xlabel("Количество ошибок")
    plt.title("Наиболее частые ошибки классификации")
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/top_confusions.png",
        dpi=200
    )
    plt.close()

    # ---------- REPORT ----------
    weakest = sorted(
        class_accuracy.items(),
        key=lambda x: x[1]
    )[:5]

    strongest = sorted(
        class_accuracy.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    with open(
        f"{RESULTS_DIR}/report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("ОТЧЁТ ОБ ЭКСПЕРИМЕНТЕ 02.1\n")
        f.write("Диагностика причин снижения Accuracy\n\n")

        f.write(f"Количество классов: {len(classes)}\n")
        f.write(f"Тестовых изображений: {total}\n")
        f.write(f"Правильных предсказаний: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")

        f.write("САМЫЕ СЛАБЫЕ КЛАССЫ:\n")
        for name, value in weakest:
            f.write(f"- {name}: {value:.2%}\n")

        f.write("\nСАМЫЕ ЧАСТЫЕ ОШИБКИ:\n")
        for count, true, pred in top[:10]:
            f.write(
                f"- {true} -> {pred}: {count} ошибок\n"
            )

        f.write("\nСАМЫЕ СИЛЬНЫЕ КЛАССЫ:\n")
        for name, value in strongest:
            f.write(f"- {name}: {value:.2%}\n")

        f.write(
            "\nИнтерпретация:\n"
            "Если ошибки концентрируются между несколькими похожими "
            "классами, это указывает на проблему различимости классов. "
            "Если слабые классы имеют существенно меньшую accuracy, "
            "нужно отдельно проверить количество и качество данных. "
            "Этот эксперимент не изменяет модель и служит только "
            "для диагностики причины ошибок.\n"
        )

    print("\nTEST:")
    print(f"Accuracy = {accuracy:.2%}")

    print("\nСамые слабые классы:")
    for name, value in weakest:
        print(f"- {name}: {value:.2%}")

    print("\nСамые частые ошибки:")
    for count, true, pred in top[:10]:
        print(f"- {true} -> {pred}: {count}")

    print(f"\nРезультаты: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
