import os
from datetime import datetime

import torch


def evaluate(
    model,
    test_loader,
    device,
    classes=None,
    results_dir=None
):
    model.eval()

    if classes is None:
        classes = [str(i) for i in range(model.fc.out_features)]

    num_classes = len(classes)

    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    confusion = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X).argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            for true_class, predicted_class in zip(
                y.cpu().tolist(),
                pred.cpu().tolist()
            ):
                class_total[true_class] += 1
                confusion[true_class][predicted_class] += 1

                if true_class == predicted_class:
                    class_correct[true_class] += 1

    accuracy = correct / total if total else 0.0

    print(f"accuracy = {accuracy:.2%}")

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

        class_accuracy = [
            class_correct[i] / class_total[i]
            if class_total[i] else 0.0
            for i in range(num_classes)
        ]

        # График accuracy по классам
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.bar(range(num_classes), class_accuracy)
            ax.set_title("Accuracy по классам")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.set_xticks(range(num_classes))
            ax.set_xticklabels(classes, rotation=75, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                os.path.join(results_dir, "accuracy_by_class.png"),
                dpi=180
            )
            plt.close(fig)

            # Confusion matrix
            fig, ax = plt.subplots(figsize=(12, 10))
            image = ax.imshow(confusion, interpolation="nearest")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(classes, rotation=90, fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(
                os.path.join(results_dir, "confusion_matrix.png"),
                dpi=180
            )
            plt.close(fig)

        except ImportError:
            print("matplotlib не установлен — графики не созданы.")

        # Короткий текстовый отчёт
        ranked = sorted(
            zip(classes, class_accuracy),
            key=lambda x: x[1]
        )

        weakest = ranked[:3]
        strongest = ranked[-3:][::-1]

        report = [
            "ОТЧЁТ ОБ ЭКСПЕРИМЕНТЕ",
            "",
            f"Дата: {datetime.now():%Y-%m-%d %H:%M}",
            f"Количество классов: {num_classes}",
            f"Тестовых изображений: {total}",
            f"Правильных предсказаний: {correct}",
            f"Accuracy: {accuracy:.2%}",
            "",
            "Самые слабые классы:"
        ]

        for name, value in weakest:
            report.append(f"- {name}: {value:.2%}")

        report += ["", "Самые сильные классы:"]

        for name, value in strongest:
            report.append(f"- {name}: {value:.2%}")

        report += [
            "",
            "Файлы:",
            "- accuracy_by_class.png",
            "- confusion_matrix.png",
            "- report.txt",
        ]

        with open(
            os.path.join(results_dir, "report.txt"),
            "w",
            encoding="utf-8"
        ) as f:
            f.write("\n".join(report))

    return accuracy
