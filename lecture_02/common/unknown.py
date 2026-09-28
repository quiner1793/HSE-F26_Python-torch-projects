"""Baseline unknown: frozen model, confidence measurements and reports."""

from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .classes import CAT_BREEDS
from .config import DATA_ROOT, DEVICE
from .dataset import SelectedBreedsDataset, create_datasets
from .model import create_model
from .train import load_model


@dataclass(frozen=True)
class UnknownConfig:
    classes: tuple[str, ...]
    model_path: Path
    source_experiment_path: Path
    results_dir: Path
    data_root: Path = DATA_ROOT
    batch_size: int = 32
    high_confidence: float = 0.9


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_unknown_datasets(data_root, classes, transform, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("Нужна validation-выборка исходного эксперимента")
    if set(classes) & set(CAT_BREEDS):
        raise ValueError("Известные классы пересекаются с unknown")
    _, known_validation, known_test = create_datasets(
        data_root,
        classes,
        transform,
        transform,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    # Reuse the official splits already loaded for the known breeds.
    unknown_validation = SelectedBreedsDataset(known_validation.dataset.base, transform, CAT_BREEDS)
    unknown_test = SelectedBreedsDataset(known_test.base, transform, CAT_BREEDS)
    return {
        "known_validation": known_validation,
        "unknown_validation": unknown_validation,
        "known_test": known_test,
        "unknown_test": unknown_test,
    }


def collect_predictions(model, loader, device):
    model.eval()
    records = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
                raise ValueError("Ожидались конечные logits минимум двух классов")
            probabilities, indices = logits.softmax(dim=1).topk(2, dim=1)
            for target, values, prediction in zip(
                targets.tolist(), probabilities.cpu().tolist(), indices[:, 0].cpu().tolist()
            ):
                records.append(
                    {
                        "target": target,
                        "prediction": prediction,
                        "confidence": values[0],
                        "margin": values[0] - values[1],
                    }
                )
    if not records:
        raise ValueError("Выборка для проверки unknown пуста")
    return records


def summarize_predictions(records, known, high_confidence):
    if not records:
        raise ValueError("Нет предсказаний для отчёта")
    total = len(records)
    high = [row for row in records if row["confidence"] >= high_confidence]
    result = {
        "total": total,
        "mean_confidence": sum(row["confidence"] for row in records) / total,
        "high_confidence_count": len(high),
        "high_confidence_rate": len(high) / total,
        "acceptance_rate": 1.0,
        "predicted_class_counts": dict(Counter(row["prediction"] for row in records)),
    }
    if known:
        correct = sum(row["prediction"] == row["target"] for row in records)
        result.update(
            {
                "correct": correct,
                "accuracy_all_known": correct / total,
                "accuracy_accepted_known": correct / total,
                "false_rejection_rate": 0.0,
                "high_confidence_errors": sum(row["prediction"] != row["target"] for row in high),
            }
        )
    else:
        # Cat labels are local indices, not dog labels, even when numbers coincide.
        result.update(
            {"unknown_detection_rate": 0.0, "unknown_false_acceptance_rate": 1.0, "high_confidence_errors": len(high)}
        )
    return result


def rejection_curve(known_records, unknown_records):
    known = sorted(row["confidence"] for row in known_records)
    unknown = sorted(row["confidence"] for row in unknown_records)
    if not known or not unknown:
        raise ValueError("Для кривой нужны обе выборки")
    thresholds = sorted({0.0, *known, *unknown, math.nextafter(1.0, math.inf)})
    return [
        {
            "threshold": threshold,
            "false_rejection_rate": bisect_left(known, threshold) / len(known),
            "unknown_detection_rate": bisect_left(unknown, threshold) / len(unknown),
        }
        for threshold in thresholds
    ]


def save_unknown_results(config, metadata, records, metrics, curve):
    import matplotlib.pyplot as plt

    directory = config.results_dir
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "method": "no_rejection",
        "threshold": None,
        "metadata": metadata,
        "metrics": metrics,
        "predictions": records,
        "validation_rejection_curve": curve,
    }
    (directory / "experiment.json").write_text(
        json.dumps(payload, default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, split in zip(axes, ("validation", "test")):
        for group, label, color in (
            ("known", "Известные: собаки", "#2475a8"),
            ("unknown", "Unknown: кошки", "#b84050"),
        ):
            scores = [row["confidence"] for row in records[f"{group}_{split}"]]
            ax.hist(
                scores,
                bins=[i / 20 for i in range(21)],
                weights=[100 / len(scores)] * len(scores),
                histtype="step",
                linewidth=2,
                label=label,
                color=color,
            )
        ax.set(title=split, xlabel="Максимальная softmax-оценка", xlim=(0, 1), ylabel="Доля изображений, %")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "score_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [100 * row["false_rejection_rate"] for row in curve],
        [100 * row["unknown_detection_rate"] for row in curve],
        color="#2475a8",
    )
    ax.axvline(5, linestyle="--", color="#b84050", label="Ориентир: 5% ложных отказов")
    ax.set(
        xlabel="Ложный отказ на известных, %",
        ylabel="Обнаружение unknown, %",
        title="Возможности порога на validation",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "rejection_tradeoff.png", dpi=160)
    plt.close(fig)

    report = [
        "U01: БАЗОВАЯ МОДЕЛЬ БЕЗ ОТКАЗА",
        "",
        "Модель всегда называет одну из 25 пород; unknown пока не выдаётся.",
        f"Уверенная ошибка: softmax-оценка >= {config.high_confidence:.0%} (не порог отказа).",
    ]
    for split in ("validation", "test"):
        known, unknown = metrics[f"known_{split}"], metrics[f"unknown_{split}"]
        report += [
            "",
            split.upper(),
            f"Известных изображений: {known['total']}; unknown: {unknown['total']}",
            f"Accuracy на известных: {known['accuracy_all_known']:.2%}",
            "Обнаружение unknown: 0.00%; ложный отказ на известных: 0.00%",
            f"Средняя уверенность: известные {known['mean_confidence']:.2%}, unknown {unknown['mean_confidence']:.2%}",
            f"Уверенных ошибок на известных: {known['high_confidence_errors']}",
            f"Уверенных ошибок на unknown: {unknown['high_confidence_errors']} ({unknown['high_confidence_rate']:.2%})",
            "Частые ответы на unknown:",
        ]
        for index, count in sorted(unknown["predicted_class_counts"].items(), key=lambda item: item[1], reverse=True)[
            :5
        ]:
            report.append(f"- {config.classes[int(index)]}: {count}")
    (directory / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def run_unknown_baseline(config):
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Нужен обученный checkpoint: {config.model_path}")
    source = json.loads(config.source_experiment_path.read_text(encoding="utf-8"))["config"]
    if source["classes"] != list(config.classes):
        raise ValueError("Классы исходного эксперимента не совпадают")
    print("device:", DEVICE)
    model, weights = create_model(len(config.classes), pretrained=False)
    model, _ = load_model(model, config.model_path, DEVICE, expected_classes=config.classes)
    datasets = create_unknown_datasets(
        config.data_root, config.classes, weights.transforms(), source["validation_fraction"], source["seed"]
    )
    records, metrics = {}, {}
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} изображений", flush=True)
        rows = collect_predictions(model, DataLoader(dataset, batch_size=config.batch_size, shuffle=False), DEVICE)
        selected = dataset.dataset if isinstance(dataset, Subset) else dataset
        positions = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
        for row, position in zip(rows, positions):
            row["source_index"] = selected.indices[position]
            row["true_class"] = selected.selected_classes[row["target"]]
        records[name] = rows
        metrics[name] = summarize_predictions(rows, name.startswith("known_"), config.high_confidence)
    metadata = {
        "checkpoint_sha256": file_digest(config.model_path),
        "torch_version": str(torch.__version__),
        "device": str(DEVICE),
        "seed": source["seed"],
        "validation_fraction": source["validation_fraction"],
        "unknown_classes": list(CAT_BREEDS),
        "source_splits": {name: "trainval" if name.endswith("validation") else "test" for name in records},
        "annotation_sha256": {
            split: file_digest(config.data_root / "oxford-iiit-pet" / "annotations" / f"{split}.txt")
            for split in ("trainval", "test")
        },
    }
    curve = rejection_curve(records["known_validation"], records["unknown_validation"])
    save_unknown_results(config, metadata, records, metrics, curve)
    print(f"Результаты: {config.results_dir}")
    return metrics
