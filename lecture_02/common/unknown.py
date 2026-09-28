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
from .decision import is_unknown
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


def rejection_curve(known_records, unknown_records, score="confidence"):
    if score not in ("confidence", "margin"):
        raise ValueError("Оценка должна быть confidence или margin")
    known = sorted(row[score] for row in known_records)
    unknown = sorted(row[score] for row in unknown_records)
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


def save_unknown_results(config, metadata, records, metrics, curve, threshold=None, selection=None):
    import matplotlib.pyplot as plt

    score = getattr(config, "score", "confidence")
    score_label = "Максимальная softmax-оценка" if score == "confidence" else "Разница p1 - p2"
    directory = config.results_dir
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "method": "no_rejection" if threshold is None else f"{score}_threshold",
        "score": score,
        "threshold": threshold,
        "selection": selection,
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
            scores = [row[score] for row in records[f"{group}_{split}"]]
            ax.hist(
                scores,
                bins=[i / 20 for i in range(21)],
                weights=[100 / len(scores)] * len(scores),
                histtype="step",
                linewidth=2,
                label=label,
                color=color,
            )
        ax.set(title=split, xlabel=score_label, xlim=(0, 1), ylabel="Доля изображений, %")
        if threshold is not None:
            ax.axvline(threshold, color="black", linestyle="--", label=f"Порог {threshold:.4f}")
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
    limit = 5 if selection is None else 100 * selection["max_false_rejection"]
    ax.axvline(limit, linestyle="--", color="#b84050", label=f"Лимит ложных отказов: {limit:g}%")
    if threshold is not None:
        for split, marker in (("validation", "o"), ("test", "x")):
            ax.scatter(100 * metrics[f"known_{split}"]["false_rejection_rate"],
                       100 * metrics[f"unknown_{split}"]["unknown_detection_rate"],
                       marker=marker, label=f"Выбранный порог: {split}", zorder=3)
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

    if threshold is not None:
        title = "U02: ПОРОГ УВЕРЕННОСТИ" if score == "confidence" else "U03: РАЗНИЦА ДВУХ ЛУЧШИХ ОТВЕТОВ"
        report = [title, "",
                  f"Правило: {score} < {threshold:.10f} -> unknown",
                  f"Порог выбран только на validation; лимит ложного отказа: {limit:g}%."]
        for split in ("validation", "test"):
            known, unknown = metrics[f"known_{split}"], metrics[f"unknown_{split}"]
            accepted_accuracy = known["accuracy_accepted_known"]
            report += ["", split.upper(),
                       f"Unknown обнаружено: {unknown['rejected']}/{unknown['total']} ({unknown['unknown_detection_rate']:.2%})",
                       f"Ложный отказ на известных: {known['rejected']}/{known['total']} ({known['false_rejection_rate']:.2%})",
                       f"Accuracy на всех известных (отказ = ошибка): {known['accuracy_all_known']:.2%}",
                       f"Доля принятых известных: {known['acceptance_rate']:.2%}",
                       (f"Accuracy среди принятых известных: {accepted_accuracy:.2%}"
                        if accepted_accuracy is not None else "Принятых известных нет"),
                       "Ложные отказы по породам:"]
            for name, values in sorted(known["per_class"].items(), key=lambda item: item[1]["false_rejection_rate"], reverse=True)[:5]:
                report.append(f"- {name}: {values['rejected']}/{values['total']} ({values['false_rejection_rate']:.2%})")
        report += ["", "Unknown в этом опыте: кошки. Результат не описывает все неизвестные объекты."]
        (directory / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report))
        return

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


@dataclass(frozen=True)
class ThresholdConfig:
    source_path: Path
    results_dir: Path
    max_false_rejection: float = 0.05
    score: str = "confidence"


def select_threshold(known_validation, unknown_validation, max_false_rejection, score="confidence"):
    if not 0 <= max_false_rejection < 1:
        raise ValueError("Лимит ложного отказа должен быть в диапазоне [0, 1)")
    curve = rejection_curve(known_validation, unknown_validation, score)
    feasible = [row for row in curve if row["false_rejection_rate"] <= max_false_rejection
                and row["threshold"] <= 1]
    # Ties: prefer fewer known rejections, then the smallest threshold.
    selected = max(feasible, key=lambda row: (row["unknown_detection_rate"],
                   -row["false_rejection_rate"], -row["threshold"]))
    return selected["threshold"], curve


def threshold_metrics(records, known, threshold, high_confidence=0.9, score="confidence"):
    result = summarize_predictions(records, known, high_confidence)
    accepted = [row for row in records if not is_unknown(row[score], threshold)]
    rejected = len(records) - len(accepted)
    result.update({"rejected": rejected, "accepted": len(accepted),
                   "acceptance_rate": len(accepted) / len(records)})
    if known:
        correct = sum(row["target"] == row["prediction"] for row in accepted)
        result.update({"correct": correct, "accuracy_all_known": correct / len(records),
                       "accuracy_accepted_known": correct / len(accepted) if accepted else None,
                       "false_rejection_rate": rejected / len(records)})
        per_class = {}
        for row in records:
            name = row.get("true_class", str(row["target"]))
            counts = per_class.setdefault(name, {"total": 0, "rejected": 0})
            counts["total"] += 1
            counts["rejected"] += int(is_unknown(row[score], threshold))
        for counts in per_class.values():
            counts["false_rejection_rate"] = counts["rejected"] / counts["total"]
        result["per_class"] = per_class
    else:
        result.update({"unknown_detection_rate": rejected / len(records),
                       "unknown_false_acceptance_rate": len(accepted) / len(records)})
    result["high_confidence_errors"] = sum(
        row["confidence"] >= high_confidence and (not known or row["target"] != row["prediction"])
        for row in accepted
    )
    return result


def run_threshold_experiment(config):
    if config.score not in ("confidence", "margin"):
        raise ValueError("Оценка должна быть confidence или margin")
    source = json.loads(config.source_path.read_text(encoding="utf-8"))
    if source["method"] != "no_rejection":
        raise ValueError("Нужны исходные предсказания U01 без отказа")
    records = source["predictions"]
    for rows in records.values():
        if any(not math.isfinite(row[config.score]) or not 0 <= row[config.score] <= 1 for row in rows):
            raise ValueError(f"Некорректная {config.score} в исходных предсказаниях")
    threshold, curve = select_threshold(records["known_validation"], records["unknown_validation"],
                                        config.max_false_rejection, config.score)
    metrics = {name: threshold_metrics(rows, name.startswith("known_"), threshold,
                                       source["config"]["high_confidence"], config.score)
               for name, rows in records.items()}
    metadata = dict(source["metadata"], source_sha256=file_digest(config.source_path),
                    classes=source["config"]["classes"])
    save_unknown_results(config, metadata, records, metrics, curve, threshold=threshold,
                         selection={"split": "validation", "max_false_rejection": config.max_false_rejection,
                                    "tie_break": "fewer_known_rejections_then_smallest_threshold"})
    print(f"Результаты: {config.results_dir}")
    return metrics


def run_confidence_threshold(config):
    return run_threshold_experiment(config)
