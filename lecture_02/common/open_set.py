"""Open-set recognition by distance to known-class feature prototypes."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

from .classes import CAT_BREEDS
from .config import DATA_ROOT, DEVICE
from .dataset import SelectedBreedsDataset, create_datasets
from .model import create_model
from .train import load_model
from .unknown import file_digest


@dataclass(frozen=True)
class FeatureDistanceConfig:
    classes: tuple[str, ...]
    model_path: Path
    source_experiment_path: Path
    results_dir: Path
    data_root: Path = DATA_ROOT
    batch_size: int = 32
    max_false_rejection: float = 0.05


def extract_features_and_logits(model, images):
    """Возвращает признаки перед fc и logits; общая основа U04 и фото-предсказаний."""
    captured = []

    def capture_features(module, inputs):
        """Запоминает вход fc через временный hook."""
        captured.append(inputs[0])

    if not hasattr(model, "fc"):
        raise ValueError("Модель должна иметь слой fc")
    handle = model.fc.register_forward_pre_hook(capture_features)
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(images)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Не удалось получить признаки перед fc")
    return F.normalize(captured[0], dim=1), logits


def _feature_batches(model, loader, device):
    """Подаёт батчи признаков и logits; внутренний помощник U04."""
    for images, targets in loader:
        features, logits = extract_features_and_logits(model, images.to(device))
        yield features, logits, targets.to(device)


def build_class_prototypes(model, loader, device, num_classes):
    """Строит центр каждой известной породы по train; первый этап U04."""
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    for features, _, targets in _feature_batches(model, loader, device):
        if sums is None:
            sums = torch.zeros(num_classes, features.shape[1], device=device)
        sums.index_add_(0, targets, features)
        counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
    if sums is None or (counts == 0).any():
        missing = (counts == 0).nonzero().flatten().tolist()
        raise ValueError(f"Нет обучающих примеров для классов: {missing}")
    return F.normalize(sums / counts.unsqueeze(1), dim=1), counts.cpu().tolist()


def nearest_prototype(features, prototypes):
    """Находит ближайший центр и cosine similarity для каждого изображения."""
    similarities, indices = (features @ prototypes.T).max(dim=1)
    return similarities.clamp(-1.0, 1.0), indices


def collect_distance_predictions(model, loader, device, prototypes):
    """Сохраняет distance и ответ модели для validation/test U04."""
    records = []
    for features, logits, targets in _feature_batches(model, loader, device):
        probabilities, predictions = logits.softmax(dim=1).max(dim=1)
        similarities, nearest = nearest_prototype(features, prototypes)
        for target, prediction, confidence, similarity, nearest_class in zip(
            targets.cpu().tolist(),
            predictions.cpu().tolist(),
            probabilities.cpu().tolist(),
            similarities.cpu().tolist(),
            nearest.cpu().tolist(),
        ):
            records.append(
                {
                    "target": target,
                    "prediction": prediction,
                    "confidence": confidence,
                    "nearest_prototype": nearest_class,
                    "distance": 1.0 - similarity,
                }
            )
    if not records:
        raise ValueError("Выборка для оценки пуста")
    return records


def select_distance_threshold(known_validation, max_false_rejection):
    """Выбирает границу distance только по известной validation U04."""
    if not 0 <= max_false_rejection < 1:
        raise ValueError("Лимит ложного отказа должен быть в диапазоне [0, 1)")
    distances = sorted(row["distance"] for row in known_validation)
    if not distances or any(not math.isfinite(value) or not 0 <= value <= 2 for value in distances):
        raise ValueError("Некорректные расстояния validation")
    allowed_rejections = math.floor(len(distances) * max_false_rejection)
    index = len(distances) - allowed_rejections - 1
    return distances[index]


def is_unknown_distance(distance, threshold):
    """Проверяет правило U04: distance выше порога означает unknown."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 2:
        raise ValueError("Порог расстояния должен быть в диапазоне [0, 2]")
    if not math.isfinite(distance) or not 0 <= distance <= 2:
        raise ValueError("Расстояние должно быть в диапазоне [0, 2]")
    return distance > threshold


def distance_metrics(records, known, threshold):
    """Считает метрики U04 для известных или неизвестных изображений."""
    accepted = [row for row in records if not is_unknown_distance(row["distance"], threshold)]
    rejected = len(records) - len(accepted)
    result = {
        "total": len(records),
        "accepted": len(accepted),
        "rejected": rejected,
        "acceptance_rate": len(accepted) / len(records),
        "mean_distance": sum(row["distance"] for row in records) / len(records),
        "predicted_class_counts": dict(Counter(row["prediction"] for row in records)),
    }
    if known:
        correct = sum(row["target"] == row["prediction"] for row in accepted)
        result.update(
            {
                "correct": correct,
                "false_rejection_rate": rejected / len(records),
                "accuracy_all_known": correct / len(records),
                "accuracy_accepted_known": correct / len(accepted) if accepted else None,
            }
        )
        per_class = {}
        for row in records:
            name = row.get("true_class", str(row["target"]))
            values = per_class.setdefault(name, {"total": 0, "rejected": 0})
            values["total"] += 1
            values["rejected"] += int(is_unknown_distance(row["distance"], threshold))
        for values in per_class.values():
            values["false_rejection_rate"] = values["rejected"] / values["total"]
        result["per_class"] = per_class
    else:
        result.update(
            {
                "unknown_detection_rate": rejected / len(records),
                "unknown_false_acceptance_rate": len(accepted) / len(records),
            }
        )
    return result


def _add_source_metadata(dataset, rows):
    """Добавляет исходный индекс и имя класса к строкам отчёта U04."""
    selected = dataset.dataset if isinstance(dataset, Subset) else dataset
    positions = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))
    for row, position in zip(rows, positions):
        row["source_index"] = selected.indices[position]
        row["true_class"] = selected.selected_classes[row["target"]]


def _save_feature_distance_results(config, metadata, records, metrics, threshold, prototypes):
    """Сохраняет policy U04, метрики, отчёт и визуализации."""
    import matplotlib.pyplot as plt

    directory = config.results_dir
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "method": "cosine_distance_to_class_prototype",
        "threshold": threshold,
        "selection": {
            "split": "known_validation",
            "unknown_used": False,
            "max_false_rejection": config.max_false_rejection,
            "rule": "distance > threshold -> unknown",
        },
        "metadata": metadata,
        "prototypes": prototypes.cpu().tolist(),
        "metrics": metrics,
        "predictions": records,
    }
    (directory / "experiment.json").write_text(
        json.dumps(payload, default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    maximum = max(threshold, *(row["distance"] for rows in records.values() for row in rows))
    bins = [maximum * index / 20 for index in range(21)]
    if maximum == 0:
        bins = [index / 20 for index in range(21)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    validation = [row["distance"] for row in records["known_validation"]]
    axes[0].hist(
        validation,
        bins=bins,
        weights=[100 / len(validation)] * len(validation),
        histtype="step",
        linewidth=2,
        color="#2475a8",
        label="Известные: собаки",
    )
    for key, label, color in (
        ("known_test", "Известные: собаки", "#2475a8"),
        ("unknown_test", "Unknown: кошки", "#b84050"),
    ):
        values = [row["distance"] for row in records[key]]
        axes[1].hist(
            values,
            bins=bins,
            weights=[100 / len(values)] * len(values),
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )
    for ax, title in zip(axes, ("validation: только известные", "test")):
        ax.axvline(threshold, color="black", linestyle="--", label=f"Порог {threshold:.4f}")
        ax.set(title=title, xlabel="Косинусное расстояние до ближайшего центра", ylabel="Доля изображений, %")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "score_distribution.png", dpi=160)
    plt.close(fig)

    known = sorted(row["distance"] for row in records["known_test"])
    unknown = sorted(row["distance"] for row in records["unknown_test"])
    candidates = sorted({0.0, *known, *unknown, math.nextafter(max(known + unknown), math.inf)})
    false_rejections = [
        100 * sum(value > threshold_value for value in known) / len(known) for threshold_value in candidates
    ]
    detections = [
        100 * sum(value > threshold_value for value in unknown) / len(unknown) for threshold_value in candidates
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(false_rejections, detections, color="#2475a8")
    ax.scatter(
        100 * metrics["known_test"]["false_rejection_rate"],
        100 * metrics["unknown_test"]["unknown_detection_rate"],
        color="#b84050",
        label="Порог, выбранный по известным validation",
        zorder=3,
    )
    ax.set(
        xlabel="Ложный отказ на известных, %",
        ylabel="Обнаружение кошек, %",
        title="Диагностика U04 на test",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "rejection_tradeoff.png", dpi=160)
    plt.close(fig)

    known_validation = metrics["known_validation"]
    known_test = metrics["known_test"]
    unknown_test = metrics["unknown_test"]
    report = [
        "U04: РАССТОЯНИЕ ДО ЦЕНТРОВ ИЗВЕСТНЫХ ПОРОД",
        "",
        "Центры построены только по train известных пород.",
        f"Порог {threshold:.10f} выбран только по известным validation; unknown не использовались.",
        f"Ложный отказ на validation: {known_validation['rejected']}/{known_validation['total']} "
        f"({known_validation['false_rejection_rate']:.2%})",
        "",
        "TEST",
        f"Unknown обнаружено: {unknown_test['rejected']}/{unknown_test['total']} "
        f"({unknown_test['unknown_detection_rate']:.2%})",
        f"Ложный отказ на известных: {known_test['rejected']}/{known_test['total']} "
        f"({known_test['false_rejection_rate']:.2%})",
        f"Accuracy на всех известных (отказ = ошибка): {known_test['accuracy_all_known']:.2%}",
        f"Доля принятых известных: {known_test['acceptance_rate']:.2%}",
        f"Accuracy среди принятых известных: {known_test['accuracy_accepted_known']:.2%}",
        "",
        "Unknown в этой проверке: кошки. Другие группы нужны для отдельной итоговой оценки.",
    ]
    (directory / "report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def run_feature_distance_experiment(config):
    """Запускает полный U04: prototypes -> threshold -> оценка -> policy."""
    if not config.model_path.is_file():
        raise FileNotFoundError(f"Нужен обученный checkpoint: {config.model_path}")
    source = json.loads(config.source_experiment_path.read_text(encoding="utf-8"))
    source_config = source["config"]
    if source_config["classes"] != list(config.classes):
        raise ValueError("Классы исходного эксперимента не совпадают")

    print("device:", DEVICE)
    model, weights = create_model(len(config.classes), pretrained=False)
    model, _ = load_model(model, config.model_path, DEVICE, expected_classes=config.classes)
    transform = weights.transforms()
    train, validation, test = create_datasets(
        config.data_root,
        config.classes,
        transform,
        transform,
        validation_fraction=source_config["validation_fraction"],
        seed=source_config["seed"],
    )
    unknown_test = SelectedBreedsDataset(test.base, transform, CAT_BREEDS)
    loaders = {
        "known_train": DataLoader(train, batch_size=config.batch_size, shuffle=False),
        "known_validation": DataLoader(validation, batch_size=config.batch_size, shuffle=False),
        "known_test": DataLoader(test, batch_size=config.batch_size, shuffle=False),
        "unknown_test": DataLoader(unknown_test, batch_size=config.batch_size, shuffle=False),
    }
    prototypes, prototype_counts = build_class_prototypes(model, loaders["known_train"], DEVICE, len(config.classes))
    datasets = {"known_validation": validation, "known_test": test, "unknown_test": unknown_test}
    records = {}
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} изображений", flush=True)
        records[name] = collect_distance_predictions(model, loaders[name], DEVICE, prototypes)
        _add_source_metadata(dataset, records[name])
    threshold = select_distance_threshold(records["known_validation"], config.max_false_rejection)
    metrics = {name: distance_metrics(rows, name.startswith("known_"), threshold) for name, rows in records.items()}
    metadata = {
        "checkpoint_sha256": file_digest(config.model_path),
        "source_experiment_sha256": file_digest(config.source_experiment_path),
        "torch_version": str(torch.__version__),
        "device": str(DEVICE),
        "classes": list(config.classes),
        "unknown_classes": list(CAT_BREEDS),
        "prototype_counts": dict(zip(config.classes, prototype_counts)),
        "feature_normalization": "l2",
        "distance": "1 - cosine_similarity",
        "seed": source_config["seed"],
        "validation_fraction": source_config["validation_fraction"],
    }
    _save_feature_distance_results(config, metadata, records, metrics, threshold, prototypes)
    print(f"Результаты: {config.results_dir}")
    return metrics


def load_feature_distance_policy(experiment_path, checkpoint_path, classes, device):
    """Загружает threshold и prototypes U04 с проверкой checkpoint."""
    experiment = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
    if experiment.get("method") != "cosine_distance_to_class_prototype":
        raise ValueError("Нужен experiment.json опыта U04")
    if "prototypes" not in experiment:
        raise ValueError("В experiment.json нет центров пород; перезапустите U04")
    if experiment["config"]["classes"] != list(classes):
        raise ValueError("Классы модели не совпадают с экспериментом U04")
    if file_digest(checkpoint_path) != experiment["metadata"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint не совпадает с экспериментом U04")
    prototypes = torch.tensor(experiment["prototypes"], dtype=torch.float32, device=device)
    if prototypes.ndim != 2 or prototypes.shape[0] != len(classes) or not torch.isfinite(prototypes).all():
        raise ValueError("Некорректные центры пород в experiment.json")
    return prototypes, experiment["threshold"]


def predict_unknown_image(image_path, experiment_path):
    """Применяет сохранённое правило U04 к одной фотографии."""
    experiment = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
    classes = experiment["config"]["classes"]
    checkpoint_path = Path(experiment["config"]["model_path"])
    prototypes, threshold = load_feature_distance_policy(experiment_path, checkpoint_path, classes, DEVICE)

    model, weights = create_model(len(classes), pretrained=False)
    model, _ = load_model(model, checkpoint_path, DEVICE, expected_classes=classes)
    with Image.open(image_path) as image:
        tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    features, logits = extract_features_and_logits(model, tensor)
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("Размерность центров пород не совпадает с моделью")
    confidence, class_id = logits.softmax(dim=1).max(dim=1)
    similarities, _ = nearest_prototype(features, prototypes)
    distance = 1.0 - similarities[0].item()
    return {
        "label": "unknown" if is_unknown_distance(distance, threshold) else classes[class_id.item()],
        "confidence": confidence.item(),
        "distance": distance,
    }
