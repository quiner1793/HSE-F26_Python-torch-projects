"""Общий сценарий: данные → обучение/загрузка → оценка → реальное фото."""

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import torch

from .config import DATA_ROOT, DEVICE
from .dataset import create_datasets, create_loaders
from .evaluate import evaluate
from .model import create_model
from .predict import predict_image
from .train import load_model, save_model, train_model
from .transforms import create_transforms


@dataclass(frozen=True)
class ExperimentConfig:
    classes: tuple[str, ...]
    model_path: Path
    results_dir: Path
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.01
    momentum: float = 0.9
    # None: обучаем только fc. Число: обучаем также layer4 с этим LR.
    backbone_lr: float | None = None
    augmentation: bool = False
    evaluate_only: bool = False
    diagnosis: bool = False
    reuse_checkpoint: bool = True
    image_path: Path | None = None
    data_root: Path = DATA_ROOT
    seed: int = 42


def run_experiment(config: ExperimentConfig):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    print("device:", DEVICE)
    use_checkpoint = config.reuse_checkpoint and config.model_path.exists()
    if config.evaluate_only and not use_checkpoint:
        raise FileNotFoundError(f"Сначала обучите модель: {config.model_path}")

    # При загрузке checkpoint не скачиваем ImageNet-веса.
    model, weights = create_model(len(config.classes), pretrained=not use_checkpoint)
    if use_checkpoint:
        model, _ = load_model(model, config.model_path, DEVICE, expected_classes=config.classes)

    train_transform, test_transform = create_transforms(weights, config.augmentation)
    train_ds, test_ds = create_datasets(config.data_root, config.classes, train_transform, test_transform)
    train_loader, test_loader = create_loaders(train_ds, test_ds, config.batch_size)
    if not use_checkpoint:
        history = []
        model = train_model(
            model,
            train_loader,
            DEVICE,
            config.epochs,
            config.learning_rate,
            config.momentum,
            config.backbone_lr,
            history=history,
        )
        save_model(model, config.classes, config.model_path)
        config.results_dir.mkdir(parents=True, exist_ok=True)
        (config.results_dir / "training.json").write_text(
            json.dumps({"config": asdict(config), "history": history}, default=str, indent=2),
            encoding="utf-8",
        )

    accuracy = evaluate(
        model,
        test_loader,
        DEVICE,
        classes=config.classes,
        results_dir=config.results_dir,
        diagnosis=config.diagnosis,
    )
    if config.image_path is not None:
        if config.image_path.exists():
            name, confidence = predict_image(model, config.image_path, test_transform, config.classes, DEVICE)
            print("REAL CASE:", name, f"{confidence:.1%}")
        else:
            print(f"Фото для REAL CASE не найдено: {config.image_path}")
    print(f"Результаты: {config.results_dir}")
    return accuracy
