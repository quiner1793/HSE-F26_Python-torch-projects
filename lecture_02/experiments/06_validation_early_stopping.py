"""Validation, scheduler и early stopping для лучшей конфигурации."""

from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.experiment import ExperimentConfig, run_experiment

CONFIG = ExperimentConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes_validation.pth",
    results_dir=RESULTS_DIR / "06_validation_early_stopping",
    epochs=20,
    backbone_lr=0.001,
    augmentation=True,
    validation_fraction=0.2,
    scheduler_patience=1,
    scheduler_factor=0.3,
    early_stopping_patience=5,
    min_delta=0.0001,
    diagnosis=True,
)


def main():
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
