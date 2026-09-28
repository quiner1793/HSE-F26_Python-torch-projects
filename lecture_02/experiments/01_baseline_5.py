"""Базовое задание: 5 пород."""
from lecture_02.common.classes import BREEDS_5
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR, REAL_DATA_DIR
from lecture_02.common.experiment import ExperimentConfig, run_experiment

CONFIG = ExperimentConfig(
    classes=BREEDS_5,
    model_path=MODELS_DIR / "model_5_classes.pth",
    results_dir=RESULTS_DIR / "5_classes",
    image_path=REAL_DATA_DIR / "praire_dog.jpg",
)


def main():
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
