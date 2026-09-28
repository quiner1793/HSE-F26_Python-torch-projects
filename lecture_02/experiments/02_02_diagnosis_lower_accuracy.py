"""Диагностика сохранённой модели на 25 классов без обучения."""
from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.experiment import ExperimentConfig, run_experiment

CONFIG = ExperimentConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes.pth",
    results_dir=RESULTS_DIR / "02_1_diagnosis",
    evaluate_only=True,
    diagnosis=True,
)


def main():
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
