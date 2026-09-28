"""Проверка более сильной адаптации layer4 для похожих пород."""

from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.experiment import ExperimentConfig, run_experiment

CONFIG = ExperimentConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes_layer4_lr1e3.pth",
    results_dir=RESULTS_DIR / "04_finetune_layer4_lr1e3",
    backbone_lr=0.001,
    diagnosis=True,
)


def main():
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
