"""Проверка аугментации поверх лучшего режима fine-tuning."""

from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.experiment import ExperimentConfig, run_experiment

CONFIG = ExperimentConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes_layer4_aug.pth",
    results_dir=RESULTS_DIR / "05_finetune_with_augmentation",
    backbone_lr=0.001,
    augmentation=True,
    diagnosis=True,
)


def main():
    return run_experiment(CONFIG)


if __name__ == "__main__":
    main()
