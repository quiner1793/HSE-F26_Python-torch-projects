"""U01: confidence of the existing model on known dogs and unknown cats."""

from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.unknown import UnknownConfig, run_unknown_baseline

CONFIG = UnknownConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes_validation.pth",
    source_experiment_path=RESULTS_DIR / "06_validation_early_stopping" / "experiment.json",
    results_dir=RESULTS_DIR / "u01_unknown_baseline",
)


def main():
    return run_unknown_baseline(CONFIG)


if __name__ == "__main__":
    main()
