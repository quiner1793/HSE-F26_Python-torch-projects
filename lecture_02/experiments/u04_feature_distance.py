"""U04: reject images far from all known-breed feature prototypes."""

from lecture_02.common.classes import BREEDS_25
from lecture_02.common.config import MODELS_DIR, RESULTS_DIR
from lecture_02.common.open_set import FeatureDistanceConfig, run_feature_distance_experiment


CONFIG = FeatureDistanceConfig(
    classes=BREEDS_25,
    model_path=MODELS_DIR / "model_25_classes_validation.pth",
    source_experiment_path=RESULTS_DIR / "06_validation_early_stopping" / "experiment.json",
    results_dir=RESULTS_DIR / "u04_feature_distance",
)


def main():
    return run_feature_distance_experiment(CONFIG)


if __name__ == "__main__":
    main()
