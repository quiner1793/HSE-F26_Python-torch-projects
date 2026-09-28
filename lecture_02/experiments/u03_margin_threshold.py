"""U03: rejection by the gap between the two highest probabilities."""

from lecture_02.common.config import RESULTS_DIR
from lecture_02.common.unknown import ThresholdConfig, run_threshold_experiment

CONFIG = ThresholdConfig(
    source_path=RESULTS_DIR / "u01_unknown_baseline" / "experiment.json",
    results_dir=RESULTS_DIR / "u03_margin_threshold",
    score="margin",
)


def main():
    return run_threshold_experiment(CONFIG)


if __name__ == "__main__":
    main()
