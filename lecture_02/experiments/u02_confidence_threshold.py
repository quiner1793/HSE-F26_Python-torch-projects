"""U02: select a confidence threshold on cached validation predictions."""

from lecture_02.common.config import RESULTS_DIR
from lecture_02.common.unknown import ThresholdConfig, run_confidence_threshold

CONFIG = ThresholdConfig(
    source_path=RESULTS_DIR / "u01_unknown_baseline" / "experiment.json",
    results_dir=RESULTS_DIR / "u02_confidence_threshold",
)


def main():
    return run_confidence_threshold(CONFIG)


if __name__ == "__main__":
    main()
