"""Shared rejection rule for cached predictions and single-image inference."""

import math


def is_unknown(score, threshold=None):
    if threshold is None:
        return False
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Порог должен быть в диапазоне [0, 1]")
    return score < threshold
