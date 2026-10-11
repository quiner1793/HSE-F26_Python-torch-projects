"""Shared settings for lecture 04 experiments."""

from dataclasses import dataclass
from pathlib import Path

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LECTURE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = LECTURE_DIR / "results"


@dataclass(frozen=True)
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2018-01-01"
    end: str = "2027-01-01"
    train_end: str = "2026-01-01"
    test_start: str = "2026-01-01"
    window: int = 20
    forecast_steps_after_test: int = 100


@dataclass(frozen=True)
class TrainConfig:
    # mlp, rnn, lstm, gru
    model_name: str = "gru"
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 42
