"""Shared settings for lecture 04 experiments."""

from dataclasses import dataclass

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2018-01-01"
    end: str = "2026-01-01"
    window: int = 20
    train_share: float = 0.8


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    seed: int = 42
