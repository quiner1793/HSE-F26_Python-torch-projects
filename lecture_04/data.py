from dataclasses import dataclass
import random

import numpy as np
import torch
import yfinance as yf
from pandas import DatetimeIndex, Series, Timestamp
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from lecture_04.config import DataConfig, TrainConfig


@dataclass(frozen=True)
class ScalingParams:
    mean: float
    std: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_close_prices(config: DataConfig) -> Series:
    df = yf.download(
        config.ticker,
        start=config.start,
        end=config.end,
        auto_adjust=True,
        progress=False,
    )
    return df["Close"].dropna()


def make_windows(series: Tensor, window: int) -> tuple[Tensor, Tensor]:
    x, y = [], []
    for i in range(len(series) - window):
        x.append(series[i : i + window])
        y.append(series[i + window])
    return torch.stack(x), torch.stack(y)


def prepare_datasets(
    prices: Series,
    config: DataConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, DatetimeIndex, ScalingParams]:
    values = prices.to_numpy().reshape(-1)
    train_end = Timestamp(config.train_end)
    test_start = Timestamp(config.test_start)

    train_values_mask = prices.index < train_end
    train_raw = prices.loc[train_values_mask].to_numpy().reshape(-1)

    mean = train_raw.mean()
    std = train_raw.std()
    scaling = ScalingParams(mean=float(mean), std=float(std))
    scaled = (values - mean) / std

    series = torch.tensor(scaled, dtype=torch.float32)
    x_all, y_all = make_windows(series, config.window)
    target_dates = prices.index[config.window :]

    train_mask = target_dates < train_end
    test_mask = target_dates >= test_start

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_test, y_test = x_all[test_mask], y_all[test_mask]
    test_dates = target_dates[test_mask]

    return x_train, y_train, x_test, y_test, test_dates, scaling


def make_train_loader(
    x_train: Tensor,
    y_train: Tensor,
    config: TrainConfig,
) -> DataLoader:
    dataset = TensorDataset(x_train, y_train)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
