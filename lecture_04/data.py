import random

import numpy as np
import torch
import yfinance as yf
from pandas import Series
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from lecture_04.config import DataConfig, TrainConfig


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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values = prices.to_numpy().reshape(-1)
    split_index = int(len(values) * config.train_share)
    train_raw = values[:split_index]

    mean = train_raw.mean()
    std = train_raw.std()
    scaled = (values - mean) / std

    series = torch.tensor(scaled, dtype=torch.float32)
    x_all, y_all = make_windows(series, config.window)

    split_windows = int(len(x_all) * config.train_share)
    x_train, y_train = x_all[:split_windows], y_all[:split_windows]
    x_test, y_test = x_all[split_windows:], y_all[split_windows:]

    return x_train, y_train, x_test, y_test


def make_train_loader(
    x_train: Tensor,
    y_train: Tensor,
    config: TrainConfig,
) -> DataLoader:
    dataset = TensorDataset(x_train, y_train)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
