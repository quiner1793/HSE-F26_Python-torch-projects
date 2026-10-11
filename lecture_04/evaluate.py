from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd
import torch
from pandas import DatetimeIndex
from torch import Tensor, nn

from lecture_04.data import ScalingParams


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    mae: float


def predict(model: nn.Module, x: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)


def forecast_after_test(model: nn.Module, x_last: Tensor, steps: int) -> Tensor:
    if steps <= 0:
        return torch.empty(0, device=x_last.device)

    model.eval()
    window = x_last.clone()
    preds = []

    with torch.no_grad():
        for _ in range(steps):
            pred = model(window.unsqueeze(0)).squeeze(0)
            preds.append(pred)
            window = torch.cat((window[1:], pred.reshape(1)))

    return torch.stack(preds)


def evaluate_predictions(pred: Tensor, target: Tensor) -> RegressionMetrics:
    return RegressionMetrics(
        mse=nn.functional.mse_loss(pred, target).item(),
        mae=nn.functional.l1_loss(pred, target).item(),
    )


def evaluate_baseline(x: Tensor, target: Tensor) -> RegressionMetrics:
    baseline = x[:, -1]
    return evaluate_predictions(baseline, target)


def print_metrics(model_name: str, metrics: RegressionMetrics) -> None:
    print(f"{model_name} MSE: {metrics.mse:.6f}")
    print(f"{model_name} MAE: {metrics.mae:.6f}")


def inverse_scale(values: Tensor, scaling: ScalingParams) -> Tensor:
    return values * scaling.std + scaling.mean


def plot_predictions(
    dates: DatetimeIndex,
    target: Tensor,
    pred: Tensor,
    label: str,
    scaling: ScalingParams,
    future_dates: DatetimeIndex | None = None,
    future_pred: Tensor | None = None,
    points: int | None = None,
) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8), dpi=140)

    plot_dates = dates if points is None else dates[:points]
    plot_target = target if points is None else target[:points]
    plot_pred = pred if points is None else pred[:points]
    target_prices = inverse_scale(plot_target, scaling)
    pred_prices = inverse_scale(plot_pred, scaling)

    ax.plot(
        plot_dates,
        target_prices.cpu().numpy(),
        label="real",
        linewidth=2.2,
    )
    ax.plot(
        plot_dates,
        pred_prices.cpu().numpy(),
        label=label,
        linewidth=2.2,
    )

    if future_dates is not None and future_pred is not None and len(future_pred) > 0:
        bridge_dates = dates[-1:].append(future_dates)
        bridge_values = torch.cat((pred[-1:].detach(), future_pred))
        bridge_prices = inverse_scale(bridge_values, scaling)
        ax.plot(
            bridge_dates,
            bridge_prices.cpu().numpy(),
            linestyle="--",
            linewidth=2.0,
            label=f"{label} forecast",
        )
        ax.axvline(dates[-1], color="gray", linestyle=":", linewidth=1.8, alpha=0.8)

    ax.set_title("Price prediction", fontsize=18, pad=14)
    ax.set_xlabel("date", fontsize=13)
    ax.set_ylabel("price", fontsize=13)
    ax.grid(True, alpha=0.35, linewidth=0.9)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)


def save_predictions_plot(
    dates: DatetimeIndex,
    target: Tensor,
    pred: Tensor,
    label: str,
    path: Path,
    scaling: ScalingParams,
    future_dates: DatetimeIndex | None = None,
    future_pred: Tensor | None = None,
    points: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(
        dates=dates,
        target=target,
        pred=pred,
        label=label,
        scaling=scaling,
        future_dates=future_dates,
        future_pred=future_pred,
        points=points,
    )
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"Saved plot: {path}")


def make_future_business_dates(last_date: pd.Timestamp, steps: int) -> DatetimeIndex:
    if steps <= 0:
        return DatetimeIndex([])
    return pd.bdate_range(start=last_date, periods=steps + 1)[1:]
