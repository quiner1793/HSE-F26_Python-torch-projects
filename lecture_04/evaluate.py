from dataclasses import dataclass
from pathlib import Path

import torch
from pandas import DatetimeIndex
from torch import Tensor, nn

from lecture_04.data import ScalingParams


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    mae: float


def predict(model: nn.Module, x: Tensor) -> Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)


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
    points: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_dates = dates if points is None else dates[:points]
    plot_target = target if points is None else target[:points]
    plot_pred = pred if points is None else pred[:points]
    target_prices = inverse_scale(plot_target, scaling)
    pred_prices = inverse_scale(plot_pred, scaling)

    plt.plot(plot_dates, target_prices.cpu().numpy(), label="real")
    plt.plot(plot_dates, pred_prices.cpu().numpy(), label=label)
    plt.xlabel("date")
    plt.ylabel("price")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.legend()


def save_predictions_plot(
    dates: DatetimeIndex,
    target: Tensor,
    pred: Tensor,
    label: str,
    path: Path,
    scaling: ScalingParams,
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
        points=points,
    )
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
