from dataclasses import dataclass

import torch
from torch import Tensor, nn


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


def plot_predictions(target: Tensor, pred: Tensor, label: str, points: int = 200) -> None:
    import matplotlib.pyplot as plt

    plt.plot(target[:points].cpu().numpy(), label="real")
    plt.plot(pred[:points].cpu().numpy(), label=label)
    plt.legend()
    plt.show()
