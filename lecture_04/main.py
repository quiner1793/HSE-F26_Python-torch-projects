from torch import nn

from lecture_04.config import DataConfig, DEVICE, RESULTS_DIR, TrainConfig
from lecture_04.data import download_close_prices, make_train_loader, prepare_datasets, set_seed
from lecture_04.evaluate import (
    evaluate_baseline,
    evaluate_predictions,
    forecast_after_test,
    make_future_business_dates,
    predict,
    print_metrics,
    save_predictions_plot,
)
from lecture_04.models import PriceMLP
from lecture_04.train import train_model


def main() -> None:
    data_config = DataConfig()
    train_config = TrainConfig()
    set_seed(train_config.seed)

    print("\nStart download\n")
    prices = download_close_prices(data_config)
    x_train, y_train, x_test, y_test, test_dates, scaling = prepare_datasets(
        prices,
        data_config,
    )

    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    train_loader = make_train_loader(x_train, y_train, train_config)
    criterion = nn.MSELoss()

    model = PriceMLP(window=data_config.window).to(DEVICE)
    print("\nStart train\n")
    train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        config=train_config,
    )

    print("\nStart evaluation\n")
    pred = predict(model, x_test)
    baseline_metrics = evaluate_baseline(x_test, y_test)
    model_metrics = evaluate_predictions(pred, y_test)

    print_metrics("Baseline", baseline_metrics)
    print_metrics("MLP", model_metrics)

    future_pred = forecast_after_test(
        model=model,
        x_last=x_test[-1],
        steps=data_config.forecast_steps_after_test,
    )
    future_dates = make_future_business_dates(
        last_date=test_dates[-1],
        steps=data_config.forecast_steps_after_test,
    )

    save_predictions_plot(
        test_dates,
        y_test,
        pred,
        label="MLP",
        path=RESULTS_DIR / "mlp_predictions.png",
        scaling=scaling,
        future_dates=future_dates,
        future_pred=future_pred,
    )


if __name__ == "__main__":
    main()
