import time

from sklearn.metrics import mean_absolute_error


def evaluate_model(
    model,
    X_test,
    y_test,
    mae_cv,
    n_predictions=100,
):
    y_pred = model.predict(
        X_test
    )

    mae_test = mean_absolute_error(
        y_test,
        y_pred,
    )

    sample = X_test.iloc[[0]]

    # Warm-up
    model.predict(sample)

    start = time.perf_counter()

    for _ in range(n_predictions):
        model.predict(sample)

    end = time.perf_counter()

    latency_ms = (
        (end - start)
        / n_predictions
        * 1000
    )

    return {
        "mae_cv": mae_cv,
        "mae_test": mae_test,
        "latency_ms": latency_ms,
    }


def check_quality_gate(
    metrics,
    max_mae_cv=4000,
    max_latency_ms=200,
):
    mae_ok = (
        metrics["mae_cv"]
        <= max_mae_cv
    )

    latency_ok = (
        metrics["latency_ms"]
        <= max_latency_ms
    )

    return (
        mae_ok
        and latency_ok
    )