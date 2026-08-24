import numpy as np


def mae(actual, predicted):
    """
    Mean Absolute Error.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return np.mean(
        np.abs(actual - predicted)
    )


def mape(actual, predicted):
    """
    Mean Absolute Percentage Error.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def walk_forward_validation(
    model_class,
    series,
    train_size,
    horizon,
    step
):
    """
    Perform walk-forward validation for one model.
    """

    results = []

    train_end = train_size

    while train_end + horizon <= len(series):

        train = series.iloc[:train_end]

        test = series.iloc[
            train_end:train_end + horizon
        ]

        # Create a fresh model for every fold
        model = model_class()

        model.fit(train)

        predictions = model.predict(horizon)

        mod_mae = mae(
            test,
            predictions
        )

        mod_mape = mape(
            test,
            predictions
        )

        results.append({
            "MAE": mod_mae,
            "MAPE": mod_mape
        })

        train_end += step

    return results


def evaluate_walk_forward(
    model_class,
    series,
    train_size,
    horizon,
    step
):
    """
    Evaluate one model using walk-forward validation.
    """

    fold_results = walk_forward_validation(
        model_class,
        series,
        train_size,
        horizon,
        step
    )

    if not fold_results:
        raise ValueError(
            "No walk-forward validation folds were created."
        )

    average_mae = np.mean([
        result["MAE"]
        for result in fold_results
    ])

    average_mape = np.mean([
        result["MAPE"]
        for result in fold_results
    ])

    return {
        "model": model_class.name,
        "MAE": average_mae,
        "MAPE": average_mape,
        "folds": fold_results
    }


def best_model_walk_forward(
    model_classes,
    series,
    train_size,
    horizon,
    step
):
    """
    Evaluate multiple forecasting models using
    walk-forward validation and select the best one.
    """

    results = []

    for model_class in model_classes:

        result = evaluate_walk_forward(
            model_class,
            series,
            train_size,
            horizon,
            step
        )

        result["model_class"] = model_class

        results.append(result)

    best = min(
        results,
        key=lambda x: x["MAPE"]
    )

    return best, results