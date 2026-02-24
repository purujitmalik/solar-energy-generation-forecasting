"""Model training and evaluation utilities."""

import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from project_settings import DEFAULT_DAYLIGHT_HOUR_RANGE, TIME_FEATURE_COLUMNS, WEATHER_FEATURE_COLUMNS
from data_processing import add_lag_and_rolling_features


def calculate_safe_mape(actual_values: pandas.Series, predicted_values: numpy.ndarray | pandas.Series) -> float:
    """MAPE with a small denominator floor to avoid division by zero."""
    return float(
        numpy.mean(numpy.abs((actual_values - predicted_values) / numpy.maximum(numpy.abs(actual_values), 1e-6))) * 100
    )


def train_forecast_model(
    training_dataframe: pandas.DataFrame,
    target_column: str,
    use_weather: bool,
    test_fraction: float,
    lags: list[int],
    rolling_windows: list[int],
) -> tuple[
    RandomForestRegressor,
    list[str],
    pandas.Series,
    pandas.DataFrame,
    numpy.ndarray,
    dict[str, float],
]:
    """Train a random-forest model and return predictions on the hold-out split."""
    # Build final feature list from time + optional weather + lag/rolling features.
    base_feature_columns = list(TIME_FEATURE_COLUMNS)
    lag_feature_columns = [f"{target_column}_lag_{lag}" for lag in lags]
    rolling_feature_columns = [f"{target_column}_roll_{window}" for window in rolling_windows]
    feature_columns = (
        base_feature_columns
        + (WEATHER_FEATURE_COLUMNS if use_weather else [])
        + lag_feature_columns
        + rolling_feature_columns
    )

    # Keep rows where target exists, then add lag features.
    training_dataframe = training_dataframe.dropna(subset=[target_column]).copy()
    training_dataframe = add_lag_and_rolling_features(training_dataframe, target_column, lags, rolling_windows)

    if use_weather and "IRRADIATION" in training_dataframe.columns:
        # Daylight-only training often improves solar prediction quality.
        training_dataframe = training_dataframe[training_dataframe["IRRADIATION"] > 0].copy()
    elif "hour" in training_dataframe.columns:
        daylight_start_hour, daylight_end_hour = DEFAULT_DAYLIGHT_HOUR_RANGE
        training_dataframe = training_dataframe[
            (training_dataframe["hour"] >= daylight_start_hour)
            & (training_dataframe["hour"] < daylight_end_hour)
        ].copy()

    training_dataframe[feature_columns] = training_dataframe[feature_columns].replace([numpy.inf, -numpy.inf], numpy.nan)
    feature_means = training_dataframe[feature_columns].mean(numeric_only=True)
    training_dataframe[feature_columns] = training_dataframe[feature_columns].fillna(feature_means)

    # Chronological split (no shuffling) for time-series behavior.
    split_index = max(1, int(len(training_dataframe) * (1 - test_fraction)))
    train_split = training_dataframe.iloc[:split_index]
    test_split = training_dataframe.iloc[split_index:]

    training_feature_matrix, training_target_values = train_split[feature_columns], train_split[target_column]
    testing_feature_matrix, testing_target_values = test_split[feature_columns], test_split[target_column]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(training_feature_matrix, training_target_values)

    predictions = model.predict(testing_feature_matrix)

    mean_absolute_error_value = mean_absolute_error(testing_target_values, predictions)
    root_mean_squared_error_value = mean_squared_error(testing_target_values, predictions, squared=False)
    mean_absolute_percentage_error_value = calculate_safe_mape(testing_target_values, predictions)

    metrics = {
        "MAE": mean_absolute_error_value,
        "RMSE": root_mean_squared_error_value,
        "MAPE (%)": mean_absolute_percentage_error_value,
    }
    return model, feature_columns, feature_means, test_split, predictions, metrics


def get_time_series_cv_metrics(
    full_dataframe: pandas.DataFrame,
    target_column: str,
    feature_columns: list[str],
    n_splits: int = 3,
    test_fraction: float = 0.2,
) -> dict[str, float] | None:
    """Compute rolling time-series CV metrics with expanding-train windows."""
    # Sort by time to avoid leakage from future rows.
    full_dataframe = full_dataframe.dropna(subset=[target_column]).copy().sort_values("DATE_TIME")

    if "hour" in full_dataframe.columns:
        lag_feature_columns = [column for column in feature_columns if "_lag_" in column or "_roll_" in column]
        if lag_feature_columns:
            lags = []
            rolls = []
            for name in lag_feature_columns:
                if "_lag_" in name:
                    lags.append(int(name.split("_lag_")[1]))
                if "_roll_" in name:
                    rolls.append(int(name.split("_roll_")[1]))
            full_dataframe = add_lag_and_rolling_features(
                full_dataframe,
                target_column,
                sorted(set(lags)),
                sorted(set(rolls)),
            )

        if "IRRADIATION" in full_dataframe.columns:
            full_dataframe = full_dataframe[full_dataframe["IRRADIATION"] > 0].copy()
        else:
            daylight_start_hour, daylight_end_hour = DEFAULT_DAYLIGHT_HOUR_RANGE
            full_dataframe = full_dataframe[
                (full_dataframe["hour"] >= daylight_start_hour) & (full_dataframe["hour"] < daylight_end_hour)
            ].copy()

    total_rows = len(full_dataframe)
    test_length = max(1, int(total_rows * test_fraction))
    # Build rolling splits: earlier data for train, later chunk for validation.
    splits = []
    for i in range(n_splits):
        train_end = total_rows - (n_splits - i) * test_length
        test_start = train_end
        test_end = min(total_rows, test_start + test_length)
        if train_end <= 1 or test_end <= test_start:
            continue
        splits.append((train_end, test_start, test_end))

    if not splits:
        return None

    metrics_per_split = []
    for train_end, test_start, test_end in splits:
        train_split = full_dataframe.iloc[:train_end]
        test_split = full_dataframe.iloc[test_start:test_end]
        training_feature_matrix, training_target_values = train_split[feature_columns], train_split[target_column]
        testing_feature_matrix, testing_target_values = test_split[feature_columns], test_split[target_column]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(training_feature_matrix, training_target_values)
        split_predictions = model.predict(testing_feature_matrix)

        mean_absolute_error_value = mean_absolute_error(testing_target_values, split_predictions)
        root_mean_squared_error_value = mean_squared_error(testing_target_values, split_predictions, squared=False)
        mean_absolute_percentage_error_value = calculate_safe_mape(testing_target_values, split_predictions)
        metrics_per_split.append(
            {
                "MAE": mean_absolute_error_value,
                "RMSE": root_mean_squared_error_value,
                "MAPE (%)": mean_absolute_percentage_error_value,
            }
        )

    if not metrics_per_split:
        return None

    return {
        "MAE": float(numpy.mean([metric["MAE"] for metric in metrics_per_split])),
        "RMSE": float(numpy.mean([metric["RMSE"] for metric in metrics_per_split])),
        "MAPE (%)": float(numpy.mean([metric["MAPE (%)"] for metric in metrics_per_split])),
    }


def evaluate_previous_day_baseline(
    historical_dataframe: pandas.DataFrame,
    test_dataframe: pandas.DataFrame,
    target_column: str,
    step_timedelta: pandas.Timedelta,
) -> tuple[dict[str, float], pandas.Series] | tuple[None, None]:
    """Baseline: predict each timestamp from the same time on the previous day."""
    steps_per_day = int(round(pandas.Timedelta(days=1) / step_timedelta))
    if steps_per_day <= 0:
        return None, None

    target_series = historical_dataframe.set_index("DATE_TIME")[target_column]
    baseline_series = target_series.shift(steps_per_day)
    baseline_predictions = baseline_series.reindex(test_dataframe["DATE_TIME"]).astype(float)

    valid_mask = baseline_predictions.notna() & test_dataframe[target_column].notna()
    if not valid_mask.any():
        return None, None

    actual_values = test_dataframe.loc[valid_mask, target_column]
    predicted_values = baseline_predictions.loc[valid_mask]

    mean_absolute_error_value = mean_absolute_error(actual_values, predicted_values)
    root_mean_squared_error_value = mean_squared_error(actual_values, predicted_values, squared=False)
    mean_absolute_percentage_error_value = calculate_safe_mape(actual_values, predicted_values)

    metrics = {
        "MAE": mean_absolute_error_value,
        "RMSE": root_mean_squared_error_value,
        "MAPE (%)": mean_absolute_percentage_error_value,
    }
    return metrics, baseline_predictions

