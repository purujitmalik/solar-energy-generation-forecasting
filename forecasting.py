"""Forecast post-processing helpers used by the Streamlit app."""

import numpy
import pandas


def predict_step_by_step(
    model,
    history_data: pandas.DataFrame,
    prediction_frame: pandas.DataFrame,
    target_column: str,
    feature_columns: list[str],
    feature_means: pandas.Series,
    lags: list[int],
    rolling_windows: list[int],
) -> tuple[numpy.ndarray, pandas.DataFrame]:
    """Predict sequentially so each new prediction becomes history for the next step."""
    prediction_frame = prediction_frame.sort_values("DATE_TIME").copy()
    history_values = history_data[target_column].dropna().astype(float).tolist()
    predicted_values = []

    for lag in lags:
        column_name = f"{target_column}_lag_{lag}"
        if column_name not in prediction_frame.columns:
            prediction_frame[column_name] = numpy.nan
    for window in rolling_windows:
        column_name = f"{target_column}_roll_{window}"
        if column_name not in prediction_frame.columns:
            prediction_frame[column_name] = numpy.nan

    for row_idx in prediction_frame.index:
        # Recompute lag/rolling values for this timestamp from growing history.
        for lag in lags:
            lag_column = f"{target_column}_lag_{lag}"
            history_position = len(history_values) - lag
            prediction_frame.at[row_idx, lag_column] = (
                history_values[history_position] if history_position >= 0 else numpy.nan
            )

        for window in rolling_windows:
            roll_column = f"{target_column}_roll_{window}"
            roll_values = history_values[-window:]
            prediction_frame.at[row_idx, roll_column] = float(numpy.mean(roll_values)) if roll_values else numpy.nan

        model_input_row = (
            prediction_frame.loc[[row_idx], feature_columns]
            .replace([numpy.inf, -numpy.inf], numpy.nan)
            .fillna(feature_means)
        )
        predicted_value = float(model.predict(model_input_row)[0])
        # Append prediction so next step can use it as lag history.
        predicted_values.append(predicted_value)
        history_values.append(predicted_value)

    return numpy.array(predicted_values), prediction_frame


def clamp_to_historical_range(
    predictions: numpy.ndarray, history_data: pandas.DataFrame, target_column: str
) -> numpy.ndarray:
    """Clip predictions to historical min/max to avoid unrealistic spikes."""
    historical_target_values = history_data[target_column].dropna().astype(float)
    if historical_target_values.empty:
        return predictions
    lower = float(max(0.0, historical_target_values.min()))
    upper = float(historical_target_values.max())
    if upper <= lower:
        return numpy.maximum(predictions, 0.0)
    return numpy.clip(predictions, lower, upper)


def cap_predictions_by_daily_history(
    predicted_data: pandas.DataFrame,
    value_column: str,
    history_data: pandas.DataFrame,
    target_column: str,
) -> tuple[pandas.DataFrame, float, int]:
    """Scale down daily totals if they exceed the historical maximum daily total."""
    historical_daily_totals = (
        history_data.assign(DATE=history_data["DATE_TIME"].dt.date)
        .groupby("DATE", as_index=False)[target_column]
        .sum()
    )
    if historical_daily_totals.empty:
        return predicted_data, 0.0, 0

    max_daily = float(historical_daily_totals[target_column].max())
    if max_daily <= 0:
        return predicted_data, max_daily, 0

    adjusted_predictions = predicted_data.copy()
    adjusted_predictions["DATE"] = adjusted_predictions["DATE_TIME"].dt.date
    predicted_daily_totals = adjusted_predictions.groupby("DATE", as_index=False)[value_column].sum()
    # Compute one scale factor per day (only scales down, never up).
    predicted_daily_totals["scale"] = (max_daily / predicted_daily_totals[value_column]).clip(upper=1.0)
    capped_day_count = int((predicted_daily_totals[value_column] > max_daily).sum())
    scaling_by_day = dict(zip(predicted_daily_totals["DATE"], predicted_daily_totals["scale"]))
    adjusted_predictions[value_column] = (
        adjusted_predictions[value_column] * adjusted_predictions["DATE"].map(scaling_by_day).fillna(1.0)
    )
    adjusted_predictions = adjusted_predictions.drop(columns=["DATE"])
    return adjusted_predictions, max_daily, capped_day_count
