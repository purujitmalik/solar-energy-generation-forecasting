"""Data loading and feature engineering helpers."""

import numpy
import pandas
import streamlit


@streamlit.cache_data
def load_csv_data(
    generation_csv_path: str, weather_csv_path: str
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Load generation and weather CSV files and parse `DATE_TIME`."""
    generation_data = pandas.read_csv(generation_csv_path)
    weather_data = pandas.read_csv(weather_csv_path)

    generation_data["DATE_TIME"] = pandas.to_datetime(generation_data["DATE_TIME"], dayfirst=True)
    weather_data["DATE_TIME"] = pandas.to_datetime(weather_data["DATE_TIME"], dayfirst=True)

    return generation_data, weather_data


def merge_generation_and_weather(
    generation_data: pandas.DataFrame, weather_data: pandas.DataFrame
) -> pandas.DataFrame:
    """Aggregate duplicate timestamps and merge generation + weather data."""
    # Multiple inverters/sensors can share a timestamp.
    # We aggregate first so each timestamp has one row.
    grouped_generation_data = (
        generation_data.groupby("DATE_TIME", as_index=False)
        .agg(
            {
                "AC_POWER": "sum",
                "DC_POWER": "sum",
                "DAILY_YIELD": "sum",
                "TOTAL_YIELD": "max",
            }
        )
        .sort_values("DATE_TIME")
    )

    grouped_weather_data = (
        weather_data.groupby("DATE_TIME", as_index=False)
        .agg(
            {
                "AMBIENT_TEMPERATURE": "mean",
                "MODULE_TEMPERATURE": "mean",
                "IRRADIATION": "mean",
            }
        )
        .sort_values("DATE_TIME")
    )

    merged_data = pandas.merge(grouped_generation_data, grouped_weather_data, on="DATE_TIME", how="inner")
    return merged_data.sort_values("DATE_TIME")


def add_calendar_features(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """Add calendar/time features used by the model."""
    data_with_time_features = dataframe.copy()
    data_with_time_features["hour"] = data_with_time_features["DATE_TIME"].dt.hour
    data_with_time_features["dayofweek"] = data_with_time_features["DATE_TIME"].dt.dayofweek
    data_with_time_features["day"] = data_with_time_features["DATE_TIME"].dt.day
    data_with_time_features["month"] = data_with_time_features["DATE_TIME"].dt.month
    data_with_time_features["weekofyear"] = data_with_time_features["DATE_TIME"].dt.isocalendar().week.astype(int)
    data_with_time_features["is_weekend"] = (data_with_time_features["dayofweek"] >= 5).astype(int)
    return data_with_time_features


def build_lag_settings(step_timedelta: pandas.Timedelta) -> tuple[list[int], list[int]]:
    """Create lag and rolling-window sizes from dataset frequency."""
    # Convert time step to "how many rows in a day".
    steps_per_day = int(round(pandas.Timedelta(days=1) / step_timedelta))
    steps_per_day = max(steps_per_day, 1)
    half_day = max(1, steps_per_day // 2)
    lags = sorted({1, 2, 4, 8, 12, 24, half_day, steps_per_day})
    rolling_windows = sorted({4, 8, 12, 24, half_day})
    return lags, rolling_windows


def add_lag_and_rolling_features(
    dataframe: pandas.DataFrame,
    target_column: str,
    lags: list[int],
    rolling_windows: list[int],
) -> pandas.DataFrame:
    """Create lag and rolling-mean features for a target column."""
    enriched = dataframe.copy()
    for lag in lags:
        enriched[f"{target_column}_lag_{lag}"] = enriched[target_column].shift(lag)
    for window in rolling_windows:
        enriched[f"{target_column}_roll_{window}"] = (
            enriched[target_column].shift(1).rolling(window=window, min_periods=1).mean()
        )
    return enriched


def add_future_lag_features_using_history(
    future_frame: pandas.DataFrame,
    history_frame: pandas.DataFrame,
    target_column: str,
    lags: list[int],
    rolling_windows: list[int],
) -> pandas.DataFrame:
    """Build lag features for future rows by appending them after historical rows."""
    # Temporary combine history + future so shifting works naturally.
    history_target = history_frame[["DATE_TIME", target_column]].copy()
    future_target = future_frame[["DATE_TIME"]].copy()
    future_target[target_column] = numpy.nan
    combined_data = pandas.concat([history_target, future_target], ignore_index=True).sort_values("DATE_TIME")
    combined_data = add_lag_and_rolling_features(combined_data, target_column, lags, rolling_windows)
    future_with_features = combined_data.loc[combined_data[target_column].isna()].drop(columns=[target_column])
    return future_frame.merge(future_with_features, on="DATE_TIME", how="left")


def infer_time_frequency(date_time_series: pandas.Series) -> pandas.Timedelta | str:
    """Infer timestamp frequency; fall back to median delta if needed."""
    # Try pandas built-in inference first.
    freq = pandas.infer_freq(date_time_series)
    if freq is not None:
        return freq

    # Fallback: use median gap between timestamps.
    deltas = date_time_series.sort_values().diff().dropna()
    if deltas.empty:
        return pandas.Timedelta(minutes=15)

    return deltas.median()


def convert_frequency_to_timedelta(
    frequency_value: pandas.Timedelta | str, reference_datetime: pandas.Timestamp
) -> pandas.Timedelta:
    """Convert a pandas frequency string/object to a concrete `Timedelta`."""
    if isinstance(frequency_value, pandas.Timedelta):
        return frequency_value
    if isinstance(frequency_value, str):
        try:
            return pandas.to_timedelta(frequency_value)
        except ValueError:
            probe = pandas.date_range(reference_datetime, periods=2, freq=frequency_value)
            return probe[1] - probe[0]
    return pandas.Timedelta(frequency_value)


def build_future_timeframe(
    historical_dataframe: pandas.DataFrame, forecast_steps: int, frequency_value: pandas.Timedelta | str
) -> pandas.DataFrame:
    """Create the next `horizon` timestamps after the last observed row."""
    last_datetime = historical_dataframe["DATE_TIME"].max()
    future_datetimes = pandas.date_range(
        last_datetime + pandas.Timedelta(frequency_value),
        periods=forecast_steps,
        freq=frequency_value,
    )
    future_frame = pandas.DataFrame({"DATE_TIME": future_datetimes})
    return add_calendar_features(future_frame)


def build_single_day_timeframe(day: pandas.Timestamp, frequency_value: pandas.Timedelta | str) -> pandas.DataFrame:
    """Create timestamps for one full day using the same data frequency."""
    day_start = pandas.Timestamp(day).normalize()
    day_end = day_start + pandas.Timedelta(days=1)
    day_range = pandas.date_range(day_start, day_end, freq=frequency_value, inclusive="left")
    day_frame = pandas.DataFrame({"DATE_TIME": day_range})
    return add_calendar_features(day_frame)


def build_date_range_timeframe(
    start_day: pandas.Timestamp, end_day: pandas.Timestamp, frequency_value: pandas.Timedelta | str
) -> pandas.DataFrame:
    """Create timestamps for an inclusive date range."""
    range_start = pandas.Timestamp(start_day).normalize()
    range_end = pandas.Timestamp(end_day).normalize() + pandas.Timedelta(days=1)
    date_range = pandas.date_range(range_start, range_end, freq=frequency_value, inclusive="left")
    range_frame = pandas.DataFrame({"DATE_TIME": date_range})
    return add_calendar_features(range_frame)
