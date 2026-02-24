"""Weather API helpers for forecast-time feature generation."""

import json
import socket
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import urlopen

import numpy
import pandas
import streamlit

from project_settings import (
    MODULE_TEMP_BASE_OFFSET,
    MODULE_TEMP_RADIATION_COEFFICIENT,
    WEATHER_FEATURE_COLUMNS,
)

REQUEST_TIMEOUT_SECONDS = 8
WEATHER_FORECAST_COLUMNS = ["DATE_TIME", "AMBIENT_TEMPERATURE", "IRRADIATION", "MODULE_TEMPERATURE"]


@streamlit.cache_data(ttl=3600)
def fetch_live_weather_forecast(latitude: float, longitude: float, timezone: str) -> pandas.DataFrame:
    """Fetch hourly Open-Meteo forecast and return model-ready weather columns."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,shortwave_radiation",
        "forecast_days": 16,
        "timezone": timezone,
    }
    url = "https://api.open-meteo.com/v1/gfs?" + urlencode(params)
    try:
        with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError):
        # Return empty table on failure so app can gracefully use fallback features.
        return pandas.DataFrame(columns=WEATHER_FORECAST_COLUMNS)

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    rad = hourly.get("shortwave_radiation", [])

    if not times:
        return pandas.DataFrame(columns=WEATHER_FORECAST_COLUMNS)

    weather_forecast_data = pandas.DataFrame(
        {
            "DATE_TIME": pandas.to_datetime(times),
            "AMBIENT_TEMPERATURE": temps,
            "IRRADIATION": rad,
        }
    )
    if hasattr(weather_forecast_data["DATE_TIME"].dt, "tz_localize"):
        weather_forecast_data["DATE_TIME"] = weather_forecast_data["DATE_TIME"].dt.tz_localize(None)

    weather_forecast_data["MODULE_TEMPERATURE"] = (
        weather_forecast_data["AMBIENT_TEMPERATURE"]
        + MODULE_TEMP_BASE_OFFSET
        + MODULE_TEMP_RADIATION_COEFFICIENT * weather_forecast_data["IRRADIATION"]
    )
    return weather_forecast_data


def merge_weather_into_timeframe(
    time_frame: pandas.DataFrame,
    weather_forecast_data: pandas.DataFrame | None,
    use_weather: bool,
    assume_last_weather: bool,
    historical_merged_data: pandas.DataFrame,
    feature_means: pandas.Series,
) -> pandas.DataFrame:
    """Attach weather values to a future time frame with robust fallbacks."""
    if not use_weather:
        return time_frame

    merged_time_frame = time_frame.copy()
    if weather_forecast_data is not None and not weather_forecast_data.empty:
        # Nearest-time merge aligns each prediction timestamp with weather timestamp.
        merged_time_frame = pandas.merge_asof(
            merged_time_frame.sort_values("DATE_TIME"),
            weather_forecast_data.sort_values("DATE_TIME"),
            on="DATE_TIME",
            direction="nearest",
            tolerance=pandas.Timedelta(minutes=45),
        )

    for column in WEATHER_FEATURE_COLUMNS:
        if column not in merged_time_frame.columns:
            merged_time_frame[column] = numpy.nan

    if weather_forecast_data is not None and not weather_forecast_data.empty:
        merged_time_frame = merged_time_frame.sort_values("DATE_TIME")
        forecast_start = weather_forecast_data["DATE_TIME"].min()
        forecast_end = weather_forecast_data["DATE_TIME"].max()
        for column in WEATHER_FEATURE_COLUMNS:
            merged_time_frame.loc[merged_time_frame["DATE_TIME"] < forecast_start, column] = (
                weather_forecast_data[column].iloc[0]
            )
            merged_time_frame.loc[merged_time_frame["DATE_TIME"] > forecast_end, column] = (
                weather_forecast_data[column].iloc[-1]
            )
        merged_time_frame[WEATHER_FEATURE_COLUMNS] = merged_time_frame[WEATHER_FEATURE_COLUMNS].ffill().bfill()

    if assume_last_weather:
        # Fallback 1: reuse latest known weather values.
        last_weather_values = historical_merged_data[WEATHER_FEATURE_COLUMNS].iloc[-1]
        for column in WEATHER_FEATURE_COLUMNS:
            merged_time_frame[column] = merged_time_frame[column].fillna(last_weather_values[column])
    else:
        # Fallback 2: use feature means from training data.
        merged_time_frame[WEATHER_FEATURE_COLUMNS] = merged_time_frame[WEATHER_FEATURE_COLUMNS].fillna(
            feature_means[WEATHER_FEATURE_COLUMNS]
        )

    return merged_time_frame


@streamlit.cache_data(ttl=86400)
def search_location_options(location_query: str, count: int = 5) -> list[dict]:
    """Search places using Open-Meteo geocoding API."""
    if not location_query.strip():
        return []

    params = {
        "name": location_query.strip(),
        "count": count,
        "language": "en",
        "format": "json",
    }
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(params)
    try:
        with urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, socket.timeout, json.JSONDecodeError):
        return []

    results = payload.get("results", [])
    cleaned = []
    for item in results:
        cleaned.append(
            {
                "name": item.get("name"),
                "country": item.get("country"),
                "admin1": item.get("admin1"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "timezone": item.get("timezone"),
            }
        )
    return cleaned

