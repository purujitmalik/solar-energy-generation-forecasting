# Solar Energy Forecasting Dashboard (Streamlit)

This project predicts short-term solar power generation using:
- historical plant generation data
- weather sensor data
- optional live weather forecast from Open-Meteo

The app is built with Streamlit and is designed to be easy to run and explain.

## Project Structure

- `main.py`: Streamlit UI and end-to-end app flow
- `project_settings.py`: constants (file names, default settings, feature lists)
- `data_processing.py`: data loading + feature engineering
- `model_training.py`: training and evaluation functions
- `weather_forecast.py`: weather API and weather merge helpers
- `forecasting.py`: autoregressive prediction and forecast post-processing

## Quick Start

1. (Optional) Create and activate a virtual environment.
2. Install dependencies.
3. Start the Streamlit app.

```bash
pip install -r requirements.txt
streamlit run main.py
```

## How the App Works

1. Read generation and weather CSV files.
2. Aggregate by timestamp and merge into one dataset.
3. Add time-based features (`hour`, `dayofweek`, `month`, etc.).
4. Train a Random Forest model on historical data.
5. Build future timestamps for selected horizon/day/range.
6. Attach weather features (live forecast if available; fallback otherwise).
7. Predict step-by-step (autoregressive) for future timestamps.
8. Show metrics, charts, daily totals, and comparisons with actuals.

## Main Sidebar Controls

- `Plant`: choose Plant 1 or Plant 2
- `Target`: choose `AC_POWER`, `DC_POWER`, or `DAILY_YIELD`
- `Use weather features`: include weather columns in modeling
- `Test split`: portion of data used for evaluation
- `Forecast horizon (steps)`: number of future timestamps to predict
- `Predict energy for date`: single-day prediction
- `Range start` and `Range end`: multi-day prediction

## Notes

- Default target is `AC_POWER` for Plant 1.
- Forecast horizon is in data steps (for 15-min data, 96 steps = 24 hours).
- Night-time predictions are forced to `0` using irradiation or daylight hours.

