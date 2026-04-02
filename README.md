# solarcast

`solarcast` is an interactive Streamlit application for solar generation forecasting.  
It combines historical plant generation data, weather telemetry, machine-learning prediction, user panel scaling, and location-based live weather inputs.

## Features

- Plant-level forecasting from provided CSV datasets (`Plant 1` / `Plant 2`).
- Target selection:
  - `AC_POWER`
  - `DC_POWER`
  - `DAILY_YIELD`
- User-side panel scaling:
  - Enter your own panel count.
  - Forecasts are automatically scaled from dataset plant size to your system size.
- Location-aware weather:
  - Live typeahead city/location selector.
  - Open-Meteo geocoding + weather forecast/archive integration.
  - Graceful fallback to learned hourly weather profile when live weather is unavailable.
- Model quality view:
  - MAE, RMSE, MAPE
  - Actual vs predicted validation chart
- Forecast workflow:
  - Forward horizon forecast chart
  - Day-specific profile and predicted total

## Project Structure

- `app.py`: Main Streamlit app and forecasting pipeline.
- `Plant_1_Generation_Data.csv`, `Plant_2_Generation_Data.csv`: Generation datasets.
- `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_Weather_Sensor_Data.csv`: Weather sensor datasets.
- `requirements.txt`: Python dependencies.

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn (`RandomForestRegressor`)
- Plotly
- Requests
- `streamlit-searchbox`

## Setup

1. (Recommended) create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m streamlit run app.py
```

4. Open in browser:

- `http://localhost:8501`

## Usage Flow

1. Select plant and prediction target from the sidebar.
2. Enter your panel count in `Your Solar Setup`.
3. Select a location in `Prediction Location` (typeahead).
4. Choose forecast horizon and date.
5. Review results across tabs:
   - `Overview`
   - `Model Quality`
   - `Forecast`
   - `Daily Planner`

## Notes

- Date parsing assumes day-first timestamp format in the provided datasets.
- Live weather uses Open-Meteo APIs:
  - Geocoding API for location resolution
  - Archive API for historical range
  - Forecast API for future range (limited horizon)
- When live weather cannot be used, the app falls back to historical hourly weather behavior to keep predictions stable.

## Troubleshooting

- If location dropdown or styling looks stale, restart Streamlit and do a hard refresh (`Ctrl+F5`).
- If imports fail after package changes, reinstall dependencies:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

- If Streamlit port is busy, run on another port:

```bash
python -m streamlit run app.py --server.port 8502
```
