# Solar Energy Forecasting Dashboard (Streamlit)

This app loads the provided solar generation and weather datasets, builds a simple forecasting model, and provides interactive charts and metrics.

## Quickstart

1. Create a virtual environment (optional).
2. Install dependencies.
3. Run the app.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Default target is `AC_POWER` for Plant 1.
- The model uses time features and optional weather features.
- Future forecasts can assume the last observed weather or use feature means.
