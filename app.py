import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Generation Forecasting", layout="wide")

DATA_DIR = Path(__file__).parent
GEN_FILES = {
    "Plant 1": "Plant_1_Generation_Data.csv",
    "Plant 2": "Plant_2_Generation_Data.csv",
}
WEATHER_FILES = {
    "Plant 1": "Plant_1_Weather_Sensor_Data.csv",
    "Plant 2": "Plant_2_Weather_Sensor_Data.csv",
}

TARGET_OPTIONS = ["AC_POWER", "DC_POWER", "DAILY_YIELD"]
WEATHER_COLUMNS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
TIME_COLUMNS = ["hour", "dayofweek", "day", "month", "weekofyear", "is_weekend"]


@st.cache_data
def load_data(gen_path: Path, weather_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)

    # Data files use day-first timestamps (e.g., 15-05-2020 00:00).
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], dayfirst=True)

    return gen, weather


def prepare_data(gen: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    agg_gen = (
        gen.groupby("DATE_TIME", as_index=False)
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

    agg_weather = (
        weather.groupby("DATE_TIME", as_index=False)
        .agg(
            {
                "AMBIENT_TEMPERATURE": "mean",
                "MODULE_TEMPERATURE": "mean",
                "IRRADIATION": "mean",
            }
        )
        .sort_values("DATE_TIME")
    )

    merged = pd.merge(agg_gen, agg_weather, on="DATE_TIME", how="inner")
    merged = merged.sort_values("DATE_TIME")

    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["hour"] = enriched["DATE_TIME"].dt.hour
    enriched["dayofweek"] = enriched["DATE_TIME"].dt.dayofweek
    enriched["day"] = enriched["DATE_TIME"].dt.day
    enriched["month"] = enriched["DATE_TIME"].dt.month
    enriched["weekofyear"] = enriched["DATE_TIME"].dt.isocalendar().week.astype(int)
    enriched["is_weekend"] = (enriched["dayofweek"] >= 5).astype(int)

    return enriched


def infer_freq(dt_series: pd.Series) -> pd.Timedelta | str:
    freq = pd.infer_freq(dt_series)
    if freq is not None:
        return freq

    deltas = dt_series.sort_values().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(minutes=15)

    return deltas.median()

def freq_to_timedelta(freq: pd.Timedelta | str, ref_dt: pd.Timestamp) -> pd.Timedelta:
    if isinstance(freq, pd.Timedelta):
        return freq
    if isinstance(freq, str):
        try:
            return pd.to_timedelta(freq)
        except ValueError:
            probe = pd.date_range(ref_dt, periods=2, freq=freq)
            return probe[1] - probe[0]
    return pd.Timedelta(freq)


def train_model(df: pd.DataFrame, target: str, use_weather: bool, test_size: float):
    base_features = list(TIME_COLUMNS)
    feature_cols = base_features + (WEATHER_COLUMNS if use_weather else [])

    df = df.dropna(subset=[target]).copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    feature_means = df[feature_cols].mean(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(feature_means)

    split_idx = max(1, int(len(df) * (1 - test_size)))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mape = np.mean(
        np.abs((y_test - predictions) / np.maximum(np.abs(y_test), 1e-6))
    ) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

    return model, feature_cols, feature_means, test, predictions, metrics


def make_future_frame(
    df: pd.DataFrame,
    horizon: int,
    freq: pd.Timedelta | str,
) -> pd.DataFrame:
    last_dt = df["DATE_TIME"].max()
    future_dt = pd.date_range(last_dt + pd.Timedelta(freq), periods=horizon, freq=freq)
    future = pd.DataFrame({"DATE_TIME": future_dt})

    return add_time_features(future)

def make_day_frame(day: pd.Timestamp, freq: pd.Timedelta | str) -> pd.DataFrame:
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    day_range = pd.date_range(day_start, day_end, freq=freq, inclusive="left")
    day_df = pd.DataFrame({"DATE_TIME": day_range})

    return add_time_features(day_df)


def main():
    st.title("Solar Energy Generation Forecasting")
    st.caption("Plant 1 forecasting dashboard with generation + weather data.")

    with st.sidebar:
        st.header("Controls")
        plant = st.selectbox("Plant", options=list(GEN_FILES.keys()), index=0)
        target = st.selectbox("Target", options=TARGET_OPTIONS, index=0)
        use_weather = st.checkbox("Use weather features", value=True)
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        horizon = st.slider("Forecast horizon (steps)", min_value=12, max_value=288, value=96, step=12)
        forecast_day = st.date_input("Predict energy for date", value=None)
        assume_last_weather = False
        if use_weather:
            assume_last_weather = st.checkbox("Assume last observed weather for future", value=True)

    gen_path = DATA_DIR / GEN_FILES[plant]
    weather_path = DATA_DIR / WEATHER_FILES[plant]

    if not gen_path.exists() or not weather_path.exists():
        st.error("CSV files not found in the project directory.")
        st.stop()

    gen, weather = load_data(gen_path, weather_path)
    merged = prepare_data(gen, weather)
    merged = add_time_features(merged)
    freq = infer_freq(merged["DATE_TIME"])
    freq_delta = freq_to_timedelta(freq, merged["DATE_TIME"].max())

    st.subheader("Dataset Overview")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{len(merged):,}")
    col_b.metric("Start", merged["DATE_TIME"].min().strftime("%Y-%m-%d %H:%M"))
    col_c.metric("End", merged["DATE_TIME"].max().strftime("%Y-%m-%d %H:%M"))

    st.dataframe(merged.head(10), use_container_width=True)

    st.subheader("Generation History")
    fig_hist = px.line(merged, x="DATE_TIME", y=target, title=f"{target} over time")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Model Training & Evaluation")
    model, feature_cols, feature_means, test_df, preds, metrics = train_model(
        merged, target, use_weather, test_size
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")

    eval_fig = go.Figure()
    eval_fig.add_trace(
        go.Scatter(
            x=test_df["DATE_TIME"],
            y=test_df[target],
            mode="lines",
            name="Actual",
        )
    )
    eval_fig.add_trace(
        go.Scatter(
            x=test_df["DATE_TIME"],
            y=preds,
            mode="lines",
            name="Predicted",
        )
    )
    eval_fig.update_layout(title="Test Split: Actual vs Predicted", xaxis_title="Time")
    st.plotly_chart(eval_fig, use_container_width=True)

    st.subheader("Forecast")
    last_dt = merged["DATE_TIME"].max()
    if forecast_day is not None:
        day_end = pd.Timestamp(forecast_day).normalize() + pd.Timedelta(days=1)
        if day_end > last_dt:
            extra_steps = int(np.ceil((day_end - last_dt) / freq_delta))
            horizon = max(horizon, extra_steps)
    future = make_future_frame(merged, horizon=horizon, freq=freq)

    if use_weather:
        if assume_last_weather:
            last_weather = merged[WEATHER_COLUMNS].iloc[-1]
            for col in WEATHER_COLUMNS:
                future[col] = last_weather[col]
        else:
            for col in WEATHER_COLUMNS:
                future[col] = np.nan
            future[WEATHER_COLUMNS] = future[WEATHER_COLUMNS].fillna(
                feature_means[WEATHER_COLUMNS]
            )

    future[feature_cols] = future[feature_cols].fillna(feature_means)
    forecast = model.predict(future[feature_cols])

    forecast_df = pd.DataFrame(
        {"DATE_TIME": future["DATE_TIME"], "Forecast": forecast}
    )

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=merged["DATE_TIME"],
            y=merged[target],
            mode="lines",
            name="History",
        )
    )
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast_df["DATE_TIME"],
            y=forecast_df["Forecast"],
            mode="lines",
            name="Forecast",
        )
    )
    forecast_fig.update_layout(title="Forecast", xaxis_title="Time")
    st.plotly_chart(forecast_fig, use_container_width=True)

    st.subheader("Predict Energy For a Date")
    if forecast_day is None:
        st.info("Pick a date in the sidebar to get a full-day energy prediction.")
    else:
        day_frame = make_day_frame(pd.Timestamp(forecast_day), freq=freq)

        if use_weather:
            if assume_last_weather:
                last_weather = merged[WEATHER_COLUMNS].iloc[-1]
                for col in WEATHER_COLUMNS:
                    day_frame[col] = last_weather[col]
            else:
                for col in WEATHER_COLUMNS:
                    day_frame[col] = np.nan
                day_frame[WEATHER_COLUMNS] = day_frame[WEATHER_COLUMNS].fillna(
                    feature_means[WEATHER_COLUMNS]
                )

        day_frame[feature_cols] = day_frame[feature_cols].fillna(feature_means)
        day_pred = model.predict(day_frame[feature_cols])
        day_pred_df = pd.DataFrame(
            {"DATE_TIME": day_frame["DATE_TIME"], "Prediction": day_pred}
        )

        predicted_total = day_pred_df["Prediction"].sum()

        actual_mask = merged["DATE_TIME"].dt.date == pd.Timestamp(forecast_day).date()
        actual_total = None
        if actual_mask.any():
            actual_total = merged.loc[actual_mask, target].sum()

        cols = st.columns(2)
        cols[0].metric("Predicted total", f"{predicted_total:,.2f}")
        if actual_total is not None:
            cols[1].metric("Actual total", f"{actual_total:,.2f}")
        else:
            cols[1].metric("Actual total", "N/A")

        day_fig = go.Figure()
        day_fig.add_trace(
            go.Scatter(
                x=day_pred_df["DATE_TIME"],
                y=day_pred_df["Prediction"],
                mode="lines",
                name="Predicted",
            )
        )
        if actual_total is not None:
            actual_day = merged.loc[actual_mask, ["DATE_TIME", target]]
            day_fig.add_trace(
                go.Scatter(
                    x=actual_day["DATE_TIME"],
                    y=actual_day[target],
                    mode="lines",
                    name="Actual",
                )
            )
        day_fig.update_layout(title="Daily prediction profile", xaxis_title="Time")
        st.plotly_chart(day_fig, use_container_width=True)

    st.caption(
        "Future forecasts use time features and optionally last observed weather. "
        "Replace with real weather forecasts for higher accuracy."
    )


if __name__ == "__main__":
    main()
