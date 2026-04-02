import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import requests
try:
    from streamlit_searchbox import st_searchbox
except ImportError:
    st_searchbox = None
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Solar Generation Forecasting", layout="wide")

APP_NAME = "solarcast"
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
TARGET_UNITS = {
    "AC_POWER": "kW",
    "DC_POWER": "kW",
    "DAILY_YIELD": "kWh",
}


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


def estimate_reference_panels(df: pd.DataFrame, panel_watt_w: float) -> int:
    panel_kw = max(panel_watt_w, 1.0) / 1000.0
    peak_dc_value = max(df["DC_POWER"].quantile(0.99), 1.0)
    # Plant DC power in this dataset is typically recorded in watts; convert to kW.
    peak_dc_kw = peak_dc_value / 1000.0 if peak_dc_value > 5000 else peak_dc_value
    return max(1, int(round(peak_dc_kw / panel_kw)))


@st.cache_data(ttl=3600)
def search_locations(location_query: str, count: int = 8) -> list[dict]:
    if not location_query.strip():
        return []

    url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(
        url,
        params={"name": location_query, "count": count, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return []

    normalized = []
    for r in results:
        normalized.append(
            {
                "name": r.get("name", ""),
                "country": r.get("country", ""),
                "admin1": r.get("admin1", ""),
                "latitude": r.get("latitude"),
                "longitude": r.get("longitude"),
                "timezone": r.get("timezone", "auto"),
            }
        )
    return normalized


def format_location_label(location: dict) -> str:
    parts = [location.get("name", ""), location.get("admin1", ""), location.get("country", "")]
    return ", ".join([p for p in parts if p])


def search_location_options(search_term: str) -> list[tuple[str, dict]]:
    if not search_term.strip():
        return []
    try:
        suggestions = search_locations(search_term, count=8)
    except Exception:
        return []
    options: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for loc in suggestions:
        label = format_location_label(loc)
        if label and label.lower() not in seen:
            seen.add(label.lower())
            options.append((label, loc))
    return options


def resolve_geo_from_label_or_query(
    selected_location: dict | str | None, location_query: str
) -> dict | None:
    if isinstance(selected_location, dict):
        return selected_location

    selected_location_label = (
        selected_location.strip() if isinstance(selected_location, str) else ""
    )
    query_candidates: list[str] = []

    if selected_location_label:
        parts = [p.strip() for p in selected_location_label.split(",") if p.strip()]
        if parts:
            # Prefer city-only lookup to avoid no-hit strings like "Delhi, Delhi, India".
            query_candidates.append(parts[0])
            if len(parts) >= 2:
                query_candidates.append(f"{parts[0]}, {parts[-1]}")
        query_candidates.append(selected_location_label)

    if location_query.strip():
        query_candidates.append(location_query.strip())

    # Preserve order while removing duplicates.
    deduped_queries = list(dict.fromkeys([q for q in query_candidates if q]))
    for query in deduped_queries:
        results = search_locations(query, count=8)
        if not results:
            continue
        if selected_location_label:
            exact = next(
                (r for r in results if format_location_label(r).lower() == selected_location_label.lower()),
                None,
            )
            if exact is not None:
                return exact
        return results[0]

    return None


@st.cache_data(ttl=1800)
def fetch_live_weather(
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    def _parse_hourly(payload: dict) -> pd.DataFrame:
        hourly = payload.get("hourly", {})
        time_vals = hourly.get("time", [])
        if not time_vals:
            return pd.DataFrame(columns=["DATE_TIME"] + WEATHER_COLUMNS)
        wx = pd.DataFrame(
            {
                "DATE_TIME": pd.to_datetime(time_vals),
                "AMBIENT_TEMPERATURE": pd.to_numeric(
                    hourly.get("temperature_2m", []), errors="coerce"
                ),
                "IRRADIATION": pd.to_numeric(
                    hourly.get("shortwave_radiation", []), errors="coerce"
                ),
            }
        )
        # Approximate module temperature from ambient + irradiation impact.
        wx["MODULE_TEMPERATURE"] = wx["AMBIENT_TEMPERATURE"] + 0.02 * wx["IRRADIATION"].fillna(0)
        return wx[["DATE_TIME"] + WEATHER_COLUMNS]

    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    today = pd.Timestamp.now().normalize()
    frames: list[pd.DataFrame] = []
    common_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,shortwave_radiation",
        "timezone": timezone or "auto",
    }

    # Historical part via archive endpoint.
    historical_end = min(end_day, today - pd.Timedelta(days=1))
    if start_day <= historical_end:
        archive_params = {
            **common_params,
            "start_date": start_day.strftime("%Y-%m-%d"),
            "end_date": historical_end.strftime("%Y-%m-%d"),
        }
        archive_resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=archive_params,
            timeout=20,
        )
        archive_resp.raise_for_status()
        frames.append(_parse_hourly(archive_resp.json()))

    # Forecast part via forecast endpoint (Open-Meteo forecast window is limited).
    if end_day >= today:
        max_forecast_end = today + pd.Timedelta(days=15)
        forecast_end = min(end_day, max_forecast_end)
        if forecast_end >= today:
            forecast_days = int((forecast_end - today).days) + 1
            forecast_params = {**common_params, "forecast_days": forecast_days}
            forecast_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params=forecast_params,
                timeout=20,
            )
            forecast_resp.raise_for_status()
            frames.append(_parse_hourly(forecast_resp.json()))

    if not frames:
        return pd.DataFrame(columns=["DATE_TIME"] + WEATHER_COLUMNS)

    weather_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["DATE_TIME"])
        .sort_values("DATE_TIME")
    )
    range_start = start_day
    range_end = end_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    weather_df = weather_df[
        (weather_df["DATE_TIME"] >= range_start) & (weather_df["DATE_TIME"] <= range_end)
    ]
    return weather_df[["DATE_TIME"] + WEATHER_COLUMNS]


def inject_weather_features(
    frame: pd.DataFrame, weather_frame: pd.DataFrame, feature_means: pd.Series
) -> pd.DataFrame:
    enriched = frame.copy()
    if weather_frame.empty:
        for col in WEATHER_COLUMNS:
            enriched[col] = feature_means.get(col, np.nan)
        return enriched

    wx = weather_frame.copy().sort_values("DATE_TIME").set_index("DATE_TIME")
    idx = pd.DatetimeIndex(enriched["DATE_TIME"])
    aligned = (
        wx.reindex(idx.union(wx.index))
        .sort_index()
        .interpolate(method="time", limit_direction="both")
        .reindex(idx)
    )
    aligned.index = enriched.index
    for col in WEATHER_COLUMNS:
        enriched[col] = aligned[col]
    enriched[WEATHER_COLUMNS] = enriched[WEATHER_COLUMNS].fillna(feature_means[WEATHER_COLUMNS])
    return enriched


def inject_typical_hourly_weather(
    frame: pd.DataFrame, source_weather_df: pd.DataFrame, feature_means: pd.Series
) -> pd.DataFrame:
    enriched = frame.copy()
    if source_weather_df.empty:
        for col in WEATHER_COLUMNS:
            enriched[col] = feature_means.get(col, np.nan)
        return enriched

    weather_with_hour = source_weather_df.copy()
    weather_with_hour["hour"] = weather_with_hour["DATE_TIME"].dt.hour
    hourly_profile = (
        weather_with_hour.groupby("hour", as_index=True)[WEATHER_COLUMNS]
        .mean(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
    )

    for col in WEATHER_COLUMNS:
        enriched[col] = enriched["hour"].map(hourly_profile[col].to_dict())
    enriched[WEATHER_COLUMNS] = enriched[WEATHER_COLUMNS].fillna(feature_means[WEATHER_COLUMNS])
    return enriched


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
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(
        np.abs((y_test - predictions) / np.maximum(np.abs(y_test), 1e-6))
    ) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

    return model, feature_cols, feature_means, test, predictions, metrics


def make_future_frame(
    df: pd.DataFrame,
    horizon: int,
    freq: pd.Timedelta,
) -> pd.DataFrame:
    last_dt = df["DATE_TIME"].max()
    future_dt = pd.date_range(last_dt + freq, periods=horizon, freq=freq)
    future = pd.DataFrame({"DATE_TIME": future_dt})

    return add_time_features(future)


def make_day_frame(day: pd.Timestamp, freq: pd.Timedelta) -> pd.DataFrame:
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    day_range = pd.date_range(day_start, day_end, freq=freq, inclusive="left")
    day_df = pd.DataFrame({"DATE_TIME": day_range})

    return add_time_features(day_df)


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --brand-1: #7ec8ff;
            --brand-2: #38bdf8;
            --brand-3: #0f172a;
            --text-soft: #b9c7d8;
            --card-border: #243247;
        }
        .stApp {
            background: radial-gradient(circle at 10% 0%, #0b1220 0%, #0a0f1a 45%, #070b14 100%);
            color: #e6edf5;
        }
        .hero {
            padding: 1rem 1.2rem;
            border: 1px solid var(--card-border);
            border-radius: 14px;
            background: linear-gradient(120deg, #111a2b 0%, #0d1626 100%);
            margin-bottom: 0.8rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.2rem, 5vw, 4.2rem);
            font-weight: 900;
            letter-spacing: 0.08em;
            line-height: 1;
            text-transform: lowercase;
            background: linear-gradient(90deg, #cbe9ff 0%, #7ed0ff 45%, #3ab7ff 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 10px 30px rgba(0, 163, 255, 0.18);
        }
        .hero p {
            margin: 0.35rem 0 0;
            color: var(--text-soft);
            font-size: 0.96rem;
        }
        .section-title {
            font-size: 1.06rem;
            font-weight: 700;
            color: #cfe7ff;
            margin-top: 0.3rem;
            margin-bottom: 0.4rem;
        }
        .status-chip {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            border: 1px solid #2f4662;
            background: #101d31;
            color: #9ed6ff;
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #afbdcc 0%, #a7b6c6 100%);
            border-right: 1px solid #879cb1;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #1c3046 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #274159 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="input"] > div,
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stDateInput > div > div,
        section[data-testid="stSidebar"] .stTextInput > div > div,
        section[data-testid="stSidebar"] .stNumberInput > div > div {
            background: #c2cedb !important;
            border: 1px solid #8399af !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            color: #1c3046 !important;
            background: transparent !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="input"] svg,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: #1c3046 !important;
            fill: #1c3046 !important;
        }
        section[data-testid="stSidebar"] .stNumberInput button {
            background: #c2cedb !important;
            border-color: #8399af !important;
            color: #1c3046 !important;
        }
        section[data-testid="stSidebar"] .stNumberInput button span,
        section[data-testid="stSidebar"] .stNumberInput button svg {
            color: #1c3046 !important;
            fill: #1c3046 !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTickBarMin"] {
            background: #6f89a4 !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stTickBarMax"] {
            background: #95aabd !important;
        }
        section[data-testid="stSidebar"] .stSlider [role="slider"] {
            background: #5f7c99 !important;
            border: 2px solid #4f6b86 !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
            background: #c2cedb !important;
            border: 1px solid #8399af !important;
            color: #1c3046 !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: #c2cedb !important;
            border: 1px solid #8399af !important;
            color: #1c3046 !important;
            border-radius: 10px !important;
        }
        section[data-testid="stSidebar"] iframe[title="streamlit_searchbox.searchbox"] {
            border-radius: 10px;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="tooltip"],
        div[data-baseweb="menu"] {
            background: #c2cedb !important;
            color: #1c3046 !important;
            border: 1px solid #8399af !important;
        }
        div[data-baseweb="menu"] [role="option"] {
            background: #c2cedb !important;
            color: #1c3046 !important;
        }
        div[data-baseweb="menu"] [role="option"][aria-selected="true"] {
            background: #a6b8ca !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig: go.Figure, title: str, yaxis_title: str | None = None) -> None:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=65, b=20),
        plot_bgcolor="#0f1726",
        paper_bgcolor="#0f1726",
        font=dict(color="#dbe8f7"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#23344c")
    fig.update_yaxes(showgrid=True, gridcolor="#23344c")
    if yaxis_title is not None:
        fig.update_yaxes(title=yaxis_title)


def render_app_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_NAME}</h1>
            <p>Forecast production with plant telemetry, panel scaling, and location-aware live weather integration.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_location_searchbox_styles() -> dict:
    return {
        "wrapper": {"backgroundColor": "#c2cedb"},
        "searchbox": {
            "control": {
                "backgroundColor": "#c2cedb",
                "borderColor": "#8399af",
                "color": "#1c3046",
            },
            "singleValue": {"color": "#1c3046"},
            "input": {"color": "#1c3046"},
            "placeholder": {"color": "#4d647b"},
            "menuList": {"backgroundColor": "#c2cedb", "color": "#1c3046"},
            "option": {
                "color": "#1c3046",
                "backgroundColor": "#c2cedb",
                "highlightColor": "#a6b8ca",
            },
        },
        "dropdown": {"fill": "#1c3046", "width": 16, "height": 16},
        "clear": {
            "icon": "cross",
            "clearable": "always",
            "fill": "#1c3046",
            "stroke": "#1c3046",
            "stroke-width": 2,
            "width": 16,
            "height": 16,
        },
    }


def get_primary_controls() -> tuple[str, str, bool, float, int, pd.Timestamp]:
    with st.sidebar:
        st.header("Controls")
        plant = st.selectbox("Plant", options=list(GEN_FILES.keys()), index=0)
        target = st.selectbox("Target", options=TARGET_OPTIONS, index=0)
        use_weather = st.checkbox("Use weather features", value=True)
        test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        horizon = st.slider("Forecast horizon (steps)", min_value=12, max_value=288, value=96, step=12)
        forecast_day = st.date_input("Predict energy for date", value=pd.Timestamp.now().date())
    return plant, target, use_weather, test_size, horizon, pd.Timestamp(forecast_day)


def get_setup_controls(
    merged: pd.DataFrame, use_weather: bool
) -> tuple[int, object, str, bool, bool, int]:
    selected_location = None
    location_query = ""
    with st.sidebar:
        st.divider()
        st.subheader("Your Solar Setup")
        panel_wattage = st.number_input(
            "Average panel wattage (W)",
            min_value=50.0,
            max_value=1000.0,
            value=330.0,
            step=10.0,
        )
    estimated_ref_panels = estimate_reference_panels(merged, panel_wattage)

    with st.sidebar:
        user_panel_count = st.number_input("Your panel count", min_value=1, value=100, step=1)
        st.divider()
        st.subheader("Prediction Location")
        if st_searchbox is not None:
            st.markdown("<span style='color:#000000;font-weight:600;'>City / location</span>", unsafe_allow_html=True)
            selected_location = st_searchbox(
                search_function=search_location_options,
                placeholder="Start typing your city...",
                label=None,
                key="location_typeahead",
                default=None,
                default_searchterm="Ahmedabad",
                default_use_searchterm=True,
                edit_after_submit="current",
                style_overrides=get_location_searchbox_styles(),
            )
            if isinstance(selected_location, dict):
                st.caption(f"Using: {format_location_label(selected_location)}")
            elif isinstance(selected_location, str) and selected_location.strip():
                location_query = selected_location.strip()
                st.caption(f"Using typed location: {location_query}")
        else:
            location_query = st.text_input("City / location", value="Ahmedabad")
            st.caption("Install `streamlit-searchbox` for live typeahead suggestions.")

        use_live_weather = st.checkbox("Use live weather API for future predictions", value=True)
        assume_last_weather = st.checkbox("Assume last observed weather for future", value=True) if use_weather else False

    return user_panel_count, selected_location, location_query, use_live_weather, assume_last_weather, estimated_ref_panels


def render_project_intro() -> None:
    st.markdown(
        """
        This project, <strong>solarcast</strong>, is an end-to-end solar energy forecasting dashboard that combines
        historical plant generation records, weather telemetry, and machine learning to estimate expected power output.
        It allows users to scale predictions to their own panel count, select any location for weather-aware forecasting,
        and analyze both short-horizon trends and day-level generation profiles. The system is designed to support
        practical decision-making for plant operators, installers, and individual system owners by turning raw
        generation and weather data into clear, actionable forecasts through an interactive interface.
        """,
        unsafe_allow_html=True,
    )


def render_portfolio_summary(plant: str, target: str, user_panel_count: int, panel_scale: float) -> None:
    st.markdown("<div class='section-title'>Portfolio Setup Summary</div>", unsafe_allow_html=True)
    summary_cols = st.columns(4)
    scale_display = f"{panel_scale:.6f}x" if panel_scale < 0.01 else f"{panel_scale:.3f}x"
    summary_cols[0].metric("Plant", plant)
    summary_cols[1].metric("Target", target)
    summary_cols[2].metric("Your Panels", f"{user_panel_count:,}")
    summary_cols[3].metric("Scale Factor", scale_display)


def resolve_live_weather(
    use_weather: bool,
    use_live_weather: bool,
    selected_location: object,
    location_query: str,
    forecast_frame: pd.DataFrame,
) -> tuple[dict | None, pd.DataFrame]:
    geo = None
    live_weather_df = pd.DataFrame(columns=["DATE_TIME"] + WEATHER_COLUMNS)
    has_location_input = (
        (isinstance(selected_location, dict))
        or (isinstance(selected_location, str) and selected_location.strip())
        or location_query.strip()
    )
    if not (use_weather and use_live_weather and has_location_input):
        return geo, live_weather_df

    try:
        geo = resolve_geo_from_label_or_query(selected_location, location_query)
        if geo is None:
            st.warning("Could not geocode the location. Falling back to dataset weather assumptions.")
            return None, live_weather_df

        live_weather_df = fetch_live_weather(
            latitude=geo["latitude"],
            longitude=geo["longitude"],
            timezone=geo["timezone"],
            start_date=forecast_frame["DATE_TIME"].min().normalize(),
            end_date=forecast_frame["DATE_TIME"].max().normalize(),
        )
    except Exception as exc:
        st.warning(f"Live weather fetch failed. Falling back to dataset weather assumptions. ({exc})")
    return geo, live_weather_df


def render_weather_status(use_weather: bool, use_live_weather: bool, geo: dict | None, live_weather_df: pd.DataFrame) -> None:
    status_chips: list[str] = []
    if use_weather:
        status_chips.append("Weather Features: ON")
        if use_live_weather and geo is not None and not live_weather_df.empty:
            status_chips.append(f"Live Weather: {format_location_label(geo)}")
        elif use_live_weather:
            status_chips.append("Live Weather: fallback profile")
        else:
            status_chips.append("Live Weather: OFF")
    else:
        status_chips.append("Weather Features: OFF")
    st.markdown("".join([f"<span class='status-chip'>{chip}</span>" for chip in status_chips]), unsafe_allow_html=True)


def apply_weather_to_frame(
    frame: pd.DataFrame,
    use_weather: bool,
    use_live_weather: bool,
    assume_last_weather: bool,
    live_weather_df: pd.DataFrame,
    merged: pd.DataFrame,
    feature_means: pd.Series,
) -> pd.DataFrame:
    if not use_weather:
        return frame

    if use_live_weather and not live_weather_df.empty:
        return inject_weather_features(frame, live_weather_df, feature_means)
    if assume_last_weather:
        return inject_typical_hourly_weather(frame, merged[["DATE_TIME"] + WEATHER_COLUMNS], feature_means)

    for col in WEATHER_COLUMNS:
        frame[col] = np.nan
    frame[WEATHER_COLUMNS] = frame[WEATHER_COLUMNS].fillna(feature_means[WEATHER_COLUMNS])
    return frame


def render_dashboard_tabs(
    merged: pd.DataFrame,
    target: str,
    panel_scale: float,
    forecast_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds: np.ndarray,
    metrics: dict[str, float],
    day_pred_df: pd.DataFrame,
    forecast_day: pd.Timestamp,
    unit: str,
    predicted_total: float,
) -> None:
    tabs = st.tabs(["Overview", "Model Quality", "Forecast", "Daily Planner"])

    with tabs[0]:
        st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rows", f"{len(merged):,}")
        col_b.metric("Start", merged["DATE_TIME"].min().strftime("%Y-%m-%d %H:%M"))
        col_c.metric("End", merged["DATE_TIME"].max().strftime("%Y-%m-%d %H:%M"))
        fig_hist = px.line(merged, x="DATE_TIME", y=target, color_discrete_sequence=["#1a7bb9"])
        apply_chart_style(fig_hist, f"{target} Historical Trend", TARGET_UNITS.get(target, "units"))
        st.plotly_chart(fig_hist, use_container_width=True)
        with st.expander("Preview merged dataset"):
            st.dataframe(merged.head(25), use_container_width=True)

    with tabs[1]:
        st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
        perf_cols = st.columns(3)
        perf_cols[0].metric("MAE", f"{metrics['MAE']:.2f}")
        perf_cols[1].metric("RMSE", f"{metrics['RMSE']:.2f}")
        perf_cols[2].metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
        eval_fig = go.Figure()
        eval_fig.add_trace(
            go.Scatter(
                x=test_df["DATE_TIME"],
                y=test_df[target],
                mode="lines",
                name="Actual",
                line=dict(color="#1a7bb9"),
            )
        )
        eval_fig.add_trace(
            go.Scatter(
                x=test_df["DATE_TIME"],
                y=preds,
                mode="lines",
                name="Predicted",
                line=dict(color="#f28f3b"),
            )
        )
        apply_chart_style(eval_fig, "Validation: Actual vs Predicted", TARGET_UNITS.get(target, "units"))
        st.plotly_chart(eval_fig, use_container_width=True)

    with tabs[2]:
        st.markdown("<div class='section-title'>Forward Forecast</div>", unsafe_allow_html=True)
        forecast_fig = go.Figure()
        forecast_fig.add_trace(
            go.Scatter(
                x=merged["DATE_TIME"],
                y=merged[target] * panel_scale,
                mode="lines",
                name="History (scaled)",
                line=dict(color="#1a7bb9"),
            )
        )
        forecast_fig.add_trace(
            go.Scatter(
                x=forecast_df["DATE_TIME"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast (scaled)",
                line=dict(color="#f28f3b"),
            )
        )
        apply_chart_style(
            forecast_fig,
            "Forecast (Scaled to Your Panel Count)",
            TARGET_UNITS.get(target, "units"),
        )
        st.plotly_chart(forecast_fig, use_container_width=True)

    with tabs[3]:
        st.markdown("<div class='section-title'>Daily Energy Planner</div>", unsafe_allow_html=True)
        top_cols = st.columns([1, 1, 2])
        top_cols[0].metric(f"Predicted total ({unit})", f"{predicted_total:,.2f}")
        top_cols[1].metric("Date", forecast_day.strftime("%Y-%m-%d"))
        top_cols[2].caption(
            "Profile uses the selected location weather when available. "
            "If unavailable, the app falls back to learned hourly weather behavior."
        )

        day_fig = go.Figure()
        if not day_pred_df.empty:
            day_fig.add_trace(
                go.Scatter(
                    x=day_pred_df["DATE_TIME"],
                    y=day_pred_df["Prediction"],
                    mode="lines",
                    name="Predicted",
                    line=dict(color="#f28f3b"),
                )
            )
        actual_mask = merged["DATE_TIME"].dt.date == forecast_day.date()
        if actual_mask.any():
            actual_day = merged.loc[actual_mask, ["DATE_TIME", target]]
            day_fig.add_trace(
                go.Scatter(
                    x=actual_day["DATE_TIME"],
                    y=actual_day[target] * panel_scale,
                    mode="lines",
                    name="Actual (scaled)",
                    line=dict(color="#1a7bb9"),
                )
            )
        apply_chart_style(day_fig, "Daily Prediction Profile", unit)
        st.plotly_chart(day_fig, use_container_width=True)


def main():
    apply_global_styles()
    render_app_header()
    plant, target, use_weather, test_size, horizon, forecast_day = get_primary_controls()

    gen_path = DATA_DIR / GEN_FILES[plant]
    weather_path = DATA_DIR / WEATHER_FILES[plant]

    if not gen_path.exists() or not weather_path.exists():
        st.error("CSV files not found in the project directory.")
        st.stop()

    gen, weather = load_data(gen_path, weather_path)
    merged = add_time_features(prepare_data(gen, weather))
    freq_delta = freq_to_timedelta(infer_freq(merged["DATE_TIME"]), merged["DATE_TIME"].max())
    (
        user_panel_count,
        selected_location,
        location_query,
        use_live_weather,
        assume_last_weather,
        estimated_ref_panels,
    ) = get_setup_controls(merged, use_weather)
    panel_scale = user_panel_count / max(estimated_ref_panels, 1)

    render_project_intro()
    render_portfolio_summary(plant, target, user_panel_count, panel_scale)
    with st.spinner("Training model and preparing forecasts..."):
        model, feature_cols, feature_means, test_df, preds, metrics = train_model(
            merged, target, use_weather, test_size
        )

    st.subheader("Forecast")
    future = make_future_frame(merged, horizon=horizon, freq=freq_delta)
    geo, live_weather_df = resolve_live_weather(
        use_weather, use_live_weather, selected_location, location_query, future
    )
    render_weather_status(use_weather, use_live_weather, geo, live_weather_df)
    future = apply_weather_to_frame(
        future, use_weather, use_live_weather, assume_last_weather, live_weather_df, merged, feature_means
    )

    future[feature_cols] = future[feature_cols].fillna(feature_means)
    forecast = model.predict(future[feature_cols]) * panel_scale

    forecast_df = pd.DataFrame(
        {"DATE_TIME": future["DATE_TIME"], "Forecast": forecast}
    )

    st.subheader("Predict Energy For a Date")
    day_frame = make_day_frame(pd.Timestamp(forecast_day), freq=freq_delta)

    day_weather_df = pd.DataFrame(columns=["DATE_TIME"] + WEATHER_COLUMNS)
    if use_weather and use_live_weather and geo is not None:
        try:
            day_weather_df = fetch_live_weather(
                latitude=geo["latitude"],
                longitude=geo["longitude"],
                timezone=geo["timezone"],
                start_date=day_frame["DATE_TIME"].min().normalize(),
                end_date=day_frame["DATE_TIME"].max().normalize(),
            )
        except Exception:
            day_weather_df = pd.DataFrame(columns=["DATE_TIME"] + WEATHER_COLUMNS)

    day_frame = apply_weather_to_frame(
        day_frame, use_weather, use_live_weather, assume_last_weather, day_weather_df, merged, feature_means
    )

    day_frame[feature_cols] = day_frame[feature_cols].fillna(feature_means)
    day_pred = model.predict(day_frame[feature_cols]) * panel_scale
    day_pred_df = pd.DataFrame(
        {"DATE_TIME": day_frame["DATE_TIME"], "Prediction": day_pred}
    )

    day_pred_df["Prediction"] = pd.to_numeric(day_pred_df["Prediction"], errors="coerce")
    predicted_total = float(np.nansum(day_pred_df["Prediction"].values))
    if day_pred_df["Prediction"].notna().sum() == 0:
        st.warning("Prediction values could not be computed for this date. Try another date or disable live weather.")

    unit = TARGET_UNITS.get(target, "units")
    render_dashboard_tabs(
        merged=merged,
        target=target,
        panel_scale=panel_scale,
        forecast_df=forecast_df,
        test_df=test_df,
        preds=preds,
        metrics=metrics,
        day_pred_df=day_pred_df,
        forecast_day=forecast_day,
        unit=unit,
        predicted_total=predicted_total,
    )


if __name__ == "__main__":
    main()
