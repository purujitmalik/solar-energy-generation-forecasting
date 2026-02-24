"""Streamlit dashboard for solar generation forecasting."""

import pandas
import plotly.express
import plotly.graph_objects
import streamlit

from project_settings import (
    DEFAULT_DAYLIGHT_HOUR_RANGE,
    DEFAULT_LOCATION_QUERY,
    GENERATION_FILES_BY_PLANT,
    PROJECT_DIR,
    TARGET_COLUMN_OPTIONS,
    WEATHER_FILES_BY_PLANT,
)
from data_processing import (
    add_calendar_features,
    build_date_range_timeframe,
    build_future_timeframe,
    build_lag_settings,
    build_single_day_timeframe,
    convert_frequency_to_timedelta,
    infer_time_frequency,
    load_csv_data,
    merge_generation_and_weather,
)
from forecasting import cap_predictions_by_daily_history, clamp_to_historical_range, predict_step_by_step
from model_training import evaluate_previous_day_baseline, get_time_series_cv_metrics, train_forecast_model
from weather_forecast import fetch_live_weather_forecast, merge_weather_into_timeframe, search_location_options

streamlit.set_page_config(page_title="Solar Generation Forecasting", layout="wide")

# Keep range predictions practical for the UI and runtime.
MAX_RANGE_DAYS = 31


def apply_nighttime_zeroing(
    prediction_data: pandas.DataFrame,
    reference_timeframe: pandas.DataFrame,
    value_column: str,
    use_weather_for_inference: bool,
) -> pandas.DataFrame:
    """Set predictions to 0 during night hours."""
    adjusted_predictions = prediction_data.copy()
    # If irradiation is available, use it directly to detect night.
    if use_weather_for_inference and "IRRADIATION" in reference_timeframe.columns:
        nighttime_mask = reference_timeframe["IRRADIATION"] <= 0
    else:
        # Otherwise, use fixed daylight hours as a fallback rule.
        daylight_start_hour, daylight_end_hour = DEFAULT_DAYLIGHT_HOUR_RANGE
        nighttime_mask = (reference_timeframe["hour"] < daylight_start_hour) | (
            reference_timeframe["hour"] >= daylight_end_hour
        )
    adjusted_predictions.loc[nighttime_mask, value_column] = 0.0
    return adjusted_predictions


def main():
    """Render UI, train model, and show forecast outputs."""
    streamlit.title("Solar Energy Generation Forecasting")
    streamlit.caption("Plant 1 forecasting dashboard with generation + weather data.")

    with streamlit.sidebar:
        # User controls: plant, target, train/test split, forecast windows.
        streamlit.header("Controls")
        selected_plant = streamlit.selectbox("Plant", options=list(GENERATION_FILES_BY_PLANT.keys()), index=0)
        selected_target_column = streamlit.selectbox("Target", options=TARGET_COLUMN_OPTIONS, index=0)
        use_weather_features = streamlit.checkbox("Use weather features", value=True)
        test_fraction = streamlit.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        forecast_horizon_steps = streamlit.slider(
            "Forecast horizon (steps)",
            min_value=12,
            max_value=288,
            value=96,
            step=12,
        )
        selected_forecast_day = streamlit.date_input("Predict energy for date", value=None)
        streamlit.markdown("**Predict energy for date range**")
        selected_range_start_day = streamlit.date_input("Range start", value=None, key="range_start")
        selected_range_end_day = streamlit.date_input("Range end", value=None, key="range_end")
        assume_last_weather = False

        if use_weather_features:
            # Optional: fetch live weather for a selected location.
            streamlit.markdown("**Location for live forecast**")
            location_search_text = streamlit.text_input("Place name", value=DEFAULT_LOCATION_QUERY)
            if len(location_search_text.strip()) >= 2:
                location_options = search_location_options(location_search_text, count=5)
            else:
                location_options = []

            location_option_labels = [
                f"{location['name']}, {location.get('admin1','')}, {location.get('country','')}"
                .replace(" ,", ",")
                .strip()
                for location in location_options
            ]

            selected_location = None
            if location_options:
                selected_location_label = streamlit.selectbox("Choose place", options=location_option_labels, index=0)
                selected_location_index = location_option_labels.index(selected_location_label)
                selected_location = location_options[selected_location_index]
            else:
                streamlit.info("Type at least 2 characters to get place suggestions.")
            timezone = streamlit.text_input("Timezone (IANA or 'auto')", value="auto")
        else:
            selected_location = None
            timezone = "auto"

    generation_csv_path = PROJECT_DIR / GENERATION_FILES_BY_PLANT[selected_plant]
    weather_csv_path = PROJECT_DIR / WEATHER_FILES_BY_PLANT[selected_plant]

    # Stop early if required files are missing.
    if not generation_csv_path.exists() or not weather_csv_path.exists():
        streamlit.error("CSV files not found in the project directory.")
        streamlit.stop()

    # Prepare one clean dataset and derive feature settings.
    generation_data, weather_data = load_csv_data(generation_csv_path, weather_csv_path)
    merged_data = add_calendar_features(merge_generation_and_weather(generation_data, weather_data))
    detected_frequency = infer_time_frequency(merged_data["DATE_TIME"])
    frequency_timedelta = convert_frequency_to_timedelta(detected_frequency, merged_data["DATE_TIME"].max())
    lag_steps, rolling_window_steps = build_lag_settings(frequency_timedelta)

    streamlit.subheader("Dataset Overview")
    overview_column_1, overview_column_2, overview_column_3 = streamlit.columns(3)
    overview_column_1.metric("Rows", f"{len(merged_data):,}")
    overview_column_2.metric("Start", merged_data["DATE_TIME"].min().strftime("%Y-%m-%d %H:%M"))
    overview_column_3.metric("End", merged_data["DATE_TIME"].max().strftime("%Y-%m-%d %H:%M"))
    streamlit.dataframe(merged_data.head(10), use_container_width=True)

    streamlit.subheader("Generation History")
    generation_history_chart = plotly.express.line(
        merged_data,
        x="DATE_TIME",
        y=selected_target_column,
        title=f"{selected_target_column} over time",
    )
    streamlit.plotly_chart(generation_history_chart, use_container_width=True)

    streamlit.subheader("Model Training & Evaluation")
    with streamlit.spinner("Training model..."):
        # Train once and evaluate on the most recent split of data.
        (
            trained_model,
            feature_column_names,
            feature_column_means,
            test_data,
            test_prediction_values,
            test_metrics,
        ) = train_forecast_model(
            merged_data,
            selected_target_column,
            use_weather_features,
            test_fraction,
            lag_steps,
            rolling_window_steps,
        )

    metric_column_1, metric_column_2, metric_column_3 = streamlit.columns(3)
    metric_column_1.metric("MAE", f"{test_metrics['MAE']:.2f}")
    metric_column_2.metric("RMSE", f"{test_metrics['RMSE']:.2f}")
    metric_column_3.metric("MAPE", f"{test_metrics['MAPE (%)']:.2f}%")

    cross_validation_metrics = get_time_series_cv_metrics(
        merged_data,
        selected_target_column,
        feature_column_names,
        n_splits=3,
        test_fraction=test_fraction,
    )
    if cross_validation_metrics is not None:
        # Cross-validation gives a more stable estimate than a single split.
        streamlit.caption(
            f"Rolling CV (avg): MAE {cross_validation_metrics['MAE']:.2f}, "
            f"RMSE {cross_validation_metrics['RMSE']:.2f}, "
            f"MAPE {cross_validation_metrics['MAPE (%)']:.2f}%"
        )

    test_evaluation_chart = plotly.graph_objects.Figure()
    test_evaluation_chart.add_trace(
        plotly.graph_objects.Scatter(
            x=test_data["DATE_TIME"],
            y=test_data[selected_target_column],
            mode="lines",
            name="Actual",
        )
    )
    test_evaluation_chart.add_trace(
        plotly.graph_objects.Scatter(
            x=test_data["DATE_TIME"],
            y=test_prediction_values,
            mode="lines",
            name="Predicted",
        )
    )
    test_evaluation_chart.update_layout(title="Test Split: Actual vs Predicted", xaxis_title="Time")
    streamlit.plotly_chart(test_evaluation_chart, use_container_width=True)

    baseline_metrics, baseline_prediction_values = evaluate_previous_day_baseline(
        merged_data,
        test_data,
        selected_target_column,
        frequency_timedelta,
    )
    if baseline_metrics is not None:
        # Baseline = same time from previous day. Good sanity check.
        streamlit.subheader("Baseline Comparison (Same Time Previous Day)")
        baseline_metric_column_1, baseline_metric_column_2, baseline_metric_column_3 = streamlit.columns(3)
        baseline_metric_column_1.metric("Baseline MAE", f"{baseline_metrics['MAE']:.2f}")
        baseline_metric_column_2.metric("Baseline RMSE", f"{baseline_metrics['RMSE']:.2f}")
        baseline_metric_column_3.metric("Baseline MAPE", f"{baseline_metrics['MAPE (%)']:.2f}%")

        baseline_chart = plotly.graph_objects.Figure()
        baseline_chart.add_trace(
            plotly.graph_objects.Scatter(
                x=test_data["DATE_TIME"],
                y=test_data[selected_target_column],
                mode="lines",
                name="Actual",
            )
        )
        baseline_chart.add_trace(
            plotly.graph_objects.Scatter(
                x=test_data["DATE_TIME"],
                y=baseline_prediction_values,
                mode="lines",
                name="Baseline",
            )
        )
        baseline_chart.update_layout(title="Baseline vs Actual (Test Split)", xaxis_title="Time")
        streamlit.plotly_chart(baseline_chart, use_container_width=True)

    streamlit.subheader("Forecast")
    future_timeframe = build_future_timeframe(
        merged_data,
        forecast_steps=forecast_horizon_steps,
        frequency_value=detected_frequency,
    )

    weather_forecast_data = None
    use_weather_for_inference = use_weather_features
    if use_weather_features:
        if selected_location is None:
            streamlit.warning("Could not resolve location. Continuing with fallback weather features.")
            use_weather_for_inference = False
        else:
            selected_latitude = float(selected_location["latitude"])
            selected_longitude = float(selected_location["longitude"])
            if timezone == "auto" and selected_location.get("timezone"):
                timezone = selected_location["timezone"]
            streamlit.caption(
                f"Using location: {selected_location['name']}, {selected_location.get('country','')} "
                f"({selected_latitude:.4f}, {selected_longitude:.4f})"
            )

            weather_forecast_data = fetch_live_weather_forecast(selected_latitude, selected_longitude, timezone)
            if weather_forecast_data is None or weather_forecast_data.empty:
                streamlit.warning("Live weather forecast unavailable. Continuing with fallback weather features.")
                use_weather_for_inference = False
                weather_forecast_data = None

            if selected_forecast_day is not None:
                selected_day_start_datetime = pandas.Timestamp(selected_forecast_day).normalize()
                selected_day_end_datetime = selected_day_start_datetime + pandas.Timedelta(days=1)
            else:
                selected_day_start_datetime = None
                selected_day_end_datetime = None

            if selected_range_start_day is not None and selected_range_end_day is not None:
                selected_range_start_datetime = pandas.Timestamp(selected_range_start_day).normalize()
                selected_range_end_datetime = pandas.Timestamp(selected_range_end_day).normalize() + pandas.Timedelta(days=1)
            else:
                selected_range_start_datetime = None
                selected_range_end_datetime = None

            minimum_needed_datetime = future_timeframe["DATE_TIME"].min()
            maximum_needed_datetime = future_timeframe["DATE_TIME"].max()
            if selected_day_start_datetime is not None and selected_day_start_datetime < minimum_needed_datetime:
                minimum_needed_datetime = selected_day_start_datetime
            if selected_day_end_datetime is not None and selected_day_end_datetime > maximum_needed_datetime:
                maximum_needed_datetime = selected_day_end_datetime
            if selected_range_start_datetime is not None and selected_range_start_datetime < minimum_needed_datetime:
                minimum_needed_datetime = selected_range_start_datetime
            if selected_range_end_datetime is not None and selected_range_end_datetime > maximum_needed_datetime:
                maximum_needed_datetime = selected_range_end_datetime

            # Keep only the weather rows required for requested forecast windows.
            weather_forecast_data = weather_forecast_data[
                (weather_forecast_data["DATE_TIME"] >= minimum_needed_datetime)
                & (weather_forecast_data["DATE_TIME"] <= maximum_needed_datetime)
            ]
            if weather_forecast_data.empty:
                use_weather_for_inference = False
                weather_forecast_data = None

    future_timeframe = merge_weather_into_timeframe(
        future_timeframe,
        weather_forecast_data,
        use_weather=use_weather_features,
        assume_last_weather=assume_last_weather,
        historical_merged_data=merged_data,
        feature_means=feature_column_means,
    )

    with streamlit.spinner("Generating near-term forecast..."):
        # Autoregressive forecast: each predicted step is used in following steps.
        forecast_prediction_values, future_timeframe = predict_step_by_step(
            model=trained_model,
            history_data=merged_data,
            prediction_frame=future_timeframe,
            target_column=selected_target_column,
            feature_columns=feature_column_names,
            feature_means=feature_column_means,
            lags=lag_steps,
            rolling_windows=rolling_window_steps,
        )

    forecast_prediction_values = clamp_to_historical_range(
        forecast_prediction_values,
        merged_data,
        selected_target_column,
    )
    forecast_prediction_data = pandas.DataFrame(
        {
            "DATE_TIME": future_timeframe["DATE_TIME"],
            "Forecast": forecast_prediction_values,
        }
    )
    forecast_prediction_data, historical_daily_cap, capped_day_count = cap_predictions_by_daily_history(
        forecast_prediction_data,
        "Forecast",
        merged_data,
        selected_target_column,
    )
    if capped_day_count > 0:
        # Safety rule: do not allow daily totals above historical daily max.
        streamlit.caption(
            f"Capped {capped_day_count} forecast day(s) at historical daily max "
            f"({historical_daily_cap:,.2f})."
        )

    forecast_prediction_data = apply_nighttime_zeroing(
        forecast_prediction_data,
        future_timeframe,
        "Forecast",
        use_weather_for_inference,
    )

    forecast_chart = plotly.graph_objects.Figure()
    forecast_chart.add_trace(
        plotly.graph_objects.Scatter(
            x=merged_data["DATE_TIME"],
            y=merged_data[selected_target_column],
            mode="lines",
            name="History",
        )
    )
    forecast_chart.add_trace(
        plotly.graph_objects.Scatter(
            x=forecast_prediction_data["DATE_TIME"],
            y=forecast_prediction_data["Forecast"],
            mode="lines",
            name="Forecast",
        )
    )
    forecast_chart.update_layout(title="Forecast", xaxis_title="Time")
    streamlit.plotly_chart(forecast_chart, use_container_width=True)

    streamlit.subheader("Predict Energy For a Date")
    if selected_forecast_day is None:
        streamlit.info("Pick a date in the sidebar to get a full-day energy prediction.")
    else:
        # Build timestamps for selected day and run the same forecasting pipeline.
        selected_day_timeframe = build_single_day_timeframe(
            pandas.Timestamp(selected_forecast_day),
            frequency_value=detected_frequency,
        )
        selected_day_timeframe = merge_weather_into_timeframe(
            selected_day_timeframe,
            weather_forecast_data,
            use_weather=use_weather_features,
            assume_last_weather=assume_last_weather,
            historical_merged_data=merged_data,
            feature_means=feature_column_means,
        )

        with streamlit.spinner("Predicting selected day..."):
            selected_day_prediction_values, selected_day_timeframe = predict_step_by_step(
                model=trained_model,
                history_data=merged_data,
                prediction_frame=selected_day_timeframe,
                target_column=selected_target_column,
                feature_columns=feature_column_names,
                feature_means=feature_column_means,
                lags=lag_steps,
                rolling_windows=rolling_window_steps,
            )

        selected_day_prediction_values = clamp_to_historical_range(
            selected_day_prediction_values,
            merged_data,
            selected_target_column,
        )
        selected_day_prediction_data = pandas.DataFrame(
            {
                "DATE_TIME": selected_day_timeframe["DATE_TIME"],
                "Prediction": selected_day_prediction_values,
            }
        )
        selected_day_prediction_data, historical_daily_cap, selected_day_capped_count = (
            cap_predictions_by_daily_history(
                selected_day_prediction_data,
                "Prediction",
                merged_data,
                selected_target_column,
            )
        )
        if selected_day_capped_count > 0:
            streamlit.caption(f"Daily prediction capped at historical max ({historical_daily_cap:,.2f}).")

        selected_day_prediction_data = apply_nighttime_zeroing(
            selected_day_prediction_data,
            selected_day_timeframe,
            "Prediction",
            use_weather_for_inference,
        )

        selected_day_predicted_total = selected_day_prediction_data["Prediction"].sum()
        selected_day_actual_mask = (
            merged_data["DATE_TIME"].dt.date == pandas.Timestamp(selected_forecast_day).date()
        )
        if selected_day_actual_mask.any():
            selected_day_actual_total = merged_data.loc[
                selected_day_actual_mask,
                selected_target_column,
            ].sum()
        else:
            selected_day_actual_total = None

        selected_day_metric_column_1, selected_day_metric_column_2 = streamlit.columns(2)
        selected_day_metric_column_1.metric("Predicted total", f"{selected_day_predicted_total:,.2f}")
        selected_day_metric_column_2.metric(
            "Actual total",
            f"{selected_day_actual_total:,.2f}" if selected_day_actual_total is not None else "N/A",
        )

        selected_day_chart = plotly.graph_objects.Figure()
        selected_day_chart.add_trace(
            plotly.graph_objects.Scatter(
                x=selected_day_prediction_data["DATE_TIME"],
                y=selected_day_prediction_data["Prediction"],
                mode="lines",
                name="Predicted",
            )
        )
        if selected_day_actual_total is not None:
            selected_day_actual_data = merged_data.loc[
                selected_day_actual_mask,
                ["DATE_TIME", selected_target_column],
            ]
            selected_day_chart.add_trace(
                plotly.graph_objects.Scatter(
                    x=selected_day_actual_data["DATE_TIME"],
                    y=selected_day_actual_data[selected_target_column],
                    mode="lines",
                    name="Actual",
                )
            )
        selected_day_chart.update_layout(title="Daily prediction profile", xaxis_title="Time")
        streamlit.plotly_chart(selected_day_chart, use_container_width=True)

    streamlit.subheader("Predict Energy For a Date Range")
    if selected_range_start_day is None or selected_range_end_day is None:
        streamlit.info("Pick a start and end date in the sidebar to get a multi-day energy prediction.")
    else:
        # Build timestamps for date range and run step-by-step prediction.
        selected_range_start_timestamp = pandas.Timestamp(selected_range_start_day)
        selected_range_end_timestamp = pandas.Timestamp(selected_range_end_day)

        if selected_range_end_timestamp < selected_range_start_timestamp:
            streamlit.error("Range end must be on or after range start.")
        else:
            selected_range_day_count = int((selected_range_end_timestamp - selected_range_start_timestamp).days) + 1
            if selected_range_day_count > MAX_RANGE_DAYS:
                streamlit.error(
                    f"Range too large ({selected_range_day_count} days). "
                    f"Please select {MAX_RANGE_DAYS} days or fewer."
                )
                return

            selected_range_timeframe = build_date_range_timeframe(
                selected_range_start_timestamp,
                selected_range_end_timestamp,
                frequency_value=detected_frequency,
            )
            selected_range_timeframe = merge_weather_into_timeframe(
                selected_range_timeframe,
                weather_forecast_data,
                use_weather=use_weather_features,
                assume_last_weather=assume_last_weather,
                historical_merged_data=merged_data,
                feature_means=feature_column_means,
            )

            with streamlit.spinner("Predicting selected date range..."):
                selected_range_prediction_values, selected_range_timeframe = predict_step_by_step(
                    model=trained_model,
                    history_data=merged_data,
                    prediction_frame=selected_range_timeframe,
                    target_column=selected_target_column,
                    feature_columns=feature_column_names,
                    feature_means=feature_column_means,
                    lags=lag_steps,
                    rolling_windows=rolling_window_steps,
                )

            selected_range_prediction_values = clamp_to_historical_range(
                selected_range_prediction_values,
                merged_data,
                selected_target_column,
            )
            selected_range_prediction_data = pandas.DataFrame(
                {
                    "DATE_TIME": selected_range_timeframe["DATE_TIME"],
                    "Prediction": selected_range_prediction_values,
                }
            )
            selected_range_prediction_data, historical_daily_cap, selected_range_capped_day_count = (
                cap_predictions_by_daily_history(
                    selected_range_prediction_data,
                    "Prediction",
                    merged_data,
                    selected_target_column,
                )
            )
            if selected_range_capped_day_count > 0:
                streamlit.caption(
                    f"Capped {selected_range_capped_day_count} day(s) in range prediction at historical daily max "
                    f"({historical_daily_cap:,.2f})."
                )

            selected_range_prediction_data = apply_nighttime_zeroing(
                selected_range_prediction_data,
                selected_range_timeframe,
                "Prediction",
                use_weather_for_inference,
            )

            selected_range_predicted_total = selected_range_prediction_data["Prediction"].sum()
            selected_range_actual_mask = (
                (merged_data["DATE_TIME"].dt.date >= selected_range_start_timestamp.date())
                & (merged_data["DATE_TIME"].dt.date <= selected_range_end_timestamp.date())
            )
            if selected_range_actual_mask.any():
                selected_range_actual_total = merged_data.loc[
                    selected_range_actual_mask,
                    selected_target_column,
                ].sum()
            else:
                selected_range_actual_total = None

            selected_range_metric_column_1, selected_range_metric_column_2 = streamlit.columns(2)
            selected_range_metric_column_1.metric("Predicted total", f"{selected_range_predicted_total:,.2f}")
            selected_range_metric_column_2.metric(
                "Actual total",
                f"{selected_range_actual_total:,.2f}" if selected_range_actual_total is not None else "N/A",
            )

            selected_range_prediction_data["DATE"] = selected_range_prediction_data["DATE_TIME"].dt.date
            selected_range_daily_predictions = selected_range_prediction_data.groupby("DATE", as_index=False)[
                "Prediction"
            ].sum()

            selected_range_daily_chart = plotly.graph_objects.Figure()
            selected_range_daily_chart.add_trace(
                plotly.graph_objects.Bar(
                    x=selected_range_daily_predictions["DATE"],
                    y=selected_range_daily_predictions["Prediction"],
                    name="Predicted",
                )
            )
            if selected_range_actual_total is not None:
                selected_range_actual_daily_totals = (
                    merged_data.loc[selected_range_actual_mask, ["DATE_TIME", selected_target_column]]
                    .assign(DATE=lambda selected_rows: selected_rows["DATE_TIME"].dt.date)
                    .groupby("DATE", as_index=False)[selected_target_column]
                    .sum()
                )
                selected_range_daily_chart.add_trace(
                    plotly.graph_objects.Bar(
                        x=selected_range_actual_daily_totals["DATE"],
                        y=selected_range_actual_daily_totals[selected_target_column],
                        name="Actual",
                    )
                )
            selected_range_daily_chart.update_layout(title="Daily totals over range", xaxis_title="Date")
            streamlit.plotly_chart(selected_range_daily_chart, use_container_width=True)

    streamlit.caption(
        "Future forecasts use time features and live weather. "
        "Live Open-Meteo forecasts provide temperature and solar radiation. "
        "Module temperature is approximated from ambient temperature and irradiance."
    )

    streamlit.subheader("Results & Discussion")
    streamlit.write(
        "The model captures the overall daily shape of generation and produces reasonable "
        "short-term forecasts. Errors increase during rapid weather changes or near sunrise/sunset."
    )

    streamlit.subheader("Limitations & Future Work")
    streamlit.write(
        "This prototype uses live Open-Meteo forecasts for future predictions. "
        "Accuracy can improve by adding more sensor features, using higher-resolution weather data, "
        "and evaluating advanced time-series models. Data quality checks and retraining policies "
        "would also strengthen long-term reliability."
    )


if __name__ == "__main__":
    main()
