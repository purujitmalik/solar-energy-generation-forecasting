"""Project-wide constants used across the Streamlit app."""

from pathlib import Path

# Folder where all project files (including CSV files) live.
PROJECT_DIR = Path(__file__).parent

# Input files for each solar plant.
GENERATION_FILES_BY_PLANT = {
    "Plant 1": "Plant_1_Generation_Data.csv",
    "Plant 2": "Plant_2_Generation_Data.csv",
}

WEATHER_FILES_BY_PLANT = {
    "Plant 1": "Plant_1_Weather_Sensor_Data.csv",
    "Plant 2": "Plant_2_Weather_Sensor_Data.csv",
}

# Common model options.
TARGET_COLUMN_OPTIONS = ["AC_POWER", "DC_POWER", "DAILY_YIELD"]
WEATHER_FEATURE_COLUMNS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
TIME_FEATURE_COLUMNS = ["hour", "dayofweek", "day", "month", "weekofyear", "is_weekend"]

# Constants used to approximate module temperature from weather forecast.
MODULE_TEMP_BASE_OFFSET = 5.0
MODULE_TEMP_RADIATION_COEFFICIENT = 0.03

# Default UI values.
DEFAULT_LOCATION_QUERY = "Pune, India"
DEFAULT_DAYLIGHT_HOUR_RANGE = (6, 18)
