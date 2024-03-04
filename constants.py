import logging
import sys
from pathlib import Path

# File paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models/model.joblib"
TRAIN_DATA_PATH = PROJECT_ROOT / "data/housing.csv"

# Database constants
DATABASE_NAME = 'housing_data.db'

# Randomization
RANDOM_STATE = 100
TEST_SIZE = 0.2

# Columns
FILTERED_COLUMNS = [
    'LONGITUDE', 'LAT', 'MEDIAN_AGE', 'ROOMS', 'BEDROOMS', 'POP',
    'HOUSEHOLDS', 'MEDIAN_INCOME', 'OCEAN_PROXIMITY_<1H OCEAN',
    'OCEAN_PROXIMITY_INLAND', 'OCEAN_PROXIMITY_ISLAND',
    'OCEAN_PROXIMITY_NEAR BAY', 'OCEAN_PROXIMITY_NEAR OCEAN'
]

# Column mapping
COLUMN_MAPPING = {
    'LONGITUDE': 'longitude',
    'LAT': 'latitude',
    'MEDIAN_AGE': 'housing_median_age',
    'ROOMS': 'total_rooms',
    'BEDROOMS': 'total_bedrooms',
    'POP': 'population',
    'HOUSEHOLDS': 'households',
    'MEDIAN_INCOME': 'median_income',
    'OCEAN_PROXIMITY_<1H OCEAN': 'ocean_proximity_<1H OCEAN',
    'OCEAN_PROXIMITY_INLAND': 'ocean_proximity_INLAND',
    'OCEAN_PROXIMITY_ISLAND': 'ocean_proximity_ISLAND',
    'OCEAN_PROXIMITY_NEAR BAY': 'ocean_proximity_NEAR BAY',
    'OCEAN_PROXIMITY_NEAR OCEAN': 'ocean_proximity_NEAR OCEAN'
}

# Columns to drop
DROP_COLUMN = ['MEDIAN_HOUSE_VALUE', 'AGENCY']

# Expected columns
EXPECTED_COLUMN = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND',
    'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY',
    'ocean_proximity_NEAR OCEAN'
]

# Set up logging configuration
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "logs.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
