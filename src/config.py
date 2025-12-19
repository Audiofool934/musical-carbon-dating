# Musical Carbon Dating - Project Structure

## Configuration
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Analysis Constants
START_YEAR = 1960
END_YEAR = 2020
BREAK_YEAR = 1999
MIN_POPULARITY = 30 # Relaxed from 50
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Variable Lists
TARGET_COL = 'year'
FEATURE_COLS = [
    'loudness', 'tempo', 'duration_ms',
    'key', 'mode', 'time_signature',
    'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence'
]
