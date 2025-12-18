import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, START_YEAR, END_YEAR, MIN_POPULARITY, RANDOM_SEED, TARGET_COL, FEATURE_COLS, TEST_SIZE

def load_data():
    """Load tracks data from CSV."""
    file_path = os.path.join(DATA_DIR, 'tracks.csv')
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Filter and clean data according to project requirements.
    """
    initial_count = len(df)
    
    # Year filter - handle different release_date formats
    df = df.copy()
    df['release_date'] = df['release_date'].astype(str)
    df = df[df['release_date'].str.len() >= 4]
    df['year'] = df['release_date'].str[:4].astype(int)
    
    # Filter years
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    
    # Popularity filter
    df = df[df['popularity'] > MIN_POPULARITY]
    
    # Drop NAs if any in critical columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    
    print(f"Data cleaning: {initial_count} -> {len(df)} tracks selected.")
    return df

def get_train_test_data(include_metadata=False):
    """Load, clean, and split data into train and test sets."""
    df = load_data()
    df = clean_data(df)
    
    # Select columns
    if include_metadata:
        cols = FEATURE_COLS + [TARGET_COL, 'name', 'artists', 'id']
    else:
        cols = FEATURE_COLS + [TARGET_COL]
        
    df_model = df[cols].copy()
    
    # Split data
    train_df, test_df = train_test_split(
        df_model, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df
