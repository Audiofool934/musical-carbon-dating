import pandas as pd
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
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Filter years
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    
    # Popularity filter
    df = df[df['popularity'] > MIN_POPULARITY]
    
    # Drop NAs if any in critical columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    
    print(f"Data cleaning: {initial_count} -> {len(df)} tracks selected.")
    return df

def get_train_test_data(include_metadata=False, split_method='random', test_years=5):
    """
    Load, clean, and split data into train and test sets.
    
    Args:
        include_metadata: Whether to include track metadata columns
        split_method: 'temporal' (time-based) or 'random' (random split)
        test_years: Number of recent years to use as test set (for temporal split)
    
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    df = load_data()
    df = clean_data(df)
    
    # Select columns
    if include_metadata:
        cols = FEATURE_COLS + [TARGET_COL, 'name', 'artists', 'id']
    else:
        cols = FEATURE_COLS + [TARGET_COL]
        
    df_model = df[cols].copy()
    
    # Split data based on method
    if split_method == 'temporal':
        # Time-based split: use recent years for test set
        threshold_year = END_YEAR - test_years
        train_df = df_model[df_model[TARGET_COL] <= threshold_year].copy()
        test_df = df_model[df_model[TARGET_COL] > threshold_year].copy()
        print(f"Temporal split: Train (≤{threshold_year}), Test (>{threshold_year})")
    elif split_method == 'random':
        # Random split (legacy method)
        train_df, test_df = train_test_split(
            df_model, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_SEED
        )
        print("Random split (legacy mode)")
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Use 'temporal' or 'random'.")
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Train year range: {train_df[TARGET_COL].min():.0f} - {train_df[TARGET_COL].max():.0f}")
    print(f"Test year range: {test_df[TARGET_COL].min():.0f} - {test_df[TARGET_COL].max():.0f}")
    
    return train_df, test_df
