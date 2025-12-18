import sys
import os

# Add parent directory to path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_train_test_data
from src.config import TARGET_COL, FEATURE_COLS, RANDOM_SEED
import statsmodels.api as sm
import pandas as pd
import numpy as np

def check_nostalgia_index():
    print("--- Nostalgia Index Validation ---")
    
    # 1. Load data with metadata using refactored loader
    train_df, test_df = get_train_test_data(include_metadata=True)
    
    # Combine for full model fitting if desired, or just use train_df
    # For "Carbon Dating", we train the model on historical data.
    # Let's fit the model on the full cleaned dataset for maximum precision in this demo.
    full_df = pd.concat([train_df, test_df])
    
    X_train = sm.add_constant(full_df[FEATURE_COLS])
    y_train = full_df[TARGET_COL]
    
    print("Fitting baseline MLR model for Carbon Dating...")
    model = sm.OLS(y_train, X_train).fit()
    
    # 2. Select modern tracks for "Nostalgia" check
    # We look for popular tracks from 2010 onwards
    retro_candidates = full_df[(full_df['year'] >= 2010) & (full_df['popularity'] >= 50)].copy()
    
    if retro_candidates.empty:
        print("No modern candidates found.")
        return

    print(f"Analyzing {len(retro_candidates)} modern tracks for retro-production styles...")
    
    # 3. Predict year based on audio features (Blind Prediction)
    X_cand = sm.add_constant(retro_candidates[FEATURE_COLS])
    preds = model.predict(X_cand)
    
    retro_candidates['predicted_year'] = preds
    retro_candidates['nostalgia_score'] = retro_candidates['predicted_year'] - retro_candidates['year']
    
    # 4. Target Specific Hits from Report
    target_hits = [
        ('Echoes Of Silence', 'The Weeknd'),
        ('Uptown Funk', 'Mark Ronson'),
        ('Physical', 'Dua Lipa')
    ]
    
    print("\n--- Verifying Specific Hits mentioned in Report ---")
    for song, artist in target_hits:
        match = full_df[
            (full_df['name'].str.contains(song, case=False, na=False)) & 
            (full_df['artists'].str.contains(artist, case=False, na=False))
        ]
        if not match.empty:
            row = match.iloc[0]
            X_hit = sm.add_constant(pd.DataFrame([row[FEATURE_COLS]]), has_constant='add')
            # Align columns with model
            X_hit = X_hit[X_train.columns]
            pred = model.predict(X_hit).iloc[0]
            score = pred - row['year']
            print(f"Song: {row['name']} | Actual: {row['year']} | Pred: {pred:.1f} | Score: {score:.1f}")
        else:
            print(f"Song not found: {song} by {artist}")

    # 5. Show general top nostalgic tracks
    print("\n--- Top 10 'Nostalgic' Tracks (Modern songs sounding old) ---")
    # Low score means Predicted Year < Actual Year
    retro_modern = retro_candidates.sort_values('nostalgia_score')
    
    # Formatting for output
    cols_to_show = ['name', 'artists', 'year', 'predicted_year', 'nostalgia_score']
    print(retro_modern[cols_to_show].head(10).to_string(index=False, formatters={
        'predicted_year': '{:,.1f}'.format,
        'nostalgia_score': '{:,.1f}'.format
    }))

if __name__ == "__main__":
    check_nostalgia_index()
