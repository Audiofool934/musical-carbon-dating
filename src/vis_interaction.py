
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.getcwd())
from src.config import DATA_DIR, FEATURE_COLS

def plot_interaction_scissor():
    """
    Visualizes the Structural Break (1999) by running two separate SLRs 
    for Acousticness vs Year (Pre-1999 vs Post-1999).
    This creates a 'Scissor' effect showing the slope reversal.
    """
    print("Generating Scissor Plot (Structural Break)...")
    
    # 1. Load Data
    DATA_PATH = os.path.join(DATA_DIR, 'tracks.csv')
    df = pd.read_csv(DATA_PATH)
    
    # Parse Year
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    # Basic Clean
    df = df.dropna(subset=['year', 'acousticness'])
    df['year'] = df['year'].astype(int)
    df = df[(df['year'] >= 1960) & (df['year'] <= 2020)]
    
    # 2. Split by Breakpoint
    BREAK_YEAR = 1999
    df_pre = df[df['year'] < BREAK_YEAR]
    df_post = df[df['year'] >= BREAK_YEAR]
    
    plt.figure(figsize=(10, 6))
    
    # 3. Scatter Plot (Sampled for visual clarity)
    sample_n = 5000
    if len(df) > sample_n:
        df_sample = df.sample(sample_n, random_state=42)
    else:
        df_sample = df
        
    # Color points by Era
    sns.scatterplot(
        data=df_sample, 
        x='acousticness', 
        y='year', 
        hue=df_sample['year'] >= BREAK_YEAR,
        palette={False: 'blue', True: 'orange'},
        alpha=0.1,
        legend=False
    )
    
    # 4. Regression Lines
    # Pre-1999
    sns.regplot(
        data=df_pre, x='acousticness', y='year', 
        scatter=False, color='blue', label='Pre-1999 Trend',
        line_kws={'linewidth': 3}
    )
    
    # Post-1999
    sns.regplot(
        data=df_post, x='acousticness', y='year', 
        scatter=False, color='orange', label='Post-1999 Trend',
        line_kws={'linewidth': 3}
    )
    
    plt.title(f'The "Scissor" Effect: Structural Break at {BREAK_YEAR}\n(Feature: Acousticness)', fontsize=14)
    plt.ylabel('Release Year')
    plt.xlabel('Acousticness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.savefig('output/figures/scissor_plot.png', dpi=300, bbox_inches='tight')
    print("Saved output/figures/scissor_plot.png")

if __name__ == "__main__":
    plot_interaction_scissor()
