import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .config import FIGURES_DIR, FEATURE_COLS
from .visualization import save_plot

def run_eda(df):
    """
    Execute comprehensive Exploratory Data Analysis.
    1. Feature Distributions
    2. Correlation Heatmap
    3. Time Series Trends (Mean values over years)
    """
    print("--- Starting Comprehensive EDA ---")
    
    # 1. Distributions
    print("Generating Feature Distributions...")
    df_features = df[FEATURE_COLS]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, col in enumerate(FEATURE_COLS):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Dist: {col}')
            
    plt.tight_layout()
    save_plot(fig, "eda_distributions.png")
    
    # 2. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    corr = df[FEATURE_COLS + ['year']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    save_plot(plt.gcf(), "eda_correlation_heatmap.png")
    
    # 3. Time Series Trends
    print("Generating Time Series Trends...")
    # Group by year and calculate mean
    yearly_avg = df.groupby('year')[FEATURE_COLS].mean().reset_index()
    
    # Plot key evolutionary features
    key_features = ['acousticness', 'danceability', 'energy', 'loudness', 'valence', 'speechiness']
    plt.figure(figsize=(14, 8))
    for feat in key_features:
        # Normalize for visualization comparison if needed, but raw values show real trend
        # Let's just plot raw with secondary axis if needed, or just multiple subplots
        # For simplicity/clarity, one plot with normalized 0-1 scaling might be best, OR subplots
        pass
        
    # Let's do a grid of line plots for all features
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(FEATURE_COLS):
        if i < len(axes):
            sns.lineplot(data=yearly_avg, x='year', y=col, ax=axes[i], linewidth=2)
            axes[i].set_title(f'Trend: {col}')
            
    plt.tight_layout()
    save_plot(fig, "eda_trends_over_time.png")
