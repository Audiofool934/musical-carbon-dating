import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .config import FEATURE_COLS, BREAK_YEAR
from .visualization import save_plot, set_plot_style, ANALOG_COLOR, DIGITAL_COLOR

def run_eda(df):
    """
    Execute comprehensive Exploratory Data Analysis.
    1. Feature Distributions
    2. Correlation Heatmap
    3. Time Series Trends (Mean values over years)
    """
    print("--- Starting Comprehensive EDA ---")
    set_plot_style()
    
    # 1. Distributions
    print("Generating Feature Distributions...")
    df_features = df[FEATURE_COLS]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, col in enumerate(FEATURE_COLS):
        if i < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[i], color=ANALOG_COLOR, edgecolor="white", linewidth=0.3)
            axes[i].set_title(f'Dist: {col}')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].axis("off")
            
    plt.tight_layout()
    save_plot(fig, "eda_distributions.png")
    
    # 2. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    corr = df[FEATURE_COLS + ['year']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap=sns.diverging_palette(30, 200, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap")
    save_plot(plt.gcf(), "eda_correlation_heatmap.png")
    
    # 3. Time Series Trends
    print("Generating Time Series Trends...")
    # Group by year and calculate mean
    yearly_avg = df.groupby('year')[FEATURE_COLS].mean().reset_index()
    
    # Let's do a grid of line plots for all features
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(FEATURE_COLS):
        if i < len(axes):
            sns.lineplot(data=yearly_avg, x='year', y=col, ax=axes[i], linewidth=2, color=ANALOG_COLOR)
            axes[i].axvline(BREAK_YEAR, color=DIGITAL_COLOR, linestyle='--', linewidth=1)
            axes[i].set_title(f'Trend: {col}')
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].axis("off")
            
    plt.tight_layout()
    save_plot(fig, "eda_trends_over_time.png")
