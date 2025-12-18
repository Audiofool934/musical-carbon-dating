import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from .config import FIGURES_DIR

def save_plot(fig, name):
    """Helper to save plot to figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {path}")
    plt.close(fig)

def plot_scatter_matrix(df, cols, title="Scatter Matrix"):
    """Plot scatter matrix for selected columns."""
    print(f"Generating Scatter Matrix: {title}...")
    g = sns.pairplot(df[cols], diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
    g.fig.suptitle(title, y=1.02)
    save_plot(g.fig, f"{title.lower().replace(' ', '_')}.png")

def plot_residuals(model, title="Residuals Plots"):
    """Plot residuals vs fitted and Q-Q plot."""
    print(f"Generating Residual Plots: {title}...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Residuals vs Fitted
    sns.scatterplot(x=fitted, y=residuals, ax=axes[0], alpha=0.5)
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    
    # Q-Q Plot
    import statsmodels.api as sm
    sm.qqplot(residuals, line='45', ax=axes[1])
    axes[1].set_title('Normal Q-Q')
    
    save_plot(fig, f"residuals_{title.lower().replace(' ', '_')}.png")

def plot_influence(model, title="Influence Plot"):
    """
    Plot Studentized Residuals vs Leverage (Scalable version).
    """
    print(f"Generating Influence Plot: {title}...")
    
    # Get influence measures
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    studentized_residuals = influence.resid_studentized_internal
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Leverage': leverage,
        'Studentized_Residuals': studentized_residuals
    })
    
    # Sample if too large to avoid overcrowding/performance issues
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)
        print("Note: Influence plot sampled to 5000 points for performance.")
        
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='Leverage', y='Studentized_Residuals', alpha=0.5)
    
    # Add thresholds
    # High leverage usually > 2p/n or 3p/n
    p = len(model.params)
    n = model.nobs
    leverage_threshold = 3 * p / n
    plt.axvline(leverage_threshold, color='r', linestyle='--', label=f'Threshold (3p/n = {leverage_threshold:.4f})')
    plt.axhline(3, color='orange', linestyle='--', label='Residual > 3')
    plt.axhline(-3, color='orange', linestyle='--')
    
    plt.title(f"{title} (Studentized Residuals vs Leverage)")
    plt.xlabel('Leverage')
    plt.ylabel('Studentized Residuals')
    plt.legend()
    
    save_plot(plt.gcf(), "influence_plot.png")

def plot_coefficients(coefs, names, title="Regression Coefficients"):
    """Plot bar chart of coefficients."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefs, y=names)
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.axvline(0, color='k', linestyle='-')
    save_plot(plt.gcf(), f"coefs_{title.lower().replace(' ', '_')}.png")

def plot_partial_regression(model, exog_idx, title="Partial Regression Plot"):
    """
    Plot Partial Regression Plot (Added Variable Plot).
    Targeting specific variable against year controlling for others.
    """
    print(f"Generating Partial Regression Plot: {title}...")
    import statsmodels.api as sm
    fig = plt.figure(figsize=(12, 8))
    sm.graphics.plot_partregress_grid(model, fig=fig)
    save_plot(fig, f"partial_regression_{title.lower().replace(' ', '_')}.png")

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Plot Actual vs Predicted values on Test Set."""
    print(f"Generating Actual vs Predicted Plot: {title}...")
    plt.figure(figsize=(10, 8))
    
    # Sample if large
    if len(y_true) > 5000:
        n_samples = len(y_true)
        sample_indices = np.random.choice(n_samples, 5000, replace=False)
        
        # Convert to numpy arrays for consistent indexing
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        y_true_sample = y_true_arr[sample_indices]
        y_pred_sample = y_pred_arr[sample_indices]
            
        plt.scatter(y_pred_sample, y_true_sample, alpha=0.3, s=10)
        plt.title(f"{title} (Sampled 5000)")
    else:
        plt.scatter(y_pred, y_true, alpha=0.3, s=10)
        plt.title(title)
        
    # 45 degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel("Predicted Year")
    plt.ylabel("Actual Year")
    save_plot(plt.gcf(), f"pred_vs_act_{title.lower().replace(' ', '_')}.png")

def plot_unified_scissor_effect(df, break_year, title="Scissor Plot"):
    """
    Generate the 'Unified' Scissor Plot with LOWESS smoothing and Era differentiation.
    """
    print(f"Generating Unified Scissor Plot: {title}...")
    plt.figure(figsize=(10, 6))
    
    # Plot unified LOWESS
    # Note: df must contain 'year' and 'acousticness'
    sns.regplot(data=df, x='year', y='acousticness', 
                scatter_kws={'alpha': 0.05, 's': 2, 'color': 'gray'}, 
                line_kws={'color': 'purple', 'label': 'Unified Trend (LOWESS)'}, 
                color='purple', lowess=True)

    # Add Visual Distinction for Structural Break
    plt.axvline(x=break_year, color='red', linestyle='--', linewidth=1.5, label=f'Structural Break ({break_year})')
    
    # Shading eras
    plt.axvspan(df['year'].min(), break_year, alpha=0.05, color='blue', label='Analog Era')
    plt.axvspan(break_year, df['year'].max(), alpha=0.05, color='orange', label='Digital Era')
    
    # Annotations - Hardcoded positions roughly optimized for this dataset
    plt.text(1980, 0.8, 'Analog Decline', fontsize=10, color='blue', ha='center')
    plt.text(2010, 0.8, 'Digital Choice', fontsize=10, color='darkorange', ha='center')

    plt.title(f'{title}: The {break_year} Turning Point', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Acousticness')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.grid(True, alpha=0.3)
    
    # Save as the standard 'scissor_plot.png' to reflect it is now the official one
    save_plot(plt.gcf(), "scissor_plot.png")
