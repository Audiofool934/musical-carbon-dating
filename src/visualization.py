import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from .config import FIGURES_DIR, START_YEAR, END_YEAR

ANALOG_COLOR = "#2c6e7f"
DIGITAL_COLOR = "#d97706"
ACCENT_COLOR = "#7c3e1d"
PAPER_BG = "#ffffff"
PANEL_BG = "#ffffff"

def set_plot_style():
    """Set a consistent, retro-leaning visual theme."""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": "#c9c9c9",
            "axes.labelcolor": "#2b2b2b",
            "text.color": "#2b2b2b",
            "grid.color": "#e6e6e6",
            "grid.linestyle": "-",
            "grid.alpha": 0.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.family": ["Avenir", "DejaVu Sans", "sans-serif"],
        },
    )

set_plot_style()

def save_plot(fig, name):
    """Helper to save plot to figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    print(f"Saved plot to {path}")
    plt.close(fig)

def plot_scatter_matrix(df, cols, title="Scatter Matrix"):
    """Plot scatter matrix for selected columns."""
    print(f"Generating Scatter Matrix: {title}...")
    g = sns.pairplot(
        df[cols],
        diag_kind='kde',
        plot_kws={'alpha': 0.4, 's': 12, 'color': ANALOG_COLOR},
        diag_kws={'color': DIGITAL_COLOR},
    )
    g.fig.suptitle(title, y=1.02)
    save_plot(g.fig, f"{title.lower().replace(' ', '_')}.png")

def plot_residuals(model, title="Residuals Plots"):
    """Plot residuals vs fitted and Q-Q plot."""
    print(f"Generating Residual Plots: {title}...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Residuals vs Fitted
    sns.scatterplot(x=fitted, y=residuals, ax=axes[0], alpha=0.35, color=ANALOG_COLOR, s=18)
    axes[0].axhline(0, color=DIGITAL_COLOR, linestyle='--', linewidth=1.5)
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
    sns.scatterplot(
        data=plot_df,
        x='Leverage',
        y='Studentized_Residuals',
        alpha=0.45,
        color=ANALOG_COLOR,
        s=18,
    )
    
    # Add thresholds
    # High leverage usually > 2p/n or 3p/n
    p = len(model.params)
    n = model.nobs
    leverage_threshold = 3 * p / n
    plt.axvline(
        leverage_threshold,
        color=DIGITAL_COLOR,
        linestyle='--',
        linewidth=1.5,
        label=f'Threshold (3p/n = {leverage_threshold:.4f})',
    )
    plt.axhline(3, color=ACCENT_COLOR, linestyle='--', linewidth=1.2, label='Residual > 3')
    plt.axhline(-3, color=ACCENT_COLOR, linestyle='--', linewidth=1.2)
    
    plt.title(f"{title} (Studentized Residuals vs Leverage)")
    plt.xlabel('Leverage')
    plt.ylabel('Studentized Residuals')
    plt.legend()
    
    save_plot(plt.gcf(), "influence_plot.png")

def plot_coefficients(coefs, names, title="Regression Coefficients"):
    """Plot bar chart of coefficients."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefs, y=names, color=ANALOG_COLOR)
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.axvline(0, color=ACCENT_COLOR, linestyle='-', linewidth=1.2)
    save_plot(plt.gcf(), f"coefs_{title.lower().replace(' ', '_')}.png")

def plot_partial_regression(model, exog_idx, title="Partial Regression Plot"):
    """
    Plot Partial Regression Plot (Added Variable Plot).
    Targeting specific variable against year controlling for others.
    """
    print(f"Generating Partial Regression Plot: {title}...")
    import statsmodels.api as sm
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor(PAPER_BG)
    sm.graphics.plot_partregress_grid(model, fig=fig)
    save_plot(fig, f"partial_regression_{title.lower().replace(' ', '_')}.png")

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """Plot Actual vs Predicted values on Test Set."""
    print(f"Generating Actual vs Predicted Plot: {title}...")
    plt.figure(figsize=(10, 8))
    cmap = LinearSegmentedColormap.from_list(
        "analog_digital",
        [PANEL_BG, ANALOG_COLOR, DIGITAL_COLOR],
    )
    plt.hexbin(y_pred, y_true, gridsize=50, cmap=cmap, mincnt=1, linewidths=0)
    plt.title(title)
    plt.colorbar(label="Count")
        
    # 45 degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color=ACCENT_COLOR, lw=2)
    
    plt.xlabel("Predicted Year")
    plt.ylabel("Actual Year")
    plt.xlim(START_YEAR, END_YEAR)
    plt.ylim(START_YEAR, END_YEAR)
    save_plot(plt.gcf(), f"pred_vs_act_{title.lower().replace(' ', '_')}.png")

def plot_nostalgia_distribution(nostalgia_index, title="Nostalgia Index Distribution"):
    """Plot distribution of the nostalgia index with key thresholds."""
    print(f"Generating Nostalgia Index Distribution: {title}...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(nostalgia_index, bins=40, kde=True, color=ANALOG_COLOR, ax=ax)

    median = float(np.median(nostalgia_index))
    p95 = float(np.quantile(nostalgia_index, 0.95))

    ax.axvline(median, color=DIGITAL_COLOR, linestyle='--', linewidth=1.5, label=f"Median: {median:.1f}")
    ax.axvline(p95, color=ACCENT_COLOR, linestyle='--', linewidth=1.5, label=f"95th pct: {p95:.1f}")
    ax.set_xlabel("Absolute Error (Years)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    save_plot(fig, "nostalgia_index_distribution.png")

def plot_error_by_era(y_true, y_pred, era_size=10, title="Prediction Error by Era"):
    """Plot absolute prediction error grouped by era (default: decade)."""
    print(f"Generating Error by Era Plot: {title}...")
    df = pd.DataFrame({"year": y_true, "pred": y_pred})
    df = df.dropna(subset=["year", "pred"])
    df["year"] = df["year"].astype(int)
    df["abs_error"] = (df["pred"] - df["year"]).abs()
    df["era"] = (df["year"] // era_size) * era_size

    era_order = sorted(df["era"].unique())
    palette = sns.color_palette(
        sns.blend_palette([ANALOG_COLOR, DIGITAL_COLOR], n_colors=len(era_order))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="era",
        y="abs_error",
        order=era_order,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Era (Start Year)")
    ax.set_ylabel("Absolute Error (Years)")
    ax.set_title(title)
    save_plot(fig, "error_by_era.png")

def plot_coefficients_with_ci(model, title="Coefficients (95% CI)", include_const=False):
    """Plot coefficient estimates with 95% confidence intervals."""
    print(f"Generating Coefficient Plot: {title}...")
    params = model.params.copy()
    conf_int = model.conf_int()

    if not include_const and "const" in params.index:
        params = params.drop("const")
        conf_int = conf_int.drop("const")

    plot_df = pd.DataFrame(
        {
            "coef": params,
            "lower": conf_int[0],
            "upper": conf_int[1],
        }
    )
    plot_df["abs_coef"] = plot_df["coef"].abs()
    plot_df = plot_df.sort_values("abs_coef", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(plot_df))
    colors = [ANALOG_COLOR if c >= 0 else DIGITAL_COLOR for c in plot_df["coef"]]

    ax.errorbar(
        plot_df["coef"],
        y_pos,
        xerr=[plot_df["coef"] - plot_df["lower"], plot_df["upper"] - plot_df["coef"]],
        fmt="o",
        color=ACCENT_COLOR,
        ecolor="#9a9186",
        elinewidth=1.2,
        capsize=3,
        zorder=2,
    )
    ax.scatter(plot_df["coef"], y_pos, color=colors, s=60, zorder=3)
    ax.axvline(0, color=ACCENT_COLOR, linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("Coefficient Estimate")
    ax.set_title(title)
    save_plot(fig, "coefficients_with_ci.png")

def plot_scale_location(model, title="Scale-Location"):
    """Plot sqrt(|standardized residuals|) vs fitted values."""
    print(f"Generating Scale-Location Plot: {title}...")
    influence = model.get_influence()
    fitted = model.fittedvalues
    standardized = influence.resid_studentized_internal
    standardized = influence.resid_studentized_internal
    scale = np.sqrt(np.abs(standardized))

    # Sample for performance (LOWESS is O(N^2))
    plot_data = pd.DataFrame({'fitted': fitted, 'scale': scale})
    if len(plot_data) > 5000:
        plot_data = plot_data.sample(5000, random_state=42)
        print("Note: Scale-Location plot sampled to 5000 points for performance.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=plot_data['fitted'],
        y=plot_data['scale'],
        ax=ax,
        color=ANALOG_COLOR,
        alpha=0.4,
        s=18,
    )
    sns.regplot(
        x=plot_data['fitted'],
        y=plot_data['scale'],
        scatter=False,
        ax=ax,
        color=DIGITAL_COLOR,
        lowess=True,
        line_kws={"linewidth": 2},
    )
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Sqrt(|Standardized Residuals|)")
    ax.set_title(title)
    save_plot(fig, "scale_location.png")

def plot_weighted_fit(y_true, y_pred, weights, title="Weighted Fit (Training Data)"):
    """
    Plot Actual vs Predicted with point size/color determined by WLS weights.
    Helps visualize why Weighted R2 is higher than unweighted visual impression.
    """
    print(f"Generating Weighted Fit Plot: {title}...")
    plt.figure(figsize=(11, 8))
    
    # Normalize weights for better plotting size
    w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
    sizes = 10 + w_norm * 100  # Size range 10 to 110
    
    cmap = LinearSegmentedColormap.from_list(
        "weight_map",
        [ "#e0e0e0", DIGITAL_COLOR, ANALOG_COLOR], 
    )
    
    scatter = plt.scatter(
        y_pred, 
        y_true, 
        c=weights, 
        s=sizes, 
        alpha=0.6, 
        cmap=cmap, 
        edgecolors='none'
    )
    
    # 45 degree line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color=ACCENT_COLOR, lw=2, label='Perfect Fit')
    
    plt.colorbar(scatter, label="Weight (Importance)")
    plt.xlabel("Predicted Year")
    plt.ylabel("Actual Year")
    plt.xlim(START_YEAR, END_YEAR)
    plt.ylim(START_YEAR, END_YEAR)
    plt.title(title)
    plt.legend()
    
    # Annotation explaining the visual
    plt.text(
        START_YEAR + 2, END_YEAR - 5, 
        "Larger/Darker points = Higher reliability\n(Model prioritizes these)", 
        fontsize=11, 
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc')
    )
    
    save_plot(plt.gcf(), "weighted_fit_training.png")

def plot_unified_scissor_effect(df, break_year, title="Scissor Plot"):
    """
    Generate the 'Unified' Scissor Plot with LOWESS smoothing and Era differentiation.
    """
    print(f"Generating Unified Scissor Plot: {title}...")
    plt.figure(figsize=(10, 6))
    
    # Plot unified LOWESS
    # Plot unified LOWESS
    # Note: df must contain 'year' and 'acousticness'
    plot_df = df.copy()
    if len(plot_df) > 5000:
        plot_df = plot_df.sample(5000, random_state=42)
        print("Note: Scissor Plot sampled to 5000 points for performance.")

    sns.regplot(
        data=plot_df,
        x='year',
        y='acousticness',
        scatter_kws={'alpha': 0.1, 's': 8, 'color': '#6b6b6b'},
        line_kws={'color': DIGITAL_COLOR, 'label': 'Unified Trend (LOWESS)', 'linewidth': 2.5},
        color=DIGITAL_COLOR,
        lowess=True,
    )

    # Add Visual Distinction for Structural Break
    plt.axvline(
        x=break_year,
        color=ACCENT_COLOR,
        linestyle='--',
        linewidth=1.5,
        label=f'Structural Break ({break_year})',
    )
    
    # Shading eras
    plt.axvspan(START_YEAR, break_year, alpha=0.08, color=ANALOG_COLOR, label='Analog Era')
    plt.axvspan(break_year, END_YEAR, alpha=0.08, color=DIGITAL_COLOR, label='Digital Era')
    
    # Annotations - Hardcoded positions roughly optimized for this dataset
    plt.text(1980, 0.8, 'Analog Decline', fontsize=10, color=ANALOG_COLOR, ha='center')
    plt.text(2010, 0.8, 'Digital Choice', fontsize=10, color=DIGITAL_COLOR, ha='center')

    plt.title(f'{title}: The {break_year} Turning Point', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Acousticness')
    plt.xlim(START_YEAR, END_YEAR)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.grid(True, alpha=0.3)
    
    # Save as the standard 'scissor_plot.png' to reflect it is now the official one
    save_plot(plt.gcf(), "scissor_plot.png")
