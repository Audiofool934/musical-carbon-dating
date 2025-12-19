from src.data_loader import get_train_test_data
from src.analysis import RegressionAnalysis
from src.visualization import (
    plot_residuals, plot_influence, plot_partial_regression, 
    plot_actual_vs_predicted, plot_nostalgia_distribution,
    plot_scale_location, plot_coefficients_with_ci, plot_error_by_era
)
from src.eda import run_eda
from src.config import TARGET_COL, BREAK_YEAR
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("Musical Carbon Dating: Two-Stage Analysis")
    print("Stage 1: Year Prediction (Regression Modeling)")
    print("Stage 2: Nostalgia Index (Model Application)")
    print("="*80)
    
    # ========== STAGE 1: YEAR PREDICTION ==========
    
    # === Part 1: Data Preparation ===
    print("\n" + "="*80)
    print("PART 1: DATA PREPARATION")
    print("="*80)
    train_df, test_df = get_train_test_data()  # Default: random split
    
    # === Part 2: Exploratory Data Analysis ===
    print("\n" + "="*80)
    print("PART 2: EXPLORATORY DATA ANALYSIS")
    print("="*80)
    run_eda(train_df)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # Initialize Analysis Engine
    analyzer = RegressionAnalysis(X_train, y_train, X_test, y_test)
    
    # === Phase I: Simple Linear Regression ===
    print("\n" + "="*80)
    print("PHASE I: SIMPLE LINEAR REGRESSION")
    print("="*80)
    slr_model = analyzer.fit_slr(feature='loudness')
    
    # === Phase II: Multiple Linear Regression (Baseline) ===
    print("\n" + "="*80)
    print("PHASE II: MULTIPLE LINEAR REGRESSION (Baseline)")
    print("="*80)
    mlr_model = analyzer.fit_mlr()
    
    # Generate baseline MLR prediction plot (for comparison with WLS)
    mlr_features = [c for c in X_train.columns if c != 'year']
    import statsmodels.api as sm
    X_test_mlr = sm.add_constant(X_test[mlr_features], has_constant='add')
    mlr_pred = mlr_model.predict(X_test_mlr)
    plot_actual_vs_predicted(y_test, mlr_pred, title="Baseline MLR Predictions")
    
    # Partial Regression Plots (Added Variable Plots)
    plot_partial_regression(mlr_model, exog_idx=0, title="MLR Partial Regression")
    
    # === Phase III: Regression Diagnostics ===
    print("\n" + "="*80)
    print("PHASE III: REGRESSION DIAGNOSTICS")
    print("="*80)
    
    # 3A. Heteroscedasticity
    print("\n>>> 3A. Heteroscedasticity Diagnosis & Remedy")
    analyzer.check_heteroscedasticity(mlr_model)
    wls_model, wls_weights = analyzer.fit_wls(mlr_model)
    
    # Visual Verification of WLS Fit
    from src.visualization import plot_weighted_fit
    plot_weighted_fit(
        y_train, 
        wls_model.fittedvalues, 
        wls_weights, 
        title="WLS Weighted Fit (Training)"
    )
    
    # 3B. Non-normality
    print("\n>>> 3B. Normality Check & Box-Cox Transformation")
    plot_residuals(mlr_model, title="MLR Residuals")
    plot_scale_location(mlr_model, title="MLR Scale-Location")
    y_transformed, lambda_best = analyzer.run_box_cox()
    # Note: For full implementation, would refit model on transformed y
    # Skipping for now to keep pipeline coherent
    
    # 3C. Outliers & Influence
    print("\n>>> 3C. Outliers & Influential Points")
    plot_influence(mlr_model, title="MLR Influence")
    
    # === Phase IV: Multicollinearity ===
    print("\n" + "="*80)
    print("PHASE IV: MULTICOLLINEARITY DIAGNOSIS & REMEDY")
    print("="*80)
    
    vif_data = analyzer.check_multicollinearity()
    analyzer.print_correlation_loudness_energy()
    
    # Test robustness without popularity
    mlr_nopop = analyzer.fit_mlr_no_popularity()
    
    # Ridge Regression
    ridge_model = analyzer.run_ridge_regression(alpha=1.0)
    
    # === Phase V: Model Selection ===
    print("\n" + "="*80)
    print("PHASE V: MODEL SELECTION")
    print("="*80)
    
    # Stepwise Selection (AIC-based)
    stepwise_features = analyzer.stepwise_selection()
    
    # LASSO Selection
    lasso_features = analyzer.model_selection_lasso(alpha=0.01)
    
    print(f"\nFeature Comparison:")
    print(f"Stepwise (AIC) selected: {len(stepwise_features)} features")
    print(f"LASSO selected: {len(lasso_features)} features")
    
    # === Phase VI: Model Comparison & Best Model Selection ===
    print("\n" + "="*80)
    print("PHASE VI: MODEL COMPARISON & FINAL EVALUATION")
    print("="*80)
    
    # Compare all candidate models
    models_dict = {
        'MLR_Full': mlr_model,
        'MLR_NoPop': mlr_nopop,
        'WLS': wls_model
    }
    
    comparison_df = analyzer.compare_models(models_dict, test_set=True)
    
    # Select best model based on AIC
    best_model_name = comparison_df.sort_values('AIC').iloc[0]['Model']
    best_model = models_dict[best_model_name]
    
    print(f"\n✓ Best Model Selected: {best_model_name}")
    plot_coefficients_with_ci(best_model, title=f"{best_model_name} Coefficients (95% CI)")
    
    # Final Evaluation on Test Set
    y_pred = analyzer.evaluate_model(best_model)
    plot_actual_vs_predicted(y_test, y_pred, title=f"Best Model ({best_model_name}) Predictions")
    
    # ========== STAGE 2: NOSTALGIA INDEX ==========
    
    print("\n" + "="*80)
    print("STAGE 2: NOSTALGIA INDEX APPLICATION")
    print("="*80)
    
    # Calculate Nostalgia Index (Residual-based approach)
    nostalgia_index = np.abs(y_pred - y_test)
    
    print(f"\nNostalgia Index Statistics:")
    print(f"  Mean: {nostalgia_index.mean():.2f} years")
    print(f"  Median: {nostalgia_index.median():.2f} years")
    print(f"  Std: {nostalgia_index.std():.2f} years")
    print(f"  Max: {nostalgia_index.max():.2f} years")

    plot_nostalgia_distribution(nostalgia_index)
    plot_error_by_era(y_test, y_pred)
    
    # Identify highly nostalgic songs (top 5%)
    threshold_95 = nostalgia_index.quantile(0.95)
    highly_nostalgic = nostalgia_index[nostalgia_index >= threshold_95]
    
    print(f"\nHighly Nostalgic Songs (Top 5%, Index ≥ {threshold_95:.1f} years):")
    print(f"  Count: {len(highly_nostalgic)}")
    
    # Save results
    test_df_with_predictions = test_df.copy()
    test_df_with_predictions['predicted_year'] = y_pred
    test_df_with_predictions['nostalgia_index'] = nostalgia_index
    
    output_path = 'output/tables/predictions_with_nostalgia_index.csv'
    test_df_with_predictions.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print(f"  - Best Model: {best_model_name}")
    print(f"  - Test RMSE: {np.sqrt(((y_pred - y_test)**2).mean()):.2f} years")
    print(f"  - Mean Nostalgia Index: {nostalgia_index.mean():.2f} years")
    print(f"  - All figures saved to output/figures/")
    print(f"  - Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
