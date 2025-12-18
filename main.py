from src.data_loader import get_train_test_data
from src.analysis import RegressionAnalysis
from src.visualization import (
    plot_residuals, plot_influence, plot_partial_regression, 
    plot_actual_vs_predicted, plot_unified_scissor_effect
)
from src.eda import run_eda
from src.config import TARGET_COL, BREAK_YEAR

def main():
    print("--- Musical Carbon Dating: Full Analysis Pipeline ---")
    
    # 1. Load and Preprocess Data
    train_df, test_df = get_train_test_data()
    
    # 2. Exploratory Data Analysis
    print("\nRunning Exploratory Data Analysis...")
    run_eda(train_df)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    # 3. Initialize Analysis Engine
    analyzer = RegressionAnalysis(X_train, y_train, X_test, y_test)
    
    # 4. Fitting Models
    analyzer.fit_slr(feature='loudness')
    mlr_model = analyzer.fit_mlr()
    
    # Robustness Checks
    analyzer.print_correlation_loudness_energy()
    analyzer.fit_mlr_no_popularity()
    
    # 5. Diagnostics & Visuals
    print("\nRunning Model Diagnostics...")
    analyzer.check_multicollinearity()
    analyzer.check_heteroscedasticity(mlr_model)
    
    plot_residuals(mlr_model, title="MLR Residuals")
    plot_influence(mlr_model, title="MLR Influence")
    plot_partial_regression(mlr_model, exog_idx='acousticness', title="Partial Regression - Acousticness")
    
    # 6. Assumption Remedies & Poly Checks
    analyzer.check_nonlinearity(mlr_model, feature='duration_ms')
    analyzer.run_ridge_regression(alpha=1.0)
    
    # 7. Model Selection
    print("\n--- Model Selection Comparison ---")
    lasso_feats = analyzer.model_selection_lasso(alpha=0.01)
    stepwise_feats = analyzer.stepwise_selection()
    
    # 8. Structural Break (Digital Revolution)
    print(f"\nModeling Structural Break at {BREAK_YEAR}...")
    final_break_model = analyzer.chow_test(break_year=BREAK_YEAR)
    
    # 9. Final Validation on Test Set
    print("\nEvaluating Baseline MLR Model (Blind Prediction)...")
    y_pred_mlr = analyzer.evaluate_model(mlr_model, features_used=list(X_train.columns))
    plot_actual_vs_predicted(y_test, y_pred_mlr, title="Baseline MLR Predictions")

    print(f"\nEvaluating Structural Break Model at {BREAK_YEAR} (Explanatory)...")
    y_pred_break = analyzer.evaluate_model(final_break_model, features_used=stepwise_feats + ['break_dummy'])
    
    # Generate Final Fit Plot
    plot_actual_vs_predicted(y_test, y_pred_break, title="Final Break Model Predictions")
    
    # Generate Unified Scissor Plot
    plot_unified_scissor_effect(train_df, BREAK_YEAR, title="Unified Scissor Plot")
    
    print("\nPipeline execution complete. Results saved to output/ folder.")

if __name__ == "__main__":
    main()
