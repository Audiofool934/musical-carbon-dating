import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_COLS, BREAK_YEAR

class RegressionAnalysis:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}

    def fit_slr(self, feature='loudness'):
        """Phase II: Simple Linear Regression"""
        print(f"\n--- Phase II: Simple Linear Regression (Feature: {feature}) ---")
        X = sm.add_constant(self.X_train[feature])
        model = sm.OLS(self.y_train, X).fit()
        self.models['SLR'] = model
        print(model.summary())
        return model

    def fit_mlr(self, features=None, compute_test_rmse=True):
        """Phase III: Multiple Linear Regression (Full Model)"""
        print("\n--- Phase III: Multiple Linear Regression (Full Model) ---")
        if features is None:
            features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
        
        X = sm.add_constant(self.X_train[features])
        model = sm.OLS(self.y_train, X).fit()
        self.models['MLR'] = model
        print(model.summary())
        
        # Also compute test RMSE for MLR
        if compute_test_rmse and self.X_test is not None:
            X_test = sm.add_constant(self.X_test[features], has_constant='add')
            y_pred = model.predict(X_test)
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            print(f"\nMLR Test Set RMSE: {rmse:.4f} years")
            
        return model
    
    def fit_mlr_no_popularity(self):
        """Phase III-B: Robustness Check - MLR without Popularity"""
        print("\n--- Phase III-B: Robustness Check (No Popularity) ---")
        features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS and c != 'popularity']
        
        X = sm.add_constant(self.X_train[features])
        model = sm.OLS(self.y_train, X).fit()
        self.models['MLR_NoPop'] = model
        print(f"R-squared (no popularity): {model.rsquared:.4f}")
        
        # Test RMSE
        X_test = sm.add_constant(self.X_test[features], has_constant='add')
        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        print(f"Test Set RMSE (no popularity): {rmse:.4f} years")
        
        return model
    
    def print_correlation_loudness_energy(self):
        """Print Loudness-Energy correlation for verification"""
        print("\n--- Correlation Check: Loudness vs Energy ---")
        corr = self.X_train['loudness'].corr(self.X_train['energy'])
        print(f"Pearson r(Loudness, Energy) = {corr:.4f}")

    def check_multicollinearity(self, features=None):
        """Phase IV: VIF Check"""
        print("\n--- Phase IV: Checking Multicollinearity (VIF) ---")
        if features is None:
            features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
            
        X = sm.add_constant(self.X_train[features])
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
        print(vif_data.sort_values(by="VIF", ascending=False))
        return vif_data

    def run_ridge_regression(self, features=None, alpha=1.0):
        """Phase IV: Ridge Regression"""
        print(f"\n--- Phase IV: Ridge Regression (Alpha={alpha}) ---")
        if features is None:
            features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train[features])
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, self.y_train)
        
        print("Ridge Coefficients:")
        for name, coef in zip(features, ridge.coef_):
            print(f"{name}: {coef:.4f}")
        return ridge

    def check_heteroscedasticity(self, model):
        """Phase IV: Heteroscedasticity Tests"""
        print("\n--- Phase IV: Heteroscedasticity Tests ---")
        test = sms.het_breuschpagan(model.resid, model.model.exog)
        print(f"Breusch-Pagan Test: LM={test[0]:.4f}, p-value={test[1]:.4f}")
        
        # Suggest Box-Cox if significant
        if test[1] < 0.05:
            print("Heteroscedasticity detected! Suggesting Box-Cox transformation.")
            
    def run_box_cox(self):
        """Phase IV: Box-Cox Transformation on Target"""
        print("\n--- Phase IV: Box-Cox Transformation ---")
        # Shift y to be positive if needed (years are positive)
        y_transformed, lambda_best = stats.boxcox(self.y_train)
        print(f"Best Lambda: {lambda_best:.4f}")
        return y_transformed, lambda_best

    def fit_wls(self, model):
        """Phase IV: Weighted Least Squares (Alternative to Box-Cox)"""
        print("\n--- Phase IV: Weighted Least Squares (WLS) ---")
        # Estimate weights using residuals from OLS
        residuals = model.resid
        # Simple approach: weight = 1 / resid^2 (or variance of residuals group)
        # More robust: use absolute residuals helpful for heteroscedasticity
        weights = 1.0 / (residuals.abs() + 1e-6) # prevent div by zero
        
        wls_model = sm.WLS(self.y_train, model.model.exog, weights=weights).fit()
        print(wls_model.summary())
        return wls_model

    def check_nonlinearity(self, model, feature='duration_ms'):
        """Phase IV: Check for non-linearity & Partial F-Test"""
        print(f"\n--- Phase IV: Non-linearity Check ({feature}) ---")
        X = self.X_train.copy()
        features = [c for c in X.columns if c != 'year' and c in FEATURE_COLS]
        
        # Base model (without poly) - assumed passed as 'model' or refit
        X_base = sm.add_constant(X[features])
        # model_base = sm.OLS(self.y_train, X_base).fit() 
        
        # Extended model (with poly)
        X[f'{feature}_sq'] = X[feature] ** 2
        X_poly = sm.add_constant(X[features + [f'{feature}_sq']])
        
        model_poly = sm.OLS(self.y_train, X_poly).fit()
        print(model_poly.summary())
        
        # Partial F-Test (anova_lm)
        # We need to ensure models are nested and fitted on same data
        # Let's simple check the t-test of the squared term from summary first
        # But explicitly:
        print("\nPartial F-Test (Base vs Poly):")
        # Re-fit base to be sure
        model_base = sm.OLS(self.y_train, X_base).fit()
        
        # F-test
        f_test = model_poly.compare_f_test(model_base)
        print(f"F-Statistic: {f_test[0]:.4f}, p-value: {f_test[1]:.4f}")
        
        if f_test[1] < 0.05:
            print(f"Significant non-linearity found in {feature}!")
            
        return model_poly

    def stepwise_selection(self):
        """Phase V: Stepwise Selection (AIC based) - Comparison with Lasso"""
        print(f"\n--- Phase V: Stepwise Selection (Forward) ---")
        features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
        initial_features = []
        best_aic = float('inf')
        best_features = []
        
        # Simplified Forward Selection
        current_features = []
        remaining_features = list(features)
        
        # Limit steps for performance if needed, but 13 features is small
        while remaining_features:
            aic_with_candidates = []
            for candidate in remaining_features:
                X = sm.add_constant(self.X_train[current_features + [candidate]])
                model = sm.OLS(self.y_train, X).fit()
                aic_with_candidates.append((model.aic, candidate))
            
            aic_with_candidates.sort()
            best_new_aic, best_candidate = aic_with_candidates[0]
            
            if best_new_aic < best_aic:
                current_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                best_aic = best_new_aic
                # print(f"Step: Added {best_candidate}, AIC: {best_aic:.2f}")
            else:
                break
                
        print(f"Stepwise Selected Features: {current_features}")
        return current_features

    def model_selection_lasso(self, features=None, alpha=0.1):
        """Phase V: Model Selection with Lasso"""
        # ... existing ...
        print(f"\n--- Phase V: Model Selection (Lasso, Alpha={alpha}) ---")
        if features is None:
            features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
            
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train[features])
        
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_scaled, self.y_train)
        
        selected_features = [f for f, c in zip(features, lasso.coef_) if c != 0]
        print(f"Lasso Selected Features: {selected_features}")
        return selected_features

    def chow_test(self, break_year=BREAK_YEAR):
        """Phase VI: Chow Test / Structural Break"""
        print(f"\n--- Phase VI: Structural Break Test (Year > {break_year}) ---")
        # Dummy variable
        X = self.X_train.copy()
        X['break_dummy'] = (self.y_train > break_year).astype(int)
        
        # Interactions
        features = [c for c in self.X_train.columns if c != 'year' and c in FEATURE_COLS]
        for f in features:
            X[f'interact_{f}'] = X[f] * X['break_dummy']
            
        full_features = features + ['break_dummy'] + [f'interact_{f}' for f in features]
        X_const = sm.add_constant(X[full_features])
        
        model_break = sm.OLS(self.y_train, X_const).fit()
        print(model_break.summary())
        return model_break

    def evaluate_model(self, model, features_used):
        """Phase VI: Evaluation on Test Set"""
        print("\n--- Phase VI: Final Model Evaluation ---")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Prepare Test Data
        # features_used must match the columns in model.params (excluding const)
        # Handle interaction terms if structural break model
        
        X_test_processed = self.X_test.copy()
        
        # Check if model has interaction terms
        is_structural = 'break_dummy' in features_used or 'break_dummy' in model.params.index
        
        if is_structural:
            break_year = BREAK_YEAR
            X_test_processed['break_dummy'] = (self.y_test > break_year).astype(int)
            base_feats = [f for f in FEATURE_COLS if f in X_test_processed.columns]
            for f in base_feats:
                if f"interact_{f}" in model.params.index:
                     X_test_processed[f'interact_{f}'] = X_test_processed[f] * X_test_processed['break_dummy']
        
        # Align columns
        # Add const
        X_test_const = sm.add_constant(X_test_processed, has_constant='add')
        
        # Select Only columns present in model
        # model.params includes const
        cols_needed = model.params.index.tolist()
        
        # Ensure all columns exist (fill 0 if missing? Should not happen if logic is correct)
        # Note: const is mostly added by statsmodels
        
        X_final = X_test_const[cols_needed]
        
        y_pred = model.predict(X_final)
        
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        
        print(f"Test Set RMSE: {rmse:.4f}")
        print(f"Test Set MAE: {mae:.4f}")
        
        return y_pred
