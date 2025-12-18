### 1. Foundations and Simple Linear Regression (SLR)
The course began by grounding regression analysis in history, tracing its origins to Francis Galton’s biogenetic studies on "regression towards mediocrity" and the development of the Least Squares method by Legendre and Gauss in astronomy.

*   **The Model:** The course defined the distinction between deterministic functional relationships (like Ohm's Law) and statistical relationships which include random disturbance terms. The theoretical SLR model is expressed as $Y = \beta_{0} + \beta_{1}X + \epsilon$.
*   **Estimation:** Two primary methods for parameter estimation were covered:
    *   **Ordinary Least Squares (OLS):** Minimizing the sum of squared deviations to find the line of best fit.
    *   **Maximum Likelihood Estimation (MLE):** Under the assumption of normality, MLE yields estimates equivalent to OLS.
*   **Properties & Inference:** A key theoretical component was the **Gauss-Markov Theorem**, which states that under specific conditions (errors have mean zero, constant variance, and are uncorrelated), the OLS estimator is the Best Linear Unbiased Estimator (BLUE). The course covered hypothesis testing (t-tests for coefficients, F-tests for model significance) and the construction of confidence intervals.

### 2. Multiple Linear Regression (MLR)
The course expanded the framework to models with multiple independent variables ($p > 1$), introducing matrix notation to handle the increased complexity.

*   **Interpretation:** Regression coefficients in MLR represent the average change in the dependent variable for a one-unit change in a specific predictor, *holding all other variables fixed*.
*   **Inference:** The course distinguished between the Global F-test (testing if *any* predictor is significant) and partial t-tests (testing specific predictors). It also introduced the **Partial F-test** to assess the contribution of a subset of variables.
*   **Standardization:** To address variables with different units (e.g., kilometers vs. currency), the course covered data centering and standardization, allowing for the comparison of relative importance via standardized regression coefficients.

### 3. Regression Diagnostics
A significant portion of the course focused on diagnosing and remedying violations of the core model assumptions (Normality, Homoscedasticity, Independence).

*   **Residual Analysis:** This is the primary diagnostic tool. The course introduced various residual types, including standardized, studentized, and deleted residuals, to identify anomalies.
*   **Heteroscedasticity:**
    *   *Diagnosis:* Residual plots, Spearman’s rank correlation test, and the Goldfeld-Quandt test.
    *   *Remedy:* **Weighted Least Squares (WLS)**, which assigns weights inversely proportional to the error variance to maintain estimator efficiency.
*   **Autocorrelation:**
    *   *Context:* Common in time-series economic data due to inertia or lag effects.
    *   *Diagnosis:* The **Durbin-Watson (DW) test** is used to detect first-order autocorrelation.
    *   *Remedy:* Iterative methods (Cochrane-Orcutt) and differencing were presented to transform the data into a stationary series.
*   **Outliers and Influence:** The course distinguished between outliers (unusual Y values) and leverage points (unusual X values). **Cook’s Distance** was introduced as a metric to identify influential points that disproportionately affect the regression coefficients.
*   **Transformations:** The **Box-Cox transformation** was taught as a method to mathematically determine the optimal power transformation for the dependent variable to correct non-normality or heteroscedasticity.

### 4. Multicollinearity
The course addressed the issue where independent variables are strongly correlated, which inflates the variance of coefficient estimates and can cause sign reversals.

*   **Diagnosis:** The Variance Inflation Factor (VIF) and Condition Indices (derived from eigenvalues) were presented as diagnostic tools.
*   **Ridge Regression:** To handle multicollinearity, the course introduced Ridge Regression. This method introduces a bias (via a penalty term $\lambda$) to significantly reduce the variance of the estimators, trading unbiasedness for stability.

### 5. Model Selection
The course moved beyond fitting models to selecting the "best" model, balancing goodness-of-fit with model complexity (parsimony).

*   **Criteria:** Metrics such as Adjusted $R^2$, **AIC** (Akaike Information Criterion), **BIC** (Bayesian Information Criterion), and Mallows' $C_p$ were used to evaluate models.
*   **Algorithms:** Traditional methods included All Subsets Regression, Forward Selection, Backward Elimination, and Stepwise Regression.
*   **Penalized Regression:** Modern techniques that perform selection and estimation simultaneously were covered, including:
    *   **LASSO:** Uses an $L_1$ penalty to shrink coefficients to exactly zero.
    *   **SCAD, MCP, and Elastic Net:** Advanced penalties addressing limitations of LASSO.
    *   **Oracle Property:** The theoretical ideal where a selection method identifies the true model and estimates non-zero coefficients as efficiently as if the true model were known.

<!-- ### 6. Generalized Linear Models (GLM)
Finally, the course extended regression beyond normal distributions to the **Exponential Family** of distributions (e.g., Binomial, Poisson).

*   **Structure:** GLMs consist of a random component, a systematic component (linear predictor), and a **link function** connecting the mean to the predictor.
*   **Logistic Regression:** A key focus was Binary Logistic Regression for dichotomous outcomes (e.g., survival).
    *   *Estimation:* Uses Maximum Likelihood via Iterative Weighted Least Squares.
    *   *Interpretation:* Coefficients are interpreted via **Odds Ratios**.
    *   *Evaluation:* The course covered the Confusion Matrix, Sensitivity/Specificity, and the **ROC curve/AUC** for assessing predictive performance.

### Analogy
To solidify your understanding, imagine regression analysis like **tuning a high-end sound system**.

*   **Simple Linear Regression** is the volume knob—one dial (variable) that directly increases or decreases the output.
*   **Multiple Linear Regression** is the full equalizer (EQ). You now have sliders for bass, treble, and mid-tones (multiple variables). You are trying to find the perfect balance where adjusting the bass doesn't ruin the clarity of the vocals (partial regression coefficients).
*   **Diagnostics** represent troubleshooting.
    *   *Heteroscedasticity* is like static that gets louder as the volume increases; you need a filter (Weighted Least Squares) to manage it.
    *   *Multicollinearity* is when two frequencies clash and create feedback; you use a limiter (Ridge Regression) to dampen one so the sound stays clear.
*   **Model Selection** is choosing which instruments to include in the mix. You don't want every instrument playing at once (overfitting); you want just enough to make the song sound perfect (AIC/BIC/LASSO).
*   **GLM** is realizing you aren't just mixing music (continuous data), but sometimes you are mixing a light show (binary data: on/off). You need a completely different interface (Link Function) to control the lights using the same underlying logic. -->