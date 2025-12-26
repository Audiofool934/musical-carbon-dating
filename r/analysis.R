# Musical Carbon Dating - R-Native Analysis Script (analysis1.R)
# This script performs the entire pipeline from raw data loading to final evaluation.

# === 1. Setup & Libraries ===
suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(readr)
    library(car) # For VIF
    library(MASS) # For RLM/WLS
    library(glmnet) # For Lasso/Ridge
    library(reshape2) # For heatmap
    library(gridExtra)
    library(tidyr) # For pivot_longer/pivot_wider
})

# Global ggplot2 theme for centering titles
theme_update(plot.title = element_text(hjust = 0.5))

# Create output directories
dir.create("output_r", showWarnings = FALSE)
dir.create("output_r/figures", showWarnings = FALSE)
dir.create("output_r/tables", showWarnings = FALSE)

cat("================================================\n")
cat("   Musical Carbon Dating: R-Native Analysis\n")
cat("================================================\n\n")

# === 2. Raw Data Loading & Preprocessing ===
# 2. Raw Data Loading & Preprocessing
cat(">>> Loading Raw Data from tracks.csv...\n")
# Flexible path checking for both dev (../data) and submission (data/) environments
data_paths <- c("data/tracks.csv", "../data/tracks.csv")
data_path <- NULL

for (path in data_paths) {
    if (file.exists(path)) {
        data_path <- path
        break
    }
}

if (is.null(data_path)) {
    stop("Error: tracks.csv not found. Checked: ", paste(data_paths, collapse=", "))
}
cat(sprintf("Found data at: %s\n", data_path))

# Reading with specific types to avoid confusion if release_date is mixed
raw_df <- read_csv(data_path, show_col_types = FALSE)

# 2.1 Extract Year from release_date and Filtering
cat("Extracting year and applying filters: 1960 <= year <= 2020, popularity > 30...\n")
filtered_df <- raw_df %>%
    mutate(year = as.numeric(substr(as.character(release_date), 1, 4))) %>%
    filter(!is.na(year)) %>%
    filter(year >= 1960 & year <= 2020) %>%
    filter(popularity > 30)

# 2.2 Feature Selection
# Audio features of interest
audio_features <- c(
    "loudness", "tempo", "duration_ms", "mode", "time_signature",
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "speechiness", "valence"
)

# Note: 'key' is excluded based on statistical insignificance (p=0.94)
data_subset <- filtered_df %>%
    dplyr::select(year, all_of(audio_features))

cat(sprintf("Filtered Sample Size: %d tracks\n", nrow(data_subset)))

# 2.3 Train/Test Split (80/20)
set.seed(42) # For reproducibility
train_idx <- sample(seq_len(nrow(data_subset)), size = 0.8 * nrow(data_subset))
train_raw <- data_subset[train_idx, ]
test_raw <- data_subset[-train_idx, ]

# 2.4 Standardization (Z-Score)
cat("Applying Z-Score Standardization...\n")
# Features to scale (all except year)
features_to_scale <- audio_features

# Compute scaling parameters from TRAINING set only
means <- sapply(train_raw[, features_to_scale], mean, na.rm = TRUE)
sds <- sapply(train_raw[, features_to_scale], sd, na.rm = TRUE)

# Manual Scaling function
standardize <- function(df, m, s) {
    for (feat in names(m)) {
        df[[feat]] <- (df[[feat]] - m[feat]) / s[feat]
    }
    return(df)
}

train_df <- standardize(train_raw, means, sds)
test_df <- standardize(test_raw, means, sds)

cat(sprintf("Train Set: %d rows\n", nrow(train_df)))
cat(sprintf("Test Set:  %d rows\n", nrow(test_df)))

# Feature list for later use
features <- features_to_scale

# === 3. Exploratory Data Analysis (EDA) ===
cat("\n>>> Running EDA...\n")

# 3.1 Correlation Heatmap
cor_matrix <- cor(train_df[, features])
melted_cor <- melt(cor_matrix)

p_corr <- ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "#2c6e7f", high = "#d97706", mid = "white", midpoint = 0, limit = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Feature Correlation Matrix")

ggsave("output_r/figures/r_correlation_heatmap.png", p_corr, width = 10, height = 8)

# 3.4 Descriptive Statistics (Reference Table - Using Raw Data)
cat("\n>>> Computing Descriptive Statistics (Raw Data)...\n")
stats_list <- lapply(train_raw, function(x) {
    if (is.numeric(x)) {
        c(Mean = mean(x, na.rm = TRUE), SD = sd(x, na.rm = TRUE), Min = min(x, na.rm = TRUE), Max = max(x, na.rm = TRUE))
    } else {
        NULL
    }
})
desc_stats <- do.call(rbind, stats_list)
desc_stats <- as.data.frame(desc_stats)
desc_stats$Feature <- rownames(desc_stats)
desc_stats <- desc_stats[, c("Feature", "Mean", "SD", "Min", "Max")]

print(desc_stats)
write_csv(desc_stats, "output_r/tables/descriptive_statistics.csv")

# === 4. Regression Analysis ===

# 4.1 Phase I: Simple Linear Regression (Loudness)
cat("\n>>> Phase I: SLR (Loudness War)...\n")
slr_model <- lm(year ~ loudness, data = train_df)
slr_summ <- summary(slr_model)
cat(sprintf("SLR R-squared: %.4f\n", slr_summ$r.squared))
cat(sprintf("Loudness Coef: %.4f (t = %.2f)\n", coef(slr_model)["loudness"], coef(slr_summ)["loudness", "t value"]))

# Figure 3: Evolution of Loudness
p_loudness <- ggplot(train_df, aes(x = year, y = loudness)) +
    geom_point(alpha = 0.05, color = "#2c6e7f", size = 0.5) +
    geom_smooth(method = "lm", color = "#d97706") +
    labs(title = "Evolution of Loudness (1960-2020)", x = "Year", y = "Loudness (Standardized)") +
    theme_minimal()

ggsave("output_r/figures/r_loudness_trend.png", p_loudness, width = 8, height = 6)

# 4.2 Phase II: Multiple Linear Regression (Baseline)
cat("\n>>> Phase II: MLR (Baseline)...\n")
mlr_model <- lm(year ~ ., data = train_df)
mlr_summ <- summary(mlr_model)
cat(sprintf("MLR R-squared: %.4f\n", mlr_summ$r.squared))

# Save MLR Coefficients
coef_df <- as.data.frame(coef(mlr_summ))
coef_df$Feature <- rownames(coef_df)
write_csv(coef_df, "output_r/tables/mlr_coefficients.csv")

# 4.2.b Partial F-Test for Non-Linearity
cat("\n>>> Checking Linearity (Partial F-Test)...\n")
# Features suspected of non-linearity (numeric only)
num_feat_diag <- c("loudness", "tempo", "duration_ms", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "valence")
poly_formula <- as.formula(paste("year ~ . +", paste(paste0("I(", num_feat_diag, "^2)"), collapse = " + ")))
mlr_poly <- lm(poly_formula, data = train_df)
anova_res <- anova(mlr_model, mlr_poly)
cat("Partial F-Test (Linear vs Quadratic):\n")
print(anova_res)

# 4.3 Phase III: Diagnostics (Breusch-Pagan)
cat("\n>>> Phase III: Diagnostics (Heteroscedasticity)...\n")
bp_test <- ncvTest(mlr_model)
cat(sprintf("Chisquare = %.4f, Df = %d, p = %.4g\n", bp_test$ChiSquare, bp_test$Df, bp_test$p))

# 4.4 Phase IV: Weighted Least Squares (WLS)
cat("\n>>> Phase IV: WLS (Refinement)...\n")
resid_sq <- resid(mlr_model)^2
log_resid_sq <- log(resid_sq + 1e-6)
aux_model <- lm(log_resid_sq ~ ., data = train_df)
wts <- 1 / exp(predict(aux_model))
wts <- wts / mean(wts)

wls_model <- lm(year ~ ., data = train_df, weights = wts)
wls_summ <- summary(wls_model)
cat(sprintf("WLS Weighted R-squared: %.4f\n", wls_summ$r.squared))

# Save WLS Coefficients
wls_coef_df <- as.data.frame(coef(wls_summ))
wls_coef_df$Feature <- rownames(wls_coef_df)
write_csv(wls_coef_df, "output_r/tables/wls_coefficients.csv")

# Save VIF Values
vif_vals <- vif(mlr_model)
vif_df <- data.frame(Feature = names(vif_vals), VIF = vif_vals)
write_csv(vif_df, "output_r/tables/vif_values.csv")

# === 5. Model Selection (Stepwise & Lasso) ===
cat("\n>>> Phase V: Model Selection...\n")
# 5.1 Stepwise
step_model <- step(mlr_model, direction = "both", trace = 0)
step_pred <- predict(step_model, newdata = test_df)

# 5.2 Lasso
x_train_mat <- as.matrix(train_df[, features])
y_train_vec <- train_df$year
lasso_cv <- cv.glmnet(x_train_mat, y_train_vec, alpha = 1)
lasso_best <- glmnet(x_train_mat, y_train_vec, alpha = 1, lambda = lasso_cv$lambda.min)
lasso_pred <- as.numeric(predict(lasso_best, newx = as.matrix(test_df[, features])))

# Metrics Comparison
model_metrics <- data.frame(
    Model = c("Full MLR", "Stepwise (AIC)", "LASSO"),
    RMSE = c(
        sqrt(mean((predict(mlr_model, newdata = test_df) - test_df$year)^2)),
        sqrt(mean((step_pred - test_df$year)^2)),
        sqrt(mean((lasso_pred - test_df$year)^2))
    ),
    R_Squared = c(
        summary(mlr_model)$r.squared,
        summary(step_model)$r.squared,
        1 - (sum((lasso_pred - test_df$year)^2) / sum((test_df$year - mean(test_df$year))^2))
    )
)
print(model_metrics)

# === 6. Final Evaluation & Nostalgia Index ===
cat("\n>>> Final Evaluation (Test Set)...\n")
wls_pred <- predict(wls_model, newdata = test_df)
rmse <- sqrt(mean((wls_pred - test_df$year)^2))
mae <- mean(abs(wls_pred - test_df$year))
med_ae <- median(abs(wls_pred - test_df$year))

cat(sprintf("Test RMSE (WLS): %.4f\n", rmse))
cat(sprintf("Test MAE (WLS):  %.4f\n", mae))
cat(sprintf("Median Abs Error: %.2f\n", med_ae))

# Save WLS Predictions & Results
test_results <- test_df
test_results$predicted_year <- wls_pred
test_results$nostalgia_index <- abs(wls_pred - test_df$year)
write_csv(test_results, "output_r/tables/r_test_predictions_v1.csv")

# Final Plot: Pred vs Actual (Hexbin)
library(hexbin)
p_final <- ggplot(test_results, aes(x = predicted_year, y = year)) +
    geom_hex(bins = 50) +
    scale_fill_gradientn(colors = c("#ffffff", "#4a8a9a", "#8b6b3e", "#d97706"), values = c(0, 0.3, 0.7, 1)) +
    geom_abline(intercept = 0, slope = 1, color = "#7c3e1d", linetype = "dashed") +
    coord_fixed(ratio = 1, xlim = c(1960, 2020), ylim = c(1960, 2020)) +
    labs(title = "Model Performance (Independent R Preprocessing)", x = "Predicted Year", y = "Actual Year") +
    theme_minimal()

ggsave("output_r/figures/r_wls_pred_vs_act_v1.png", p_final, width = 8, height = 7)

cat("\n>>> Generating Additional Figures for Slides...\n")

# Figure: Prediction Error by Era
# Create Era Bins
test_results$era <- cut(test_results$year, breaks = seq(1960, 2020, by = 10), include.lowest = TRUE, labels = paste0(seq(1960, 2010, by = 10), "s"))
p_error_era <- ggplot(test_results, aes(x = era, y = abs(nostalgia_index), fill = era)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    scale_fill_brewer(palette = "Spectral") +
    labs(title = "Prediction Error by Era (WLS)", x = "Musical Era", y = "Absolute Error (Years)") +
    ylim(0, 30) +
    theme_minimal() +
    theme(legend.position = "none")

ggsave("output_r/figures/r_error_by_era.png", p_error_era, width = 8, height = 6)

# Figure: Nostalgia Index Distribution
p_nostalgia <- ggplot(test_results, aes(x = nostalgia_index)) +
    geom_histogram(binwidth = 1, fill = "#d97706", color = "white", alpha = 0.8) +
    labs(title = "Distribution of Nostalgia Index", x = "Nostalgia Index (Abs Prediction Error)", y = "Count of Songs") +
    xlim(0, 40) +
    theme_minimal()

ggsave("output_r/figures/r_nostalgia_distribution.png", p_nostalgia, width = 8, height = 6)

# Figure: Feature Trends (Loudness + Acousticness + Energy)
long_trends <- train_df %>%
    dplyr::select(year, loudness, acousticness, energy) %>%
    tidyr::pivot_longer(cols = c(loudness, acousticness, energy), names_to = "Feature", values_to = "Value")

p_trends <- ggplot(long_trends, aes(x = year, y = Value, color = Feature)) +
    geom_smooth(method = "loess", se = FALSE, linewidth = 1.2) +
    labs(title = "Feature Trends Over Time (Smoothed)", x = "Year", y = "Standardized Value") +
    theme_minimal() +
    scale_color_manual(values = c("#2c6e7f", "#d97706", "#8b6b3e"))

ggsave("output_r/figures/r_feature_trends.png", p_trends, width = 10, height = 6)
cat("Saved r_feature_trends.png\n")

# Figure: Feature Distributions (for Report) - SAMPLED
cat("Generating Feature Distributions (Sampled)...\n")
# Sample 10000 rows to avoid memory crash
plot_sample <- train_df %>% dplyr::sample_n(min(10000, nrow(train_df)))

long_features <- plot_sample %>%
    tidyr::pivot_longer(cols = all_of(features), names_to = "Feature", values_to = "Value")

p_dist <- ggplot(long_features, aes(x = Value)) + 
    geom_histogram(bins = 30, fill = "#2c6e7f", color = "white", alpha = 0.7) +
    facet_wrap(~Feature, scales = "free") +
    labs(title = "Feature Distributions (Standardized - Sampled)") +
    theme_minimal()

ggsave("output_r/figures/r_feature_distributions.png", p_dist, width = 12, height = 10)
cat("Saved r_feature_distributions.png\n")

# Figure: Residuals vs Fitted (for Report)
cat("Generating Residuals vs Fitted...\n")
# We can use the WLS residuals
diag_df <- data.frame(Fitted = fitted(wls_model), Residuals = resid(wls_model)) %>%
    dplyr::sample_n(min(10000, nrow(train_df))) # Sampled for performance

p_resid <- ggplot(diag_df, aes(x = Fitted, y = Residuals)) +
    geom_point(alpha = 0.2, color = "#2c6e7f") +
    geom_smooth(color = "#d97706") +
    labs(title = "Residuals vs Fitted (WLS - Sampled)", x = "Fitted Values", y = "Residuals") +
    theme_minimal()

ggsave("output_r/figures/r_residuals_vs_fitted.png", p_resid, width = 8, height = 6)
cat("Saved r_residuals_vs_fitted.png\n")

# Figure: Lasso CV (for Report)
cat("Generating Lasso CV...\n")
png("output_r/figures/r_lasso_cv.png", width = 800, height = 600)
plot(lasso_cv)
title("LASSO Cross-Validation", line = 2.5)
dev.off()
cat("Saved r_lasso_cv.png\n")

# Figure: Model Selection Comparison (RMSE Bar Chart)
cat("Generating Model Comparison...\n")
p_models <- ggplot(model_metrics, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    coord_flip() +
    labs(title = "Model Comparison (RMSE)", x = "Model", y = "RMSE (Years)") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave("output_r/figures/r_model_selection_comparison.png", p_models, width = 8, height = 5)
cat("Saved r_model_selection_comparison.png\n")

# Figure: Coefficients with CI (for Report)
cat("Generating Coefficients CI...\n")
# Extract WLS coefficients and CI
wls_ci <- confint(wls_model)
wls_coefs <- coef(wls_model)
coef_plot_df <- data.frame(
    Feature = names(wls_coefs),
    Coefficient = wls_coefs,
    Lower = wls_ci[, 1],
    Upper = wls_ci[, 2]
)
# Remove Intercept for better visualization
coef_plot_df <- coef_plot_df[coef_plot_df$Feature != "(Intercept)", ]

p_coef <- ggplot(coef_plot_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
    geom_point(size = 3, color = "#d97706") +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "#2c6e7f") +
    coord_flip() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "WLS Coefficients (95% CI)", x = "Feature", y = "Standardized Effect on Year") +
    theme_minimal()

ggsave("output_r/figures/r_coefficients_ci.png", p_coef, width = 8, height = 8)
cat("Saved r_coefficients_ci.png\n")

cat("\n>>> R-Native Analysis Complete. Outputs saved to output_r/ \n")
