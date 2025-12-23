## Speaker 3: The Audit (9:00 - 14:00)
**Role**: The "Auditor". Discusses diagnostics, heteroscedasticity, and physical interpretations of residuals.

**(9:00) Slide 10: Diagnostic Audit**
"Thanks, Speaker 2.
In data science, a model is only as good as its assumptions. Before blindly trying to improve accuracy, we checked validity. I ran a rigorous **Diagnostic Audit** of the Gauss-Markov assumptions.
This is where the project revealed its statistical complexity.

1.  **Independence**: We ran the **Durbin-Watson Test** to check for autocorrelation. Often in time-series data, errors "leak" into each other. But our result was a perfect **2.001**. This is a "textbook" pass (where 2.0 is ideal). It means our residuals are truly independent; the model has successfully extracted the deterministic trends, leaving behind pure, uncorrelated "white noise".
2.  **Multicollinearity**: This was a surprise. We expected high overlap between Loudness and Energy. While they are correlated with a Pearson $r$ of **0.73**, the Variance Inflation Factor (VIF) for Energy was only **3.62**. This is well below the danger threshold of 5 or 10. This means that while loud songs *tend* to be energetic, there are enough 'quiet but energetic' songs (like minimal techno) and 'loud but calm' songs (like shoegaze) for the model to distinguish them.

**(11:00) Slide 11: Homoscedasticity (The Critical Failure)**
"Then we checked Homoscedasticity—the assumption that the variance of errors is constant over time.
This was our critical failure.
We ran the **Breusch-Pagan Test**. The test statistic was a massive **19,567**, with a p-value of 0.0000.
To put that in perspective, a 'significant' result is usually around 4 or 5. 19,000 implies that the variance of music production is *strictly* time-dependent."

**(12:30) Slide 10 (Continued): The Variance Explosion**
"Let's look at this **Boxplot of Error by Decade**. This is the clearest proof of our finding.
- **The Box (IQR)**: Represents the middle 50% of songs. In the 60s (Left), the boxes are short and tight. Production was standardized.
- **The Explosion**: As we move right to the 2000s, the boxes grow significantly taller.
- This visualizes **Heteroscedasticity** perfectly. The 'spread' of valid musical styles is widening. Using a Boxplot here is standard statistical practice to show how variance stability ($Var(\epsilon)$) is violated across groups."
