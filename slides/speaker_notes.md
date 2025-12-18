# Speaker Notes: Musical Carbon Dating (Group Presentation - 5 Speakers)
**Target Time**: 25 Minutes (approx. 5 mins per speaker)
**Format**: Verbatim Script (Read-aloud friendly)

---

## Speaker 1: Context & Data (0:00 - 5:00)
**Role**: Sets the stage, defines the problem, and introduces the dataset.

**(0:00) Slide 1: Title Slide**
"Good morning everyone. **We** are excited to present our group project: **'Musical Carbon Dating'**.

In archaeology, scientists use carbon-14 isotopes to determine the history of organic matter. **We** set out to answer a similar question for culture: Can **we** take a raw audio file—without knowing its artist or release date—and predict exactly when it was produced based solely on its acoustic properties?

Today, **we** will walk you through how **we** built a statistical model to capture the 'Arrow of Time' in music."

**(1:00) Slide 2: Table of Contents**
"Here is our roadmap.
I will start by defining the research problem and our dataset.
Then, **Speaker 2** will build our foundation using Linear Regression.
**Speaker 3** will present our rigorous Diagnostic Audit and the issues we found.
**Speaker 4** will reveal the core discovery of our project: the Structural Break of 1999.
Finally, **Speaker 5** will cover Model Selection and our commercial Applications."

**(1:30) Slide 3: The Research Question**
"Our core question is: 'Does music evolve linearly, or are there distinct eras defined by technology?'
We know a Beatles track sounds different from a Dua Lipa track. But can **we** quantify this?

This matters for two reasons.
First, commercially, platforms like Spotify need to understand 'vintage' aesthetics to improve recommendations.
Second, culturally, **we** want to measure the impact of the Digital Revolution. Did the invention of ProTools and MP3s actually change the mathematical structure of music? As **we'll** see, the answer is yes."

**(3:30) Slide 4: The Dataset**
"To answer this, **we** analyzed the Spotify 600k Tracks Dataset.
**We** applied strict filtering criteria:
- **We** selected tracks from 1960 to 2020.
- **We** filtered for 'Popularity > 30' to ensure **we** are analyzing culturally relevant trends.
- This left **us** with a massive sample size of **N = 250,971 tracks**.

Our feature set covers Physical properties (Loudness), Perceptual qualities (Acousticness), and Musical attributes (Key).
A key observation to keep in mind is the strong correlation ($r \approx 0.7$) between Loudness and Energy, hinting at the 'Loudness War' **we'll** discuss next.
Now, I'll hand it over to **Speaker 2** to discuss our initial models."

---

## Speaker 2: Linear Foundations (5:00 - 10:00)
**Role**: Explains the baseline models (SLR & MLR) and initial findings.

**(5:00) Slide 5: Phase II - Simple Linear Regression**
"Thanks, Speaker 1.
**We** began our analysis with the simplest possible model: Phase II.
**We** regressed 'Year' on a single variable: **Loudness**.

The results were statistically highly significant. With a t-statistic of **183.7**, **we** can say with near certainty that music has gotten louder over time.
However, look at the $R^2$. It's only **0.144**.
This tells **us** that while the 'Loudness War' is real—adding about 1.2 years for every decibel—it's not the only driver of evolution."

**(7:30) Slide 6: Phase III - Multiple Linear Regression**
"So, **we** moved to Phase III: The Full Multiple Linear Regression Model.
**We** used a matrix formulation $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ incorporating all 13 features.

This doubled our explanatory power to an $R^2$ of **0.296**, with an error (RMSE) of about **12.2 years**.
**We** started seeing nuanced trends:
- **Acousticness** had a strong negative coefficient (-2.8), reflecting the decline of folk instruments.
- **Danceability** had a massive positive coefficient (+22), showing the rise of rhythmic genres like Disco and Hip-hop.

But an $R^2$ of 0.3 is still quite low. **We** suspected **we** were missing something fundamental.
To investigate why, **Speaker 3** will walk us through the Diagnostics."

---

## Speaker 3: The Diagnostic Audit (10:00 - 15:00)
**Role**: The "Auditor". Discusses assumptions, violations, and physical interpretations.

**(10:00) Slide 7: Diagnostics Overview**
"Thanks, Speaker 2.
Before adding complexity, I performed a rigorous **Diagnostic Audit** of the Gauss-Markov assumptions.
This is where the project gets interesting statistically.

1.  **Linearity**: **We** ran a Partial F-Test adding a quadratic term for Duration. The F-statistic was **304.7**. **We** strongly rejected the null; music evolution isn't linear.
2.  **Multicollinearity**: Good news here. All VIF scores were under 4.0. Our features are distinct.
3.  **Homoscedasticity**: This was the big red flag. The Breusch-Pagan test yielded a massive statistic of **19,391**. **We** found severe heteroscedasticity."

**(12:00) Slide 8: Residual Analysis**
"Let's look at this 'Residuals vs Fitted' plot. Do you see the sharp diagonal edges?
This isn't random noise. It's a **Boundedness Artifact**. Because our data only goes from 1960 to 2020, the residuals are mathematically boxed in. For example, if we predict a song is from 2010, the maximum possible positive error is 10 years, because time stops at 2020. 

Furthermore, our Q-Q plot shows heavy tails. This confirms that music evolution isn't a Gaussian process. Instead of a model failure, we interpret this as **Stylistic Heterogeneity**. Those 'fat tails' represent tracks that sound way older or newer than their release date—the very basis for our **Nostalgia Index**.

> [!TIP]
> **Defense Strategy (If asked about OLS validity):**
> 1. **Boundedness**: OLS is robust as an approximation. Tobit or Beta regression could handle bounds, but OLS provides maximum interpretability for this project.
> 2. **Non-normality**: With $N=250,000$, the **Central Limit Theorem** ensures our coefficient estimates $\hat{\beta}$ are asymptotically normal, making our t-tests and p-values perfectly valid despite non-normal errors."

Now, this structural change in variance hinted at a deeper issue... which **Speaker 4** will now reveal."

---

## Speaker 4: The Structural Break (15:00 - 20:00)
**Role**: The "Historian". Explains the main breakthrough (1999 split) and Interaction effects.

**(15:00) Slide 9: Phase VI - The Digital Revolution**
"Thanks, Speaker 3.
Based on those diagnostics, **we** hypothesized that music history isn't continuous. It was broken by the **Digital Revolution**.
Technologically, 1999 is the year Napster and ProTools changed the industry.

**We** modeled this as a **Structural Break** using interaction terms—effectively allowing every feature to have a different 'slope' before and after 1999.

The result was stunning.
Our $R^2$ jumped from 0.30 to **0.734**.
Our error dropped from 12 years to **7.37 years**.
**We** confirmed that 1999 was a singularity where the 'physics' of music creation changed."

**(17:30) Slide 10: The Scissor Effect**
"The most dramatic evidence is what **we** call the **'Scissor Effect'**.
Look at the plot for **Acousticness**.

Before 1999 (blue line), the slope is steep and negative. Technology forced people to abandon acoustic instruments.
But after 1999 (orange line), the slope flips and becomes positive!

Why? **We** believe that in the digital era, being 'acoustic' became a stylistic *choice* (like Ed Sheeran) rather than a limitation. The relationship inverted.
Now, **Speaker 5** will show how **we** selected our final model and applied it."

---

## Speaker 5: Selection & Applications (20:00 - 25:00)
**Role**: Summarizes model selection, presents the "Nostalgia Index", and concludes.

**(20:00) Slide 11: Phase V - Model Selection**
"Thanks, Speaker 4.
With such a complex model, **we** needed to ensure **we** weren't overfitting.
**We** compared LASSO ($L_1$) and Stepwise (AIC) selection.

**We** chose LASSO. It kept critical features like 'Speechiness' (for the Rap era) and 'Instrumentalness' (for the decline of solos), while zeroing out noise. This gave **us** the most robust model."

**(21:30) Slide 12: The Nostalgia Index**
"Finally, **we** applied this model commercially.
If our model predicts the 'production year', what does it mean when it gets it wrong?
**We** define the **Nostalgia Index** as 'Predicted Year minus Actual Year'.

**We** tested this on verified 'Retro' hits:
1.  **The Weeknd (2012)**: Predicted **1995**. **We** correctly heard the 90s R&B influence.
2.  **Bruno Mars (2017)**: Predicted **2000**, catching the Funk vibe.
3.  **Dua Lipa (2020)**: Predicted **2008**, picking up 80s Synthwave.
This proves our model captures **Genre DNA**."

**(23:30) Slide 13 & 14: Conclusion**
"Looking at our final accuracy plot, you can see how tight the fit becomes after 1999.

To conclude:
1.  **We succeeded**: **We** can date a song to within $\pm 7$ years.
2.  **We discovered**: The 1999 Structural Break is the dominant event in music history.
3.  **We applied**: The Nostalgia Index is a viable metric for recommendation algorithms.

Thank you for your time. **We** are happy to take any questions."
