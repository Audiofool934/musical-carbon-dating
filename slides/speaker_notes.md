# Speaker Notes: Musical Carbon Dating (Group Presentation - 5 Speakers)
**Target Time**: 22 Minutes (approx. 4-5 mins per speaker)
**Format**: Verbatim Script (Read-aloud friendly)

---

## Speaker 1: Context & Data (0:00 - 4:00)
**Role**: Sets the stage, defines the problem as "Feature Recognition," and introduces the dataset.

**(0:00) Slide 1: Title Slide**
"Good morning everyone. **We** are excited to present our group project: **'Musical Carbon Dating'**.

In archaeology, scientists use Carbon-14 to date organic matter. **We** asked a similar question for culture: Can **we** take a raw audio file—without knowing its artist or release date—and predict exactly when it was produced based solely on its acoustic properties?

Today, **we** will show you how **we** built a statistical model to recognize the 'acoustic signature' of every era from 1960 to 2020."

**(1:00) Slide 2: Table of Contents**
"Here is our roadmap.
I will start by defining the research problem and our dataset.
Then, **Speaker 2** will walk us through our Regression Pipeline, from simple models to complex ones.
**Speaker 3** will present our rigorous Diagnostic Audit—where we found and fixed critical statistical violations.
**Speaker 4** will explain our final solution: Weighted Least Squares and Model Selection.
Finally, **Speaker 5** will demonstrate our commercial application: The Nostalgia Index."

**(1:30) Slide 3: The Research Question**
"Our core objective is **Feature Recognition**.
Unlike stock prices which depend on history, a song's features are intrinsic. A 1970s rock song has a specific 'sound'—dry drums, warm bass—that is distinct from a 2010s pop song with digital sub-bass and auto-tune.

Our goal is to quantify these differences.
Why does this matter?
1.  **Archival**: Imagine discovering a lost tape in an attic. Our model could date it instantly.
2.  **Commercial**: Recommendation algorithms need to understand 'Vibe'. A user listening to 80s music might also like a modern song if it *sounds* like the 80s."

**(3:00) Slide 4: The Dataset**
"To build this, **we** analyzed the Spotify 600k Tracks Dataset.
We applied strict quality control:
- **Timeframe**: 1960 to 2020.
- **Filter**: Popularity > 30. We focus on culturally relevant music to capture the 'sound of the times'.
- **Sample Size**: This left us with **N = 250,971 tracks**.

Our features ($p=13$) cover the full spectrum:
- **Physical**: Loudness, Tempo, Duration.
- **Musical**: Key, Mode, Time Signature.
- **Perceptual**: Acousticness, Energy, Valence.

Now, I'll hand it over to **Speaker 2** to start our modeling journey."

---

## Speaker 2: Initial Models (4:00 - 8:00)
**Role**: Explains the baseline models (SLR & MLR) and the initial results.

**(4:00) Slide 5: Methodology Overview**
"Thanks, Speaker 1.
Our methodology treats this as a regression problem. We used a **Random Split (80/20)** strategy.
Why random split? Because we are testing the model's ability to 'recognize' the era of any given song, much like an art historian identifying a painting's period by its style. We want our test set to cover all decades equally."

**(5:00) Slide 6: Phase I - The 'Loudness War'**
"We started simple: Phase I.
We regressed 'Year' on a single variable: **Loudness**.
The results confirmed a famous phenomenon known as the 'Loudness War'.
- Coefficient: **1.2**. This means for every decibel louder a song is, it sounds about 1.2 years 'newer'.
- Significance: Extremely high ($t=183.7$).

However, the $R^2$ is only **0.144**. Loudness tells us *something*—music got louder—but it's a blunt instrument."

**(6:30) Slide 7: Phase II - Multiple Linear Regression**
"So, we moved to Phase II: The Full Model.
We included all 13 features.
$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

This baseline model achieved an $R^2$ of **0.296**.
Our error (RMSE) dropped to about **12 years**.
We started seeing nuanced trends:
- **Acousticness**: Strong negative trend. As time goes on, acoustic instruments are replaced by synthesizers.
- **Danceability**: Positive trend. Music has become more rhythmic.

But an $R^2$ of 0.3 means 70% of the variance is still unexplained. More importantly, we suspected our model was technically invalid.
To investigate, **Speaker 3** will perform the Audit."

---

## Speaker 3: The Audit (8:00 - 12:30)
**Role**: The "Auditor". Discusses diagnostics, heteroscedasticity, and physical interpretations of residuals.

**(8:00) Slide 8: Diagnostic Audit**
"Thanks, Speaker 2.
Before trying to improve accuracy, we must check validity. I ran a rigorous **Diagnostic Audit** of the Gauss-Markov assumptions.
This is where the project gets interesting statistically.

1.  **Linearity**: We ran a Partial F-Test ($F \approx 305$). We soundly rejected linearity. Music evolution is complex and non-linear.
2.  **Multicollinearity**: Good news. All VIF scores were under 4.0, even with the Loudness/Energy correlation. Our features are distinct.
3.  **Homoscedasticity**: This was the critical failure. The Breusch-Pagan test statistic was **over 22,000**. The variance is NOT constant."

**(10:00) Slide 9: Residual Analysis**
"Let's look at the plots.
The Residuals vs Fitted plot shows sharp diagonal boundaries—an artifact of our time-bounded data (1960-2020).
The Q-Q plot shows heavy tails. This confirms 'Stylistic Heterogeneity'—some songs are naturally retro or futuristic outliers.
This variance instability kills standard OLS. We need a fix."

---

## Speaker 4: The Solution (12:30 - 17:00)
**Role**: Explains Model Selection and WLS.

**(12:30) Slide 10: Phase IV - Model Selection**
"Thanks, Speaker 3.
Before fixing validity, we fixed structure. We ran **Model Selection** to ensure parsimony.
We compared **Stepwise AIC** vs **LASSO**.
- Stepwise suggested dropping 'Key'.
- **LASSO** ($L_1$ regularization) kept **all 13 features**.

We chose to stick with the full LASSO set.
Why? Because subtle features like 'Key' and 'Mode' matter. The shift from complex modulations in 70s Jazz Fusion to the loops of modern Pop is real. Our model needs every acoustic signal it can get."

**(14:00) Slide 11: Phase V - Weighted Least Squares (WLS)**
"Now for the heteroscedasticity fix.
We implemented **Weighted Least Squares**.
We weighted every observation inversely to its variance.

**But here's the key insight**: WLS doesn't make our *predictions* more accurate. It makes our *inference* valid.
- The **Weighted R² = 0.77** tells us we're capturing 77% of the *reliable trends* in music evolution.
- But the test set still contains 'wild' songs—retro throwbacks, experimental outliers—that no model can perfectly predict.

What WLS *does* fix:
- Our **p-values** are now trustworthy.
- Our **confidence intervals** are correct.
- When we say 'Danceability has coefficient +24', that number is now statistically defensible."

**(16:00) Slide 12: Phase VI-A - Technological Drivers**
"Our regression also revealed the 'Physics' of modern production.
Two massive insights here:
1.  **The Loudness-Energy Paradox**:
    - We know music is getting louder.
    - BUT, our model shows that for the *same loudness*, modern songs have **lower Energy** ($\beta \approx -5.4$).
    - This is the signature of the **Loudness War**: Dynamic range is crushed by compression. It's 'loud', but not energetic.

2.  **The Attention Economy**:
    - Duration is dropping statistically. The streaming era is literally shortening our attention spans."

**(17:00) Slide 13: Phase VI-B - Cultural Evolution**
"And finally, the 'Psychology' of the era.
1.  **The 'Sad Banger'**:
    - Danceability is WAY up (+24).
    - Valence (Optimism) is WAY down (-16).
    - We are dancing more, but feeling less.

2.  **The Acousticness Paradox**:
    - Everyone thinks acoustic music died.
    - Wrong. If you control for Loudness, acousticness is actually **positive** (+2.08) in our model.
    - That means once you strip away the compression, modern Indie and Lo-Fi are keeping the acoustic spirit alive. Simple correlation missed this; WLS found it."

---

## Speaker 5: Applications & Conclusion (18:00 - 22:00)
**Role**: Post-Model Analysis. Nostalgia Index. Note: Slide numbers shifted due to deep dive.

**(18:00) Slide 14: The Nostalgia Index**
"Thanks, Speaker 4.
So we have a model that works. But what about when it 'fails'?
If our model predicts a 2020 song was made in 1980, is the model wrong? Or is the *song* Retro?

We define the **Nostalgia Index** as: $| \text{Predicted Year} - \text{Actual Year} |$.
A high index means a song is 'Time-Displaced'."

**(18:30) Slide 15: Validation**
"We validated this on verified tracks.
**Dua Lipa's 'Physical'** (2020) sounds like 1980s aerobics music.
Our model predicted **2009**. It sensed the 'old' features, giving it a high Nostalgia Index of **11 years**.
**The Weeknd's 'Blinding Lights'** has an even higher index (14.8 years) due to its 80s synth-pop style.

This proves the model can distinguish 'release date' from 'aesthetic date'."

**(20:30) Slide 16: Conclusion**
"To wrap up:

1.  **Feasibility**: We can date music to within **±9 years** purely from acoustics.
2.  **Rigor**: OLS was invalid (heteroscedasticity). WLS gave us **trustworthy p-values and coefficients**.
3.  **Transparency**: The Weighted R² of 0.77 measures *trend capture*, not prediction. Our practical accuracy is **MAE = 9 years**.
4.  **Application**: The Nostalgia Index turns residuals into a commercial metric.

Thank you. We are happy to take questions."
