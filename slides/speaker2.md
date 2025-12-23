## Speaker 2: Initial Models (4:15 - 9:00)
**Role**: Explains the baseline models (SLR & MLR) and the initial results.

**(4:15) Slide 7: Methodology Overview**
"Thanks, Speaker 1.
Our methodology treats this as a managed regression problem. We utilized a **Random Split (80/20)** strategy for validation.
Why random split? Because we are testing the model's ability to 'recognize' the era of any given song, much like an art historian identifying a painting's period by its style. We want our test set to cover all decades equally, rather than treating this as a time-series forecasting problem.

Crucially, we also applied **Data Standardization** (Z-Scores). Since our features have vastly different units—Loudness is in negative decibels (-60 to 0), while Valence is a probability (0 to 1)—we normalized them. This ensures that when we look at our regression coefficients later, their varying magnitudes actually reflect their statistical power, not just their scale."

**(5:40) Slide 8: Phase I - The 'Loudness War'**
"We started simple with Phase I: Simple Linear Regression.
We regressed 'Year' on a single variable: **Loudness**.
The results confirmed a famous phenomenon in audio engineering known as the 'Loudness War'.
- The coefficient is **~1.2**. This means for every standard deviation increase in loudness, the track is predicted to be 1.2 years 'newer'.
- The significance is overwhelming with a $t$-statistic of **159.4**.
- However, the $R^2$ is only **0.130**.

Why is this important? It proves that while music has undeniably gotten louder due to digital limiting, Loudness alone is a 'blunt instrument'. It can tell you if a song is pre- or post-1995, but it fails to capture the cultural nuance between, say, the Disco era and the Grunge era."

**(7:00) Slide 9: Phase II - Multiple Linear Regression**
"So, we moved to Phase II: The Full Model.
We included all 13 features in a standard Ordinary Least Squares (OLS) model.
$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

This baseline model achieved an $R^2$ of **0.238** and an RMSE of **12.05 years**.
Here, we started to see specific feature contributions:
- **Acousticness**: Strong negative trend. As time goes on, acoustic instruments are replaced by synthesizers.
- **Danceability**: Positive trend. Music has become significantly more rhythmic.
- **Standard Errors**: We noticed our standard errors were alarmingly small ($< 0.02$). In OLS, artificially small standard errors often signal that the model is 'overconfident' because it ignores the changing noise levels in the data.

We realized an $R^2$ of 0.24—leaving 76% of variance unexplained—wasn't just a performance issue; it was a validity issue. The model was assuming music history is a straight, predictable line, but we know culture is chaotic.
To investigate this, **Speaker 3** will perform the Audit."

Crucially, we also applied **Data Standardization** (Z-Scores). Since our features have vastly different units—Loudness is in negative decibels (-60 to 0), while Valence is a probability (0 to 1)—we normalized them. This ensures that when we look at our regression coefficients later, their varying magnitudes actually reflect their statistical power, not just their scale."

**(5:40) Slide 7: Phase I - The 'Loudness War'**
"We started simple with Phase I: Simple Linear Regression.
We regressed 'Year' on a single variable: **Loudness**.
The results confirmed a famous phenomenon in audio engineering known as the 'Loudness War'.
- The coefficient is **~1.2**. This means for every standard deviation increase in loudness, the track is predicted to be 1.2 years 'newer'.
- The significance is overwhelming with a $t$-statistic of **159.4**.
- However, the $R^2$ is only **0.130**.

Why is this important? It proves that while music has undeniably gotten louder due to digital limiting, Loudness alone is a 'blunt instrument'. It can tell you if a song is pre- or post-1995, but it fails to capture the cultural nuance between, say, the Disco era and the Grunge era."

**(7:00) Slide 8: Phase II - Multiple Linear Regression**
"So, we moved to Phase II: The Full Model.
We included all 13 features in a standard Ordinary Least Squares (OLS) model.
$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

This baseline model achieved an $R^2$ of **0.238** and an RMSE of **12.05 years**.
Here, we started to see specific feature contributions:
- **Acousticness**: Strong negative trend. As time goes on, acoustic instruments are replaced by synthesizers.
- **Danceability**: Positive trend. Music has become significantly more rhythmic.
- **Standard Errors**: We noticed our standard errors were alarmingly small ($< 0.02$). In OLS, artificially small standard errors often signal that the model is 'overconfident' because it ignores the changing noise levels in the data.

We realized an $R^2$ of 0.24—leaving 76% of variance unexplained—wasn't just a performance issue; it was a validity issue. The model was assuming music history is a straight, predictable line, but we know culture is chaotic.
To investigate this, **Speaker 3** will perform the Audit."
