## Speaker 4: The Solution (14:00 - 19:30)
**Role**: Explains Model Selection, WLS, and Technological Drivers.

**(14:00) Slide 12: Phase IV - Model Selection**
"Thanks, Speaker 3.
Before fixing the variance issue, we wanted to ensure our variable list was solid.
We compared two methods:
1.  **Stepwise AIC**: This aggressive approach suggested dropping 'Key' to minimize information loss.
2.  **LASSO**: This regularization technique ($L_1$ penalty) actually kept **all 13 features**.

We prioritized the LASSO result. Why? Because music theory tells us that subtle features like 'Key' and 'Mode' *do* matter. The shift from complex modulations in 70s Jazz Fusion to the loops of modern Pop is real, even if the signal is weak. We decided our model needs every acoustic signal it can get."

**(15:30) Slide 13: Phase V - Weighted Least Squares (WLS)**
"Now for the heteroscedasticity fix.
We implemented **Weighted Least Squares (WLS)**.
Mathematically, we weighted every observation by the inverse of its variance ($w_i = 1/\sigma^2_i$). This effectively down-weights the 'noisy' modern era and up-weights the 'consistent' classic era.

The results were transformative for **validity**, if not raw power:
- Our **Weighted R² is 0.275**. This confirms that while the *trends* are real, the *variance* in music is massive. We aren't predicting hit songs perfectly; we are modeling the flow of history.
- The **F-statistic** is a strong $4.94 \times 10^3$.
- Crucially, this restored validity to our p-values. When we say 'Danceability has a coefficient of +3.9', that number is now statistically defensible."

**(17:00) Slide 14: Phase VI-A - Technological Drivers**
"Our WLS regression revealed the 'Physics' of modern production with high precision.
1.  **The Loudness-Energy Paradox**:
    - With WLS, we found the coefficient for **Loudness** is **+8.19**, but **Energy** is **-2.24**.
    - This is a paradox: Music is getting louder, but *less energetic*.
    - This confirms the **Loudness War**: Dynamic range is crushed by compression. Modern tracks are 'loud' digitally, but lack the explosive dynamic energy or 'punch' of older recordings.

2.  **The Minor-Key Shift (Mode)**:
    - The coefficient for **Mode** is **-0.78**.
    - Since Major is 1 and Minor is 0, a negative coefficient means music is trending towards **Minor keys**. Pop music has become darker and more somber over the decades.

3.  **The Attention Economy**:
    - **Duration** has a coefficient of **-0.75**.
    - **Tempo** has a positive coefficient of **+0.99**.
    - We are moving faster, but for less time. The streaming era is literally shortening our attention spans and speeding up the pulse of culture."
