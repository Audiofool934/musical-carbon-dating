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

The results were transformative:
- Our **Weighted R² shot up to 0.763**. This tells us we are capturing 76% of the *reliable, systematic trends* in music history.
- The **F-statistic** is a massive $4.19 \times 10^4$.
- Crucially, this restored validity to our p-values. When we say 'Danceability has a coefficient of +3.8', that number is now statistically defensible."

**(17:00) Slide 14: Phase VI-A - Technological Drivers**
"Our WLS regression revealed the 'Physics' of modern production with high precision.
1.  **The Loudness-Energy Paradox**:
    - With WLS, we found the coefficient for **Loudness** is **+6.51**, but **Energy** is **-1.20**.
    - This is a paradox: Music is getting louder, but *less energetic*.
    - This confirms the **Loudness War**: Dynamic range is crushed by compression. Modern tracks are 'loud' digitally, but lack the explosive dynamic energy or 'punch' of older recordings.

2.  **The Minor-Key Shift (Mode)**:
    - The coefficient for **Mode** is **-0.88**.
    - Since Major is 1 and Minor is 0, a negative coefficient means music is trending towards **Minor keys**. Pop music has become darker and more somber over the decades.

3.  **The Attention Economy**:
    - **Duration** has a coefficient of **-0.89**.
    - **Tempo** has a positive coefficient of **+0.68**.
    - We are moving faster, but for less time. The streaming era is literally shortening our attention spans and speeding up the pulse of culture."
