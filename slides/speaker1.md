## Speaker 1: Context & Data (0:00 - 4:15)
**Role**: Sets the stage, defines the problem as "Feature Recognition," and introduces the dataset.

**(0:00) Slide 1: Title Slide**
"Good morning everyone. **We** are excited to present our group project: **'Musical Carbon Dating'**.

In archaeology, scientists use Carbon-14 to date organic matter based on its decay. **We** asked a similar question for culture: Can **we** take a raw audio file, without knowing its artist or release date, and predict exactly when it was produced based solely on its acoustic properties?

Today, **we** will show you how **we** built a statistical model to recognize the 'acoustic signature' of every era from 1960 to 2020. This is not just about prediction—it's about understanding the 'Arrow of Time' in music history."

**(0:50) Slide 2: Project Logo**
(Pause for visual impact of the project identity)

**(1:00) Slide 3: Table of Contents**
"Here is our roadmap for the next 20 minutes.
I will start by defining the research problem and our dataset.
Then, **Speaker 2** will walk you through our Regression Pipeline, from simple models to the Phase II baseline.
Speaker 3 will present our rigorous Diagnostic Audit—this is the core of our statistical work, where we found and fixed critical violations like heteroscedasticity.
Speaker 4 will explain our final solution: Weighted Least Squares and Model Selection.
Finally, Speaker 5 will demonstrate our practical application: The Nostalgia Index.

**(1:45) Slide 4: The Research Question**
"Our core objective is **Feature Recognition**.
Think about it: unlike stock prices which depend on history, a song's features are intrinsic. A 1970s rock song has a specific 'sound'—dry drums, warm bass, high dynamic range—that is distinct from a 2010s pop song with digital sub-bass and heavy auto-tune.

Our goal is to quantify these differences statistically.
Why does this matter?
1.  **Archival**: Imagine discovering a lost tape in an attic. Our model could date it instantly.
2.  **Commercial**: Recommendation algorithms need to understand 'Vibe'. A user listening to 80s music might also like a modern song if it *sounds* like the 80s. We want to measure that similarity mathematically."

**(3:00) Slide 5: The Dataset**
"To build this, **we** analyzed the Spotify 600k Tracks Dataset.
We applied strict quality control to ensure our model learns 'Culture', not noise:
- **Timeframe**: 1960 to 2020. This covers the modern pop era.
- **Filter**: We filtered for `popularity > 30`. This is crucial. We focus on culturally relevant music to capture the 'sound of the times', rather than obscure garage demos that might not reflect the era's production standards.
- **Sample Size**: This left us with a massive dataset of **N = 250,971 tracks**.

Our features ($p=13$) cover the full spectrum of audio engineering:
- **Physical**: Loudness (dB), Tempo (BPM), Duration (ms). These are objective measurements.
- **Musical**: Key, Mode (Major/Minor), Time Signature.
- **Perceptual**: Acousticness, Energy, Valence (Positivity), Danceability. These are high-level algorithms provided by Spotify.

Now, I'll hand it over to **Speaker 2** to start our modeling journey."