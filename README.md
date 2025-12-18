# Musical Carbon Dating 🎵
**A Structural Break Analysis of Music Evolution (1960-2020)**

> *"Does the 'Arrow of Time' exist in music production?"*

This project performs a rigorous statistical analysis on over **250,000 tracks** to quantify how music production has evolved over the last 60 years. By identifying a critical **Structural Break in 1999**, we prove that the relationship between audio features (like *Acousticness*) and time is not linear, but underwent a paradigm shift during the Digital Revolution.

---

## 📊 Key Findings

### 1. The 1999 Singularity
We identified a massive structural break in music history around **1999** (Napster/ProTools era).
- **Baseline Model**: $R^2 \approx 0.30$ (Blind prediction).
- **Structural Break Model**: $R^2 \rightarrow 0.73$ (Explanatory power).

### 2. The "Scissor Effect" (Unified Trend)
Our **Unified LOWESS Analysis** reveals a "check-mark" trend in features like *Acousticness*:
- **Pre-1999 (Analog Decline)**: Strong negative variance. Technology replaced acoustic instruments.
- **Post-1999 (Digital Choice)**: Variance flips. Acousticness becomes a *stylistic choice* rather than a limitation.
- *See `output/figures/scissor_plot.png` for the unified visualization.*

### 3. The "Nostalgia Index"
We define prediction error as a commercial metric: $\mathcal{N}_i = \hat{y}_{blind} - y_{actual}$.
- **Insight**: Large negative $\mathcal{N}$ values identify modern songs with "Vintage DNA".
- **Verified Hits**: *Uptown Funk* (-2 yrs), *Physical* (-11 yrs), *Echoes of Silence* (-19 yrs).

---

## 🛠 Project Structure

```text
├── data/                   # Dataset
├── src/                    # Source Code
│   ├── analysis.py         # Regression Engine (OLS, WLS, Chow Test)
│   ├── config.py           # Central Configuration
│   ├── visualization.py    # Plotting Logic (Unified Scissor Plot)
│   └── ...
├── scripts/                # Utility Scripts
│   └── check_robustness.py # "Nostalgia Index" Validator
├── report/                 # [FINAL] Formal LaTeX Report
│   ├── main.tex Source
│   └── figures/            # Generated Figures for Report
├── slides/                 # Presentation Slides (Beamer)
├── output/                 # Generated Artifacts (Logs & Figures)
└── main.py                 # Main Execution Pipeline
```

---

## 🚀 Usage

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Main Analysis
Execute the full pipeline (Data Load -> EDA -> OLS -> Diagnostics -> Structural Break -> Plots):
```bash
python3 main.py
```
*Check `output/pipeline.log` for statistical summaries.*

### 3. "Carbon Date" Specific Tracks
Run the robustness script to test the "Nostalgia Index" on modern hits:
```bash
python3 scripts/check_robustness.py
```

---

## 📄 Documentation

- **Full Report**: [report/main.tex](report/main.tex) (LaTeX Source)
- **Presentation**: [slides/main.tex](slides/main.tex) (Beamer Source)
- **Walkthrough**: See `walkthrough.md` for a step-by-step project tour.

---
*University Statistical Analysis Project | Term: Fall 2024*