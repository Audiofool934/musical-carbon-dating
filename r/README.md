# Musical Carbon Dating

## Overview
This directory contains the **R-native implementation** of the Musical Carbon Dating project.

## Structure
- `analysis.R`: The main script. Runs the entire pipeline.
- `report/`: LaTeX source for the academic report.
- `slides/`: LaTeX source for the presentation slides.
- `output_r/`: All generated figures and tables are valid here.

## Usage
1. Ensure `data/tracks.csv` exists in the parent project root.
2. Run the analysis script:
   ```bash
   Rscript analysis.R
   ```
3. Compile the report:
   ```bash
   cd report
   pdflatex report.tex
   ```
