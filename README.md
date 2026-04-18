# Module 5 Week A — Stretch: Learning Curves Diagnostic

This repository contains a learning-curve diagnostic for a logistic regression model on the telecom churn dataset.

## Files

- `learning_curve_diagnostic.py` — builds the preprocessing + logistic regression pipeline, computes learning curves, and saves the plot
- `learning_curve_plot.png` — training and validation ROC-AUC learning curve with standard deviation bands
- `analysis.md` — short written interpretation of the bias-variance diagnostic
- `requirements.txt` — Python dependencies

## Why ROC-AUC?

The target is imbalanced, so accuracy is not the best metric. ROC-AUC is more appropriate because it measures ranking/separation quality across thresholds.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python learning_curve_diagnostic.py
```
