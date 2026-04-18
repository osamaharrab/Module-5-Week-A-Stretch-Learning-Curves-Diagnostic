# Learning Curves Diagnostic Analysis

## Dataset & Model Configuration
- **Dataset**: Telecom Churn (n=1500, churn rate=16.3%)
- **Model**: Logistic Regression with `class_weight='balanced'`
- **Cross-validation**: Stratified 5-Fold

---

## Metric Selection Justification

**ROC-AUC was chosen as the primary metric because:**

1. The dataset has significant class imbalance (83.7% vs 16.3%)
2. ROC-AUC considers both sensitivity and specificity
3. It's threshold-independent, providing a holistic view of model performance
4. F1 score was also computed for reference (focuses on positive class)

---

## Results

| Metric | Training Score | Validation Score | Gap |
|--------|----------------|------------------|-----|
| ROC-AUC | 0.7095 (±0.0089) | 0.6677 (±0.0297) | 0.0417 |
| F1 | 0.3750 (±0.0055) | 0.3510 (±0.0234) | 0.0240 |

---

## Diagnosis

**Assessment: HIGH BIAS (Underfitting)**

---

## Written Analysis

Based on the learning curve shape, the logistic regression model exhibits characteristics of **high bias (underfitting)**. Both training and validation scores converge at a modest level (~0.67 ROC-AUC), with a small gap between them. This pattern indicates the model lacks sufficient complexity to capture the underlying patterns in the data. The model is underfitting.

### Q: Would collecting more data help?
**A: No** — Since both curves have converged, adding more data would not significantly improve validation performance. The bottleneck is model capacity, not data quantity.

### Q: Would increasing model complexity help?
**A: Yes** — The model is too simple for this problem. Adding polynomial features, interaction terms, or switching to a more flexible algorithm (Random Forest, Gradient Boosting) would likely improve performance.

### Q: What is your recommended next step for improving this model?
**A: Increase model complexity** by engineering polynomial or interaction features, or switch to a non-linear model like Random Forest or XGBoost. The current linear decision boundary appears insufficient for capturing the complex relationships in customer churn behavior.

---

## Conclusion

The learning curves clearly demonstrate that the logistic regression model is underfitting the telecom churn data. The small gap between training and validation curves (~0.04) indicates low variance, while the moderate performance ceiling (~0.67 ROC-AUC) indicates high bias. The model has plateaued and cannot extract more signal from the existing data. The recommended action is to increase model complexity rather than collect more data.
