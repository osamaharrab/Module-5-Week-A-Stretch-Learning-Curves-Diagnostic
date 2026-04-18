"""
Learning Curves Diagnostic for Telecom Churn Prediction
Analyzes bias-variance tradeoff using sklearn's learning_curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('telecom_churn.csv')

# Separate features and target
X = df.drop(['customer_id', 'churned'], axis=1)
y = df['churned']

print(f"Dataset shape: {df.shape}")
print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"Churn rate: {y.mean()*100:.1f}%")

# Define feature types
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'num_support_calls', 'senior_citizen']
categorical_features = ['gender', 'contract_type', 'internet_service', 'payment_method', 'has_partner', 'has_dependents']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Create logistic regression pipeline with balanced class weights
# Using balanced class weights to handle class imbalance
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Define training sizes for learning curve
train_sizes = np.linspace(0.1, 1.0, 10)  # 10 different sizes

# Use stratified K-fold for cross-validation (important for class imbalance)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nComputing learning curves with ROC-AUC scoring...")
print("(ROC-AUC is appropriate for imbalanced datasets as it considers both sensitivity and specificity)")

# Compute learning curves using ROC-AUC (appropriate for imbalanced data)
train_sizes_abs, train_scores_auc, val_scores_auc = learning_curve(
    model, X, y,
    train_sizes=train_sizes,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    return_times=False
)

# Also compute learning curves with F1 score for comparison
print("Computing learning curves with F1 scoring...")
train_sizes_abs_f1, train_scores_f1, val_scores_f1 = learning_curve(
    model, X, y,
    train_sizes=train_sizes,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    return_times=False
)

# Calculate mean and std for ROC-AUC
train_mean_auc = np.mean(train_scores_auc, axis=1)
train_std_auc = np.std(train_scores_auc, axis=1)
val_mean_auc = np.mean(val_scores_auc, axis=1)
val_std_auc = np.std(val_scores_auc, axis=1)

# Calculate mean and std for F1
train_mean_f1 = np.mean(train_scores_f1, axis=1)
train_std_f1 = np.std(train_scores_f1, axis=1)
val_mean_f1 = np.mean(val_scores_f1, axis=1)
val_std_f1 = np.std(val_scores_f1, axis=1)

# Create the learning curve plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ROC-AUC Learning Curve
ax1 = axes[0]
ax1.fill_between(train_sizes_abs, train_mean_auc - train_std_auc,
                  train_mean_auc + train_std_auc, alpha=0.1, color='blue')
ax1.fill_between(train_sizes_abs, val_mean_auc - val_std_auc,
                  val_mean_auc + val_std_auc, alpha=0.1, color='orange')
ax1.plot(train_sizes_abs, train_mean_auc, 'o-', color='blue',
         label='Training score', linewidth=2)
ax1.plot(train_sizes_abs, val_mean_auc, 'o-', color='orange',
         label='Cross-validation score', linewidth=2)
ax1.set_xlabel('Training Set Size', fontsize=11)
ax1.set_ylabel('ROC-AUC Score', fontsize=11)
ax1.set_title('Learning Curve (ROC-AUC)\nLogistic Regression on Telecom Churn', fontsize=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.5, 1.0])

# Add gap annotation
gap_auc = train_mean_auc[-1] - val_mean_auc[-1]
ax1.annotate(f'Gap: {gap_auc:.3f}',
             xy=(train_sizes_abs[-1], (train_mean_auc[-1] + val_mean_auc[-1])/2),
             fontsize=10, ha='left')

# Plot 2: F1 Learning Curve
ax2 = axes[1]
ax2.fill_between(train_sizes_abs, train_mean_f1 - train_std_f1,
                  train_mean_f1 + train_std_f1, alpha=0.1, color='blue')
ax2.fill_between(train_sizes_abs, val_mean_f1 - val_std_f1,
                  val_mean_f1 + val_std_f1, alpha=0.1, color='orange')
ax2.plot(train_sizes_abs, train_mean_f1, 'o-', color='blue',
         label='Training score', linewidth=2)
ax2.plot(train_sizes_abs, val_mean_f1, 'o-', color='orange',
         label='Cross-validation score', linewidth=2)
ax2.set_xlabel('Training Set Size', fontsize=11)
ax2.set_ylabel('F1 Score', fontsize=11)
ax2.set_title('Learning Curve (F1 Score)\nLogistic Regression on Telecom Churn', fontsize=12)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.0])

# Add gap annotation
gap_f1 = train_mean_f1[-1] - val_mean_f1[-1]
ax2.annotate(f'Gap: {gap_f1:.3f}',
             xy=(train_sizes_abs[-1], (train_mean_f1[-1] + val_mean_f1[-1])/2),
             fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('learning_curve.pdf', bbox_inches='tight')
print("\nLearning curve plot saved as 'learning_curve.png' and 'learning_curve.pdf'")

# Print diagnostic summary
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print(f"\nFinal Training ROC-AUC: {train_mean_auc[-1]:.4f} (+/- {train_std_auc[-1]:.4f})")
print(f"Final Validation ROC-AUC: {val_mean_auc[-1]:.4f} (+/- {val_std_auc[-1]:.4f})")
print(f"Gap (Training - Validation): {gap_auc:.4f}")
print(f"\nFinal Training F1: {train_mean_f1[-1]:.4f} (+/- {train_std_f1[-1]:.4f})")
print(f"Final Validation F1: {val_mean_f1[-1]:.4f} (+/- {val_std_f1[-1]:.4f})")
print(f"Gap (Training - Validation): {gap_f1:.4f}")

# Determine bias-variance status
print("\n" + "="*70)
print("BIAS-VARIANCE ANALYSIS")
print("="*70)

# Check if curves have converged
early_gap = train_mean_auc[3] - val_mean_auc[3]
late_gap = train_mean_auc[-1] - val_mean_auc[-1]
convergence = (early_gap - late_gap) / early_gap if early_gap > 0 else 0

print(f"\nEarly gap (at ~25% data): {early_gap:.4f}")
print(f"Final gap (at 100% data): {late_gap:.4f}")
print(f"Gap reduction: {convergence*100:.1f}%")

if val_mean_auc[-1] < 0.75 and late_gap < 0.05:
    diagnosis = "HIGH BIAS (Underfitting)"
    recommendation = "Increase model complexity: add polynomial features, use more flexible models (e.g., Random Forest, XGBoost), or engineer better features."
elif late_gap > 0.10:
    diagnosis = "HIGH VARIANCE (Overfitting)"
    recommendation = "Collect more data, increase regularization, or reduce model complexity."
else:
    diagnosis = "BALANCED (Good fit)"
    recommendation = "Model is performing well. Consider fine-tuning hyperparameters for marginal improvements."

print(f"\nDiagnosis: {diagnosis}")
print(f"Recommendation: {recommendation}")

# Save analysis results to a text file
with open('learning_curve_analysis.txt', 'w') as f:
    f.write("LEARNING CURVES DIAGNOSTIC ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write("Dataset: Telecom Churn (n=1500, churn rate=16.3%)\n")
    f.write("Model: Logistic Regression with class_weight='balanced'\n")
    f.write("Cross-validation: Stratified 5-Fold\n\n")

    f.write("METRIC SELECTION JUSTIFICATION\n")
    f.write("-"*40 + "\n")
    f.write("ROC-AUC was chosen as the primary metric because:\n")
    f.write("1. The dataset has significant class imbalance (83.7% vs 16.3%)\n")
    f.write("2. ROC-AUC considers both sensitivity and specificity\n")
    f.write("3. It's threshold-independent, providing a holistic view of model performance\n")
    f.write("4. F1 score was also computed for reference (focuses on positive class)\n\n")

    f.write("RESULTS\n")
    f.write("-"*40 + "\n")
    f.write(f"Final Training ROC-AUC: {train_mean_auc[-1]:.4f} (+/- {train_std_auc[-1]:.4f})\n")
    f.write(f"Final Validation ROC-AUC: {val_mean_auc[-1]:.4f} (+/- {val_std_auc[-1]:.4f})\n")
    f.write(f"Training-Validation Gap: {gap_auc:.4f}\n\n")

    f.write(f"Final Training F1: {train_mean_f1[-1]:.4f} (+/- {train_std_f1[-1]:.4f})\n")
    f.write(f"Final Validation F1: {val_mean_f1[-1]:.4f} (+/- {val_std_f1[-1]:.4f})\n")
    f.write(f"Training-Validation Gap: {gap_f1:.4f}\n\n")

    f.write("DIAGNOSIS\n")
    f.write("-"*40 + "\n")
    f.write(f"Assessment: {diagnosis}\n\n")

    f.write("WRITTEN ANALYSIS\n")
    f.write("-"*40 + "\n")
    f.write("Based on the learning curve shape, the logistic regression model exhibits\n")
    f.write(f"characteristics of {diagnosis.lower()}. ")

    if "HIGH BIAS" in diagnosis:
        f.write(f"Both training and validation scores converge at a modest level (~{val_mean_auc[-1]:.2f} ROC-AUC),\n")
        f.write("with a small gap between them. This pattern indicates the model lacks sufficient\n")
        f.write("complexity to capture the underlying patterns in the data. The model is underfitting.\n\n")
        f.write("Q: Would collecting more data help?\n")
        f.write("A: No - since both curves have converged, adding more data would not significantly\n")
        f.write("   improve validation performance. The bottleneck is model capacity, not data quantity.\n\n")
        f.write("Q: Would increasing model complexity help?\n")
        f.write("A: Yes - the model is too simple for this problem. Adding polynomial features,\n")
        f.write("   interaction terms, or switching to a more flexible algorithm (Random Forest,\n")
        f.write("   Gradient Boosting) would likely improve performance.\n\n")
        f.write("RECOMMENDED NEXT STEP: Increase model complexity by engineering polynomial or\n")
        f.write("interaction features, or switch to a non-linear model like Random Forest or XGBoost.\n")
    elif "HIGH VARIANCE" in diagnosis:
        f.write("There is a large gap between training and validation scores, indicating the model\n")
        f.write("is memorizing training data rather than generalizing. The model is overfitting.\n\n")
        f.write("Q: Would collecting more data help?\n")
        f.write("A: Yes - the validation curve is still improving at the maximum dataset size.\n")
        f.write("   More data would help the model generalize better.\n\n")
        f.write("Q: Would increasing model complexity help?\n")
        f.write("A: No - the model is already too complex. Reducing complexity through regularization\n")
        f.write("   or feature selection would be more beneficial.\n\n")
        f.write("RECOMMENDED NEXT STEP: Increase regularization strength (C parameter) or collect\n")
        f.write("more training data.\n")
    else:
        f.write("The curves show a reasonable gap and good validation performance.\n")
        f.write("The model strikes a good balance between bias and variance.\n\n")
        f.write("RECOMMENDED NEXT STEP: Fine-tune hyperparameters for marginal improvements.\n")

print("\nAnalysis saved to 'learning_curve_analysis.txt'")

plt.show()
