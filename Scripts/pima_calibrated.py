# %%
from sklearn.calibration import CalibratedClassifierCV
import joblib


# %%
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# %%
SEED = 42
np.random.seed(SEED)

# NOTE: For training from scratch, download the PIMA dataset and place in ../Datasets/
# For inference only, this file is not needed - only the trained model is used.
DATA_PATH = Path("../Datasets/diabetes.csv")

# %%
df = pd.read_csv(DATA_PATH)
print(df.head())
print("\nShape:", df.shape)

# %%
cols_zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

df[cols_zero_as_missing] = df[cols_zero_as_missing].replace(0, np.nan)

print(df.isna().sum())

# %%
FEATURE_COLS = [c for c in df.columns if c != "Outcome"]
TARGET_COL = "Outcome"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].astype(int).values

# First: hold out a test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED,
)

print("Train+Val shape:", X_train_full.shape, "Test shape:", X_test.shape)

# Second: split train_full into train and calibration sets
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,          # 20% of train_full for calibration
    stratify=y_train_full,
    random_state=SEED,
)

print("Train shape:", X_train.shape, "Calibration shape:", X_calib.shape)



# %%
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,        # lower lr
    "num_leaves": 31,             # maybe 31–63
    "max_depth": -1,
    "min_child_samples": 10,      # a bit smaller
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "n_estimators": 2000,         # rely on early stopping
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",   # you can experiment with this
}


# %%
def train_and_eval_lightgbm(X_tr, y_tr, X_va, y_va, params, early_stopping_rounds=100):
    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(period=0)  # silent training
        ]
    )

    # Predict probabilities and labels
    y_proba = model.predict_proba(X_va)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_va, y_pred)
    prec = precision_score(y_va, y_pred, zero_division=0)
    rec = recall_score(y_va, y_pred, zero_division=0)
    f1 = f1_score(y_va, y_pred, zero_division=0)
    roc = roc_auc_score(y_va, y_proba)

    return model, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
    }


# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_metrics = []
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    model, metrics = train_and_eval_lightgbm(X_tr, y_tr, X_va, y_va, LGB_PARAMS)
    models.append(model)
    fold_metrics.append(metrics)

    print(
        f"Fold {fold}: "
        f"Acc={metrics['accuracy']:.3f}, "
        f"Prec={metrics['precision']:.3f}, "
        f"Rec={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"ROC-AUC={metrics['roc_auc']:.3f}"
    )


# %%
cv_df = pd.DataFrame(fold_metrics, index=[f"fold_{i}" for i in range(1, 6)])
cv_df.loc["mean"] = cv_df.mean()
cv_df.loc["std"] = cv_df.std()

print("\n===== 5-fold CV results (LightGBM, train set) =====")
print(cv_df.round(3))

# %%
# ===== Final training with calibration =====

# 1) Train base LightGBM on training set, validate on calibration set
base_model = lgb.LGBMClassifier(**LGB_PARAMS)

base_model.fit(
    X_train,
    y_train,
    eval_set=[(X_calib, y_calib)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(period=50)
    ]
)

# 2) Wrap with probability calibration
# method="isotonic" (more flexible) or "sigmoid" (Platt scaling)
calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")


calibrated_model.fit(X_calib, y_calib)

# 3) Evaluate on test set using calibrated probabilities
y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_roc = roc_auc_score(y_test, y_test_proba)


# %%
print("\n===== Test results (LightGBM) =====")
print(classification_report(y_test, y_test_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
print(
    f"Accuracy: {test_acc:.4f}, "
    f"F1: {test_f1:.4f}, "
    f"Precision: {test_prec:.4f}, "
    f"Recall: {test_rec:.4f}, "
    f"ROC-AUC: {test_roc:.4f}"
)

# %%
print("\n===== Test results (LightGBM, CALIBRATED) =====")
print(classification_report(y_test, y_test_pred, digits=4))
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix:\n", cm)
print(
    f"Accuracy: {test_acc:.4f}, "
    f"F1: {test_f1:.4f}, "
    f"Precision: {test_prec:.4f}, "
    f"Recall: {test_rec:.4f}, "
    f"ROC-AUC: {test_roc:.4f}"
)

# %%
# Visualize Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages for better interpretation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title(f'{title} (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_test_pred, "Test Set Confusion Matrix (Calibrated Model)")


# %%
# LightGBM Visualizations

# 1. Feature Importance (using LightGBM's built-in plot)
print("\n===== LightGBM Visualizations =====")

# Method 1: Using LightGBM's built-in plot_importance
lgb.plot_importance(base_model, importance_type='gain', max_num_features=10, 
                    title='LightGBM Feature Importance (Gain)', figsize=(10, 6))
plt.tight_layout()
plt.show()

# Method 2: Custom bar plot (already exists, keeping it)
importances = base_model.feature_importances_
fi_df = pd.DataFrame(
    {"feature": FEATURE_COLS, "importance": importances}
).sort_values("importance", ascending=False)

print("\nFeature importances:")
print(fi_df)

plt.figure(figsize=(8, 5))
plt.barh(fi_df["feature"], fi_df["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("LightGBM Feature Importance (PIMA) - Custom Plot")
plt.tight_layout()
plt.show()

# 2. Plot training metrics (AUC during training)
lgb.plot_metric(base_model, metric='auc', title='LightGBM Training Metrics (AUC)', figsize=(10, 6))
plt.tight_layout()
plt.show()

# 3. Plot a sample tree (first tree in the ensemble)
# Note: This can be large, so we'll plot just one tree
# Tree visualization requires graphviz to be installed
try:
    ax = lgb.plot_tree(base_model, tree_index=0, figsize=(20, 12), 
                       show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
    plt.title('LightGBM Tree Visualization (First Tree)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Note: Tree visualization requires graphviz. Error: {e}")
    print("To enable tree plots, install:")
    print("  1. pip install graphviz")
    print("  2. Install graphviz system package:")
    print("     - Windows: Download from https://graphviz.org/download/")
    print("     - Linux: sudo apt-get install graphviz")
    print("     - Mac: brew install graphviz")

# %%
def predict_single_patient(model, sample_dict):
    """
    sample_dict: {"Pregnancies": 2, "Glucose": 120, ...}
    returns: probability of diabetes
    """
    x = pd.DataFrame([sample_dict], columns=FEATURE_COLS)
    # Make sure we also convert 0s to NaN for relevant columns
    x[cols_zero_as_missing] = x[cols_zero_as_missing].replace(0, np.nan)
    proba = model.predict_proba(x)[:, 1][0]
    return proba

# Example (fake patient, just to test the function)
example_patient = {
    "Pregnancies": 2,
    "Glucose": 130,
    "BloodPressure": 80,
    "SkinThickness": 20,
    "Insulin": 100,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 35,
}

print("\nExample patient diabetes risk:",
      f"{predict_single_patient(calibrated_model, example_patient)*100:.1f}%")


# %%
# Save models for later use in the fusion project
joblib.dump(calibrated_model, "pima_lgbm_diabetes_calibrated.pkl")
joblib.dump(base_model, "pima_lgbm_diabetes_base.pkl")



