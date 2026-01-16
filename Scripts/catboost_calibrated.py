# %%
from sklearn.calibration import CalibratedClassifierCV


# %%
# 1. Imports & basic setup

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support
)

from catboost import CatBoostClassifier, Pool

import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# %%
# 2. Load dataset

# NOTE: For training from scratch, download the dataset and place in ../Datasets/
# For inference only, this file is not needed - only the trained model is used.
df = pd.read_csv("../Datasets/DatasetofDiabetes.csv")

print("Shape:", df.shape)
display(df.head())

print("\nInfo:")
print(df.info())

print("\nClass distribution:")
print(df["CLASS"].value_counts())


# %% [markdown]
# 3. Preprocessing

# %%
df_proc = df.copy()

# 3.1 Drop ID-like columns (adjust if your names differ)
id_cols = [col for col in df_proc.columns if col.lower() in ["id", "no_of_patient", "no_patient", "patient_no", "patient_id", "no_pation"]]
print("Dropping ID-like columns:", id_cols)
df_proc = df_proc.drop(columns=id_cols)

# %%
# 3.2 Standardize / encode Gender
# Assumes values like 'Male'/'Female' or 'M'/'F'
if "Gender" in df_proc.columns:
    df_proc["Gender"] = df_proc["Gender"].astype(str).str.strip().str.lower()
    df_proc["Gender"] = df_proc["Gender"].replace({
        "male": 1, "m": 1,
        "female": 0, "f": 0
    })
else:
    print("WARNING: 'Gender' column not found. Adjust preprocessing if necessary.")

# %%
# 3.3 Treat physiologically impossible zeros as missing (NaN)
# Adjust these names to match your CSV exactly
zero_as_missing_cols = [
    "HbA1c",
    "Cr",
    "BMI",
    "Urea",
    "Chol",
    "LDL",
    "VLDL",
    "TG",
    "HDL"
]

for col in zero_as_missing_cols:
    if col in df_proc.columns:
        # Only replace zeros if there are any
        n_zeros = (df_proc[col] == 0).sum()
        if n_zeros > 0:
            print(f"Replacing {n_zeros} zeros with NaN in column '{col}'")
            df_proc.loc[df_proc[col] == 0, col] = np.nan
    else:
        print(f"Column '{col}' not found in dataframe (OK if naming differs).")

# %%
# 3.4 Separate features & target
target_col = "CLASS"
assert target_col in df_proc.columns, f"Target column '{target_col}' not found!"

# %%

X = df_proc.drop(columns=[target_col])
y_raw = df_proc[target_col]

# Encode class labels to integers for metrics
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("\nEncoded classes:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print("X shape:", X.shape, "y shape:", y.shape)

# Check missing values
print("\nMissing values per column:")
print(X.isna().sum())


# %%
# 4. Train / validation / test split

# First: hold out a test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train+Val shape:", X_train_full.shape, "Test shape:", X_test.shape)

# Second: split train_full into train and calibration sets
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,          # 20% of train_full for calibration
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

print("Train shape:", X_train.shape, "Calibration shape:", X_calib.shape)


# %%
# # 4. Train / validation / test split

# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     stratify=y,
#     random_state=RANDOM_STATE
# )

# print("Train+Val shape:", X_train_full.shape, "Test shape:", X_test.shape)

# # Optionally, we can later do K-fold CV on X_train_full, y_train_full


# %%
# 5. CatBoost parameters (good strong baseline for multiclass)

N_CLASSES = len(np.unique(y))

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.03,
    "depth": 5,
    "l2_leaf_reg": 3.0,
    "random_seed": RANDOM_STATE,
    "iterations": 3000,        # we'll rely on early stopping
    "od_type": "Iter",
    "od_wait": 100,            # early stopping rounds
    "thread_count": -1,
    "verbose": 100,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.5
}


# %%
# 6. Stratified K-Fold cross-validation (optional but recommended)

N_SPLITS = 5

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_metrics = []

fold_idx = 1

for train_idx, val_idx in skf.split(X_train_full, y_train_full):
    print(f"\n========== Fold {fold_idx} / {N_SPLITS} ==========")
    
    X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_va = y_train_full[train_idx], y_train_full[val_idx]
    
    # Identify categorical features indices for CatBoost (Gender + any others that are not purely numeric)
    cat_features = [i for i, col in enumerate(X_tr.columns) if X_tr[col].dtype == "object" or col.lower() == "gender"]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    valid_pool = Pool(X_va, y_va, cat_features=cat_features)
    
    model_cv = CatBoostClassifier(**CATBOOST_PARAMS)
    model_cv.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Predictions
    y_va_pred = model_cv.predict(valid_pool)
    y_va_pred = y_va_pred.flatten().astype(int)
    
    acc = accuracy_score(y_va, y_va_pred)
    f1_macro = f1_score(y_va, y_va_pred, average="macro")
    
    print(f"Fold {fold_idx} Accuracy: {acc:.4f}, Macro-F1: {f1_macro:.4f}")
    
    fold_metrics.append({"fold": fold_idx, "accuracy": acc, "f1_macro": f1_macro})
    fold_idx += 1

cv_df = pd.DataFrame(fold_metrics)
print("\n===== CV Summary =====")
display(cv_df)
print("Mean accuracy:", cv_df["accuracy"].mean().round(4), "±", cv_df["accuracy"].std().round(4))
print("Mean macro F1:", cv_df["f1_macro"].mean().round(4), "±", cv_df["f1_macro"].std().round(4))


# %%
# 7. Final training with calibration

# Categorical features (reuse logic)
cat_features = [i for i, col in enumerate(X_train.columns)
                if X_train[col].dtype == "object" or col.lower() == "gender"]

# Pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
calib_pool = Pool(X_calib, y_calib, cat_features=cat_features)

# 7.1 Train base CatBoost model
base_model = CatBoostClassifier(**CATBOOST_PARAMS)
base_model.fit(
    train_pool,
    eval_set=calib_pool,   # early stopping uses calibration set
    use_best_model=True
)

# 7.2 Wrap with probability calibration
# CalibratedClassifierCV expects raw X, not Pool
calibrated_model = CalibratedClassifierCV(
    base_model,
    method="isotonic",     # or "sigmoid" (Platt). Isotonic is more flexible.
    cv="prefit"            # base_model is already trained
)

calibrated_model.fit(X_calib, y_calib)

# 7.3 Evaluate on test using calibrated probabilities
y_test_proba = calibrated_model.predict_proba(X_test)   # calibrated probs
y_test_pred = np.argmax(y_test_proba, axis=1)           # predicted class indices

acc = accuracy_score(y_test, y_test_pred)
f1_macro = f1_score(y_test, y_test_pred, average="macro")

try:
    roc_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
except Exception as e:
    print("ROC-AUC could not be computed:", e)
    roc_auc = None

print("\n===== Test results (CatBoost, multiclass, CALIBRATED) =====")
print("Accuracy:", round(acc, 4))
print("Macro F1:", round(f1_macro, 4))
if roc_auc is not None:
    print("ROC-AUC (ovr):", round(roc_auc, 4))

print("\nClassification report (labels are encoded classes):")
print(classification_report(y_test, y_test_pred, digits=4))

print("\nLabel mapping (encoded -> original):")
for enc, lab in enumerate(label_encoder.classes_):
    print(f"  {enc} -> {lab}")


# %%
# # 7. Final training on full train+val and evaluation on test

# # Categorical features (reuse logic)
# cat_features = [i for i, col in enumerate(X_train_full.columns) if X_train_full[col].dtype == "object" or col.lower() == "gender"]

# train_pool_full = Pool(X_train_full, y_train_full, cat_features=cat_features)
# test_pool = Pool(X_test, y_test, cat_features=cat_features)

# final_model = CatBoostClassifier(**CATBOOST_PARAMS)
# final_model.fit(train_pool_full, eval_set=test_pool, use_best_model=True)

# # Predictions on test
# y_test_pred = final_model.predict(test_pool).flatten().astype(int)
# y_test_proba = final_model.predict_proba(test_pool)  # shape: [n_samples, n_classes]

# acc = accuracy_score(y_test, y_test_pred)
# f1_macro = f1_score(y_test, y_test_pred, average="macro")

# # For multiclass ROC-AUC, use 'ovr' or 'ovo'
# try:
#     roc_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
# except Exception as e:
#     print("ROC-AUC could not be computed:", e)
#     roc_auc = None

# print("\n===== Test results (CatBoost, multiclass) =====")
# print("Accuracy:", round(acc, 4))
# print("Macro F1:", round(f1_macro, 4))
# if roc_auc is not None:
#     print("ROC-AUC (ovr):", round(roc_auc, 4))

# print("\nClassification report (labels are encoded classes):")
# print(classification_report(y_test, y_test_pred, digits=4))

# # Mapping back to original class labels
# print("\nLabel mapping (encoded -> original):")
# for enc, lab in enumerate(label_encoder.classes_):
#     print(f"  {enc} -> {lab}")


# %%
# 8. Enhanced Confusion Matrix Visualization

def plot_confusion_matrix(y_true, y_pred, label_encoder, title="Confusion Matrix"):
    """Plot an enhanced heatmap of the confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels from encoder
    class_labels = label_encoder.classes_
    
    # Calculate percentages for better interpretation
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create DataFrames
    cm_df = pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in class_labels],
        columns=[f"Pred: {c}" for c in class_labels]
    )
    
    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"True: {c}" for c in class_labels],
        columns=[f"Pred: {c}" for c in class_labels]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                ax=axes[0], cbar_kws={'label': 'Count'}, linewidths=0.5)
    axes[0].set_title(f'{title} (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Plot 2: Percentages
    sns.heatmap(cm_percent_df, annot=True, fmt='.1f', cmap='Blues',
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    axes[1].set_title(f'{title} (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return cm

cm = plot_confusion_matrix(y_test, y_test_pred, label_encoder, 
                           "Test Set Confusion Matrix (CatBoost Calibrated)")


# %%
# 9. CatBoost Visualizations

print("\n===== CatBoost Visualizations =====")

# 9.1 Feature Importance
importances = base_model.get_feature_importance(train_pool)
feat_imp_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature importances:")
print(feat_imp_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis")
plt.title("CatBoost Feature Importances", fontsize=14, fontweight='bold')
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# 9.2 Feature importance using CatBoost's built-in method (if available)
try:
    # CatBoost can plot feature importance directly
    base_model.plot_feature_importance(plot_size=(10, 6))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Note: Built-in plot_feature_importance not available. Error: {e}")

# 9.3 Training metrics visualization (if training history is available)
try:
    # Get training history
    history = base_model.get_evals_result()
    if history:
        train_metrics = history.get('learn', {})
        val_metrics = history.get('validation', {})
        
        if 'MultiClass' in train_metrics or 'MultiClass' in val_metrics:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot training loss/metric
            if 'MultiClass' in train_metrics:
                train_values = train_metrics['MultiClass']
                axes[0].plot(train_values, label='Train', linewidth=2)
                axes[0].set_xlabel('Iteration', fontsize=12)
                axes[0].set_ylabel('MultiClass Metric', fontsize=12)
                axes[0].set_title('Training Metric', fontsize=13, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot validation loss/metric
            if 'MultiClass' in val_metrics:
                val_values = val_metrics['MultiClass']
                axes[1].plot(val_values, label='Validation', color='orange', linewidth=2)
                axes[1].set_xlabel('Iteration', fontsize=12)
                axes[1].set_ylabel('MultiClass Metric', fontsize=12)
                axes[1].set_title('Validation Metric', fontsize=13, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
except Exception as e:
    print(f"Note: Training metrics visualization not available. Error: {e}")

# 9.4 Class-wise performance visualization
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, labels=range(len(label_encoder.classes_)), zero_division=0
)

metrics_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Precision
sns.barplot(data=metrics_df, x='Class', y='Precision', ax=axes[0], palette='Blues')
axes[0].set_title('Precision by Class', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].tick_params(axis='x', rotation=45)

# Recall
sns.barplot(data=metrics_df, x='Class', y='Recall', ax=axes[1], palette='Greens')
axes[1].set_title('Recall by Class', fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].tick_params(axis='x', rotation=45)

# F1-Score
sns.barplot(data=metrics_df, x='Class', y='F1-Score', ax=axes[2], palette='Oranges')
axes[2].set_title('F1-Score by Class', fontsize=13, fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\nClass-wise metrics:")
print(metrics_df.round(4))


# %%
# # 9. Feature importance

# importances = final_model.get_feature_importance(train_pool_full)
# feat_imp_df = pd.DataFrame({
#     "feature": X_train_full.columns,
#     "importance": importances
# }).sort_values(by="importance", ascending=False)

# print("\nFeature importances:")
# display(feat_imp_df)

# plt.figure(figsize=(8, 5))
# sns.barplot(data=feat_imp_df, x="importance", y="feature")
# plt.title("CatBoost Feature Importances")
# plt.tight_layout()
# plt.show()


# %%
# # 10. Save model & label encoder (optional)

# import joblib

# joblib.dump(final_model, "catboost_diabetes_multiclass_model.pkl")
# joblib.dump(label_encoder, "catboost_diabetes_label_encoder.pkl")

# print("Saved model and label encoder to disk.")


# %%
# 10. Save model & label encoder (optional)

import joblib

joblib.dump(calibrated_model, "catboost_diabetes_multiclass_calibrated.pkl")
joblib.dump(base_model, "catboost_diabetes_multiclass_base.pkl")  # optional
joblib.dump(label_encoder, "catboost_diabetes_label_encoder.pkl")


print("Saved model and label encoder to disk.")


# %%
import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------
# 1. LOAD TRAINED + CALIBRATED MODEL
# -----------------------------------------------------
calibrated_model = joblib.load("catboost_diabetes_multiclass_calibrated.pkl")
label_encoder = joblib.load("catboost_diabetes_label_encoder.pkl")

# -----------------------------------------------------
# 2. CREATE A SAMPLE INPUT ROW (must match training columns)
# Replace values however you want for testing.
# Columns inferred from your uploaded DatasetofDiabetes.csv:
# ['Gender','Age','Urea','Cr','HbA1c','Chol','TG','HDL','LDL','VLDL','BMI']
# -----------------------------------------------------

sample = pd.DataFrame([{
    "Gender": "Male",        # or "Female"
    "AGE": 45,
    "Urea": 38,
    "Cr": 0.9,
    "HbA1c": 6.4,
    "Chol": 190,
    "TG": 165,
    "HDL": 45,
    "LDL": 120,
    "VLDL": 25,
    "BMI": 28.4
}])

# -----------------------------------------------------
# 3. RUN PREDICTION
# -----------------------------------------------------
proba = calibrated_model.predict_proba(sample)[0]
pred_class_idx = np.argmax(proba)
pred_label = label_encoder.inverse_transform([pred_class_idx])[0]

# -----------------------------------------------------
# 4. DISPLAY RESULTS
# -----------------------------------------------------
print("===== CALIBRATED PROBABILITIES =====")
print(f"Probability Non-Diabetic (N):     {proba[0]:.4f}")
print(f"Probability Prediabetic (P):       {proba[1]:.4f}")
print(f"Probability Diabetic (Y):          {proba[2]:.4f}")

print("\nFinal Predicted Class:", pred_label)



