import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold
)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ============================
# CONFIG
# ============================

CONFIG = {
    "dataset_path": "CsSnI3_dataset.csv",
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "search_iterations": 30,
    "output_file": "results_lightgbm.csv"
}

# ============================
# LOAD DATASET
# ============================

if lgb is None:
    raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")

df = pd.read_csv(CONFIG["dataset_path"]) 

# ============================
# PREPROCESSING
# ============================

# replicate the preprocessing used in rf_framework.py
if "DopingDensity" in df.columns:
    df["log_DopingDensity"] = np.log10(df["DopingDensity"]) 
if "DefectDensity" in df.columns:
    df["log_DefectDensity"] = np.log10(df["DefectDensity"]) 

# Get dummies for Material column if present
if "Material" in df.columns:
    df = pd.get_dummies(
        df,
        columns=["Material"],
        prefix=["Mat"]
    )

# ============================
# FEATURES
# ============================

mat_cols = [col for col in df.columns if col.startswith("Mat_")]

feature_cols = [
    "Thickness",
    "log_DopingDensity",
    "log_DefectDensity"
] + mat_cols

# filter out missing features gracefully
feature_cols = [c for c in feature_cols if c in df.columns]

targets = [t for t in ["Voc", "Jsc", "FF", "PCE"] if t in df.columns]

# ==================================
# ZERO-VARIANCE FEATURE HANDLING
# ==================================

print("\n" + "="*60)
print("ZERO-VARIANCE FEATURE REPORT")
print("="*60)

cols_to_remove = []
for col in feature_cols:
    num_unique = df[col].nunique()
    if num_unique == 1:
        print(f"Column: {col}, Unique Values: {num_unique} -> REMOVE (zero variance)")
        cols_to_remove.append(col)
    else:
        print(f"Column: {col}, Unique Values: {num_unique} -> KEEP")

feature_cols = [col for col in feature_cols if col not in cols_to_remove]

print("\nFinal Feature List after removing zero-variance columns:")
print(feature_cols)

# ============================
# HYPERPARAMETER SEARCH SPACE
# ============================

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [-1, 3, 5, 7, 10],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [5, 10, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# ============================
# LOOP OVER TARGETS
# ============================

results = []

for target in targets:

    print("\n" + "="*60)
    print("TARGET:", target)
    print("="*60)

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    estimator = lgb.LGBMRegressor(random_state=CONFIG["random_state"])

    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=CONFIG["search_iterations"],
        cv=KFold(
            CONFIG["cv_folds"],
            shuffle=True,
            random_state=CONFIG["random_state"]
        ),
        scoring="r2",
        random_state=CONFIG["random_state"],
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_lgb = search.best_estimator_

    pred_train = best_lgb.predict(X_train)
    pred_test = best_lgb.predict(X_test)

    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    print("\nBest Parameters")
    print(search.best_params_)

    print("\nMetrics")
    print("Train R2 :", train_r2)
    print("Test R2  :", test_r2)
    print("Train RMSE :", train_rmse)
    print("Test RMSE  :", test_rmse)
    print("Train MAE :", train_mae)
    print("Test MAE  :", test_mae)

    results.append([
        target,
        train_r2,
        test_r2,
        train_rmse,
        test_rmse,
        train_mae,
        test_mae,
        str(search.best_params_)
    ])

    # ============================
    # FEATURE IMPORTANCE
    # ============================

    plt.figure(figsize=(8, 5))

    importance = pd.Series(
        best_lgb.feature_importances_,
        index=feature_cols
    )

    importance.sort_values().plot.barh()

    plt.title(f"{target} Feature Importance")

    plt.tight_layout()
    plt.savefig(f"{target}_lgb_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ============================
    # PARITY PLOT
    # ============================

    plt.figure(figsize=(5, 5))

    plt.scatter(y_test, pred_test, alpha=0.7)

    mn = min(y_test.min(), pred_test.min())
    mx = max(y_test.max(), pred_test.max())

    plt.plot([mn, mx], [mn, mx], "r--")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{target} Parity Plot")

    plt.tight_layout()
    plt.savefig(f"{target}_lgb_parity_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ============================
    # SHAP
    # ============================

    explainer = shap.TreeExplainer(best_lgb)

    shap_values = explainer.shap_values(X_test, check_additivity=False)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{target}_lgb_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================
# FINAL TABLE
# ============================

results_df = pd.DataFrame(
    results,
    columns=[
        "Target",
        "Train_R2",
        "Test_R2",
        "Train_RMSE",
        "Test_RMSE",
        "Train_MAE",
        "Test_MAE",
        "Best_Parameters"
    ]
)

print("\nFINAL RESULTS")
print(results_df)

results_df.to_csv(CONFIG["output_file"], index=False)

print(f"\nResults saved to {CONFIG['output_file']}")
