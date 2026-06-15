import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold
)

from sklearn.ensemble import RandomForestRegressor

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
    "output_file": "results_random_forest.csv"
}

# ============================
# LOAD DATASET
# ============================

df = pd.read_csv(CONFIG["dataset_path"])

# ============================
# PREPROCESSING
# ============================

df["log_DopingDensity"] = np.log10(df["DopingDensity"])
df["log_DefectDensity"] = np.log10(df["DefectDensity"])

# Get dummies and keep track of which columns were created
df = pd.get_dummies(
    df,
    columns=["Material"],
    prefix=["Mat"]
)

# ============================
# FEATURES
# ============================

# Dynamically identify material columns that actually exist after get_dummies
mat_cols = [col for col in df.columns if col.startswith("Mat_")]

feature_cols = [
    "Thickness",
    "log_DopingDensity",
    "log_DefectDensity"
] + mat_cols

targets = [
    "Voc",
    "Jsc",
    "FF",
    "PCE"
]

# ==================================
# ZERO-VARIANCE FEATURE HANDLING
# ==================================

print("\n"+"="*60)
print("ZERO-VARIANCE FEATURE REPORT")
print("="*60)

cols_to_remove = []
for col in feature_cols:
    # Using df.nunique() directly on the original dataframe is robust
    num_unique = df[col].nunique()
    if num_unique == 1:
        print(f"Column: {col}, Unique Values: {num_unique} -> REMOVE (zero variance)")
        cols_to_remove.append(col)
    else:
        print(f"Column: {col}, Unique Values: {num_unique} -> KEEP")

# Remove zero-variance columns from feature_cols
feature_cols = [col for col in feature_cols if col not in cols_to_remove]

print("\nFinal Feature List after removing zero-variance columns:")
print(feature_cols)

# ============================
# HYPERPARAMETER SEARCH SPACE
# ============================

param_dist = {
    "n_estimators":[100,200,300,400,500],
    "max_depth":[None,5,10,15,20],
    "min_samples_leaf":[1,2,4],
    "min_samples_split":[2,5,10],
    "max_features":["sqrt","log2",None]
}

# ============================
# LOOP OVER TARGETS
# ============================

results = []

for target in targets:

    print("\n"+"="*60)
    print("TARGET:",target)
    print("="*60)

    X = df[feature_cols]
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    rf = RandomForestRegressor(
        random_state=CONFIG["random_state"]
    )

    search = RandomizedSearchCV(
        rf,
        param_dist,
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

    search.fit(X_train,y_train)

    best_rf = search.best_estimator_

    pred_train = best_rf.predict(X_train)
    pred_test = best_rf.predict(X_test)

    train_r2 = r2_score(y_train,pred_train)
    test_r2 = r2_score(y_test,pred_test)

    train_rmse = np.sqrt(
        mean_squared_error(y_train,pred_train)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test,pred_test)
    )

    train_mae = mean_absolute_error(
        y_train,pred_train
    )

    test_mae = mean_absolute_error(
        y_test,pred_test
    )

    print("\nBest Parameters")
    print(search.best_params_)

    print("\nMetrics")
    print("Train R2 :",train_r2)
    print("Test R2  :",test_r2)
    print("Train RMSE :",train_rmse)
    print("Test RMSE  :",test_rmse)
    print("Train MAE :",train_mae)
    print("Test MAE  :",test_mae)

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

    plt.figure(figsize=(8,5))

    importance = pd.Series(
        best_rf.feature_importances_,
        index=feature_cols
    )

    importance.sort_values().plot.barh()

    plt.title(
        f"{target} Feature Importance"
    )

    plt.tight_layout()
    plt.savefig(f"{target}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ============================
    # PARITY PLOT
    # ============================

    plt.figure(figsize=(5,5))

    plt.scatter(
        y_test,
        pred_test,
        alpha=0.7
    )

    mn = min(y_test.min(),pred_test.min())
    mx = max(y_test.max(),pred_test.max())

    plt.plot(
        [mn,mx],
        [mn,mx],
        "r--"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{target} Parity Plot")

    plt.tight_layout()
    plt.savefig(f"{target}_parity_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # ============================
    # SHAP
    # ============================

    explainer = shap.TreeExplainer(best_rf)

    # Ensure X_test uses only the kept features for SHAP
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{target}_shap_summary.png", dpi=300, bbox_inches='tight')
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

results_df.to_csv(
    CONFIG["output_file"],
    index=False
)

print(f"\nResults saved to {CONFIG['output_file']}")
