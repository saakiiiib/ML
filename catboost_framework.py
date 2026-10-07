from base_model import BaseModel
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


class CatBoostModel(BaseModel):
    def __init__(self, random_state=42, iterations=1000, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3, cat_features=None):
        super().__init__(name="CatBoost", random_state=random_state)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.cat_features = cat_features

    def _build_model(self):
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed. Install it with: pip install catboost")
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="RMSE",
            random_state=self.random_state,
            cat_features=self.cat_features,
            verbose=False,
        )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG = {
        "dataset_path": os.path.join(BASE_DIR, "datasets", "CsSnI3_dataset.csv"),
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "search_iterations": 30,
        "output_dir": os.path.join(BASE_DIR, "output", "catboost"),
        "output_file": "results_catboost.csv",
        "native_cat_features": None,
    }

    OUTPUT_DIR = CONFIG["output_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if CatBoostRegressor is None:
        raise ImportError("CatBoost is not installed. Install it with: pip install catboost")

    df = pd.read_csv(CONFIG["dataset_path"])

    if "DopingDensity" in df.columns:
        df["log_DopingDensity"] = np.log10(df["DopingDensity"])
    if "DefectDensity" in df.columns:
        df["log_DefectDensity"] = np.log10(df["DefectDensity"])

    native_cats = CONFIG["native_cat_features"] or []
    if "Material" in df.columns and "Material" not in native_cats:
        df = pd.get_dummies(df, columns=["Material"], prefix=["Mat"])

    mat_cols = [col for col in df.columns if col.startswith("Mat_")]

    feature_cols = ["Thickness", "log_DopingDensity", "log_DefectDensity"] + mat_cols + native_cats
    feature_cols = [c for c in feature_cols if c in df.columns]

    targets = [t for t in ["Voc", "Jsc", "FF", "PCE"] if t in df.columns]

    print("\n" + "=" * 60)
    print("ZERO-VARIANCE FEATURE REPORT")
    print("=" * 60)

    cols_to_remove = []
    for col in feature_cols:
        num_unique = df[col].nunique()
        if num_unique == 1:
            print(f"Column: {col}, Unique Values: {num_unique} -> REMOVE (zero variance)")
            cols_to_remove.append(col)
        else:
            print(f"Column: {col}, Unique Values: {num_unique} -> KEEP")

    feature_cols = [c for c in feature_cols if c not in cols_to_remove]
    native_cats = [c for c in native_cats if c in feature_cols]
    print("\nFinal Feature List after removing zero-variance columns:")
    print(feature_cols)

    cat_feature_indices = [feature_cols.index(c) for c in native_cats] if native_cats else None

    param_dist = {
        "iterations": [300, 500, 800, 1000, 1500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "depth": [4, 6, 7, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128],
        "bagging_temperature": [0, 0.5, 1.0],
    }

    results = []

    for target in targets:
        print("\n" + "=" * 60)
        print("TARGET:", target)
        print("=" * 60)

        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
        )

        estimator = CatBoostRegressor(
            random_state=CONFIG["random_state"],
            loss_function="RMSE",
            cat_features=cat_feature_indices,
            verbose=False,
        )

        search = RandomizedSearchCV(
            estimator, param_distributions=param_dist, n_iter=CONFIG["search_iterations"],
            cv=KFold(CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
            scoring="r2", random_state=CONFIG["random_state"], n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_cb = search.best_estimator_

        pred_train = best_cb.predict(X_train)
        pred_test = best_cb.predict(X_test)

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
            target, train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
            str(search.best_params_)
        ])

        imp_path = os.path.join(OUTPUT_DIR, f"{target}_catboost_feature_importance.png")
        plt.figure(figsize=(8, 5))
        importance = pd.Series(best_cb.get_feature_importance(), index=feature_cols)
        importance.sort_values().plot.barh()
        plt.title(f"{target} Feature Importance")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {imp_path}")

        parity_path = os.path.join(OUTPUT_DIR, f"{target}_catboost_parity_plot.png")
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, pred_test, alpha=0.7)
        mn = min(y_test.min(), pred_test.min())
        mx = max(y_test.max(), pred_test.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target} Parity Plot")
        plt.tight_layout()
        plt.savefig(parity_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {parity_path}")

        shap_path = os.path.join(OUTPUT_DIR, f"{target}_catboost_shap_summary.png")
        explainer = shap.TreeExplainer(best_cb)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {shap_path}")

    results_df = pd.DataFrame(
        results,
        columns=["Target", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                 "Train_MAE", "Test_MAE", "Best_Parameters"]
    )

    print("\nFINAL RESULTS")
    print(results_df)

    results_df.to_csv(os.path.join(OUTPUT_DIR, CONFIG["output_file"]), index=False)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, CONFIG['output_file'])}")
