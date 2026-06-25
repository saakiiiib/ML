from base_model import BaseModel
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LightGBMModel(BaseModel):
    def __init__(self, random_state=42, n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31):
        super().__init__(name="LightGBM", random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves

    def _build_model(self):
        if lgb is None:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )


if __name__ == "__main__":
    matplotlib.use('Agg')

    CONFIG = {
        "dataset_path": "CsSnI3_dataset.csv",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "search_iterations": 30,
        "output_file": "results_lightgbm.csv"
    }

    if lgb is None:
        raise ImportError("lightgbm is not installed. Install it with: pip install lightgbm")

    df = pd.read_csv(CONFIG["dataset_path"])

    if "DopingDensity" in df.columns:
        df["log_DopingDensity"] = np.log10(df["DopingDensity"])
    if "DefectDensity" in df.columns:
        df["log_DefectDensity"] = np.log10(df["DefectDensity"])

    if "Material" in df.columns:
        df = pd.get_dummies(df, columns=["Material"], prefix=["Mat"])

    mat_cols = [col for col in df.columns if col.startswith("Mat_")]

    feature_cols = ["Thickness", "log_DopingDensity", "log_DefectDensity"] + mat_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    targets = [t for t in ["Voc", "Jsc", "FF", "PCE"] if t in df.columns]

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

    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [-1, 3, 5, 7, 10],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    results = []

    for target in targets:
        print("\n" + "="*60)
        print("TARGET:", target)
        print("="*60)

        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
        )

        estimator = lgb.LGBMRegressor(random_state=CONFIG["random_state"])

        search = RandomizedSearchCV(
            estimator, param_distributions=param_dist, n_iter=CONFIG["search_iterations"],
            cv=KFold(CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
            scoring="r2", random_state=CONFIG["random_state"], n_jobs=-1
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
            target, train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae,
            str(search.best_params_)
        ])

        imp_path = f"{target}_lgb_feature_importance.png"
        plt.figure(figsize=(8, 5))
        importance = pd.Series(best_lgb.feature_importances_, index=feature_cols)
        importance.sort_values().plot.barh()
        plt.title(f"{target} Feature Importance")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {imp_path}")

        parity_path = f"{target}_lgb_parity_plot.png"
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
        plt.close()
        print(f"Saved: {parity_path}")

        shap_path = f"{target}_lgb_shap_summary.png"
        explainer = shap.TreeExplainer(best_lgb)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {shap_path}")

    results_df = pd.DataFrame(
        results,
        columns=["Target", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                 "Train_MAE", "Test_MAE", "Best_Parameters"]
    )

    print("\nFINAL RESULTS")
    print(results_df)

    results_df.to_csv(CONFIG["output_file"], index=False)
    print(f"\nResults saved to {CONFIG['output_file']}")
