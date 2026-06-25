from base_model import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class RandomForestModel(BaseModel):
    def __init__(self, random_state=42, n_estimators=300, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        super().__init__(name="RandomForest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def _build_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,
        )


if __name__ == "__main__":
    CONFIG = {
        "dataset_path": "CsSnI3_dataset.csv",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "search_iterations": 30,
        "output_file": "results_random_forest.csv"
    }

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
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None]
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

        rf = RandomForestRegressor(random_state=CONFIG["random_state"])

        search = RandomizedSearchCV(
            rf, param_dist, n_iter=CONFIG["search_iterations"],
            cv=KFold(CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
            scoring="r2", random_state=CONFIG["random_state"], n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_rf = search.best_estimator_

        pred_train = best_rf.predict(X_train)
        pred_test = best_rf.predict(X_test)

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

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import shap

        plt.figure(figsize=(8, 5))
        importance = pd.Series(best_rf.feature_importances_, index=feature_cols)
        importance.sort_values().plot.barh()
        plt.title(f"{target} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{target}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, pred_test, alpha=0.7)
        mn = min(y_test.min(), pred_test.min())
        mx = max(y_test.max(), pred_test.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target} Parity Plot")
        plt.tight_layout()
        plt.savefig(f"{target}_parity_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(f"{target}_shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

    results_df = pd.DataFrame(
        results,
        columns=["Target", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                 "Train_MAE", "Test_MAE", "Best_Parameters"]
    )

    print("\nFINAL RESULTS")
    print(results_df)

    results_df.to_csv(CONFIG["output_file"], index=False)
    print(f"\nResults saved to {CONFIG['output_file']}")
