from base_model import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class KNNModel(BaseModel):
    def __init__(self, random_state=42, n_neighbors=5, weights="uniform", metric="minkowski"):
        super().__init__(name="KNN", random_state=random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        # KNN needs feature scaling — store scaler separately
        self.scaler = StandardScaler()

    def _build_model(self):
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            n_jobs=-1,
        )

    # Override fit to apply scaling
    def fit(self, X, y):
        if self.model is None:
            self._build_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    # Override predict to apply same scaling
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # KNN has no feature_importances_ — override with permutation importance
    def feature_importance(self, X_test, y_test, feature_names, save_path, random_state=42):
        import os
        from sklearn.inspection import permutation_importance
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        X_scaled = self.scaler.transform(X_test)
        result = permutation_importance(
            self.model, X_scaled, y_test,
            n_repeats=10, random_state=random_state, n_jobs=-1
        )
        importances = result.importances_mean
        idx = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        ax.barh(range(len(feature_names)), importances[idx])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_xlabel("Mean Permutation Importance")
        ax.set_title(f"KNN - {self.target_name} Permutation Feature Importance")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return importances

    # KNN does not support TreeExplainer — use KernelExplainer with background sample
    def shap_analysis(self, X_test, feature_names, save_path, background_size=50):
        import os
        import shap
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            X_scaled = self.scaler.transform(X_test)
            X_df = pd.DataFrame(X_scaled, columns=feature_names)
            background = shap.sample(X_df, min(background_size, len(X_df)))
            explainer = shap.KernelExplainer(self.model.predict, background)
            shap_values = explainer.shap_values(X_df.iloc[:100], silent=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
            shap.summary_plot(shap_values, X_df.iloc[:100], show=False)
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"SHAP failed: {e}", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    CONFIG = {
        "dataset_path": "CsSnI3_dataset.csv",
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "search_iterations": 30,
        "output_file": "results_knn.csv"
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

    # KNN pipeline uses StandardScaler internally via the class
    # For RandomizedSearchCV we build a sklearn Pipeline so scaling
    # is applied correctly inside each CV fold
    param_dist = {
        "knn__n_neighbors": [3, 5, 7, 10, 15, 20, 30, 50],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski", "euclidean", "manhattan"],
        "knn__p": [1, 2]          # only used when metric=minkowski
    }

    results = []

    for target in targets:
        print("\n" + "="*60)
        print("TARGET:", target)
        print("="*60)

        X = df[feature_cols].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
        )

        # Pipeline: scale inside CV to avoid data leakage
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_jobs=-1))
        ])

        search = RandomizedSearchCV(
            pipe, param_distributions=param_dist, n_iter=CONFIG["search_iterations"],
            cv=KFold(CONFIG["cv_folds"], shuffle=True, random_state=CONFIG["random_state"]),
            scoring="r2", random_state=CONFIG["random_state"], n_jobs=-1
        )

        search.fit(X_train, y_train)

        best_pipe = search.best_estimator_

        pred_train = best_pipe.predict(X_train)
        pred_test = best_pipe.predict(X_test)

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

        # Permutation importance (KNN has no native feature_importances_)
        from sklearn.inspection import permutation_importance
        scaler_fit = best_pipe.named_steps["scaler"]
        knn_fit = best_pipe.named_steps["knn"]
        X_test_scaled = scaler_fit.transform(X_test)

        perm = permutation_importance(
            knn_fit, X_test_scaled, y_test,
            n_repeats=10, random_state=CONFIG["random_state"], n_jobs=-1
        )
        imp_path = f"{target}_knn_feature_importance.png"
        plt.figure(figsize=(8, 5))
        importance = pd.Series(perm.importances_mean, index=feature_cols)
        importance.sort_values().plot.barh()
        plt.title(f"{target} Permutation Feature Importance (KNN)")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {imp_path}")

        # Parity plot
        parity_path = f"{target}_knn_parity_plot.png"
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, pred_test, alpha=0.7)
        mn = min(y_test.min(), pred_test.min())
        mx = max(y_test.max(), pred_test.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target} Parity Plot (KNN)")
        plt.tight_layout()
        plt.savefig(parity_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {parity_path}")

        # SHAP via KernelExplainer (model-agnostic, slower)
        shap_path = f"{target}_knn_shap_summary.png"
        try:
            import shap
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
            background = shap.sample(X_test_df, min(50, len(X_test_df)))
            explainer = shap.KernelExplainer(knn_fit.predict, background)
            # limit to 100 samples — KernelExplainer is slow
            shap_values = explainer.shap_values(X_test_df.iloc[:100], silent=True)
            plt.figure()
            shap.summary_plot(shap_values, X_test_df.iloc[:100],
                              feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Saved: {shap_path}")
        except Exception as e:
            print(f"SHAP skipped for {target}: {e}")

    results_df = pd.DataFrame(
        results,
        columns=["Target", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                 "Train_MAE", "Test_MAE", "Best_Parameters"]
    )

    print("\nFINAL RESULTS")
    print(results_df)

    results_df.to_csv(CONFIG["output_file"], index=False)
    print(f"\nResults saved to {CONFIG['output_file']}")
