from base_model import BaseModel
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# NOTE ON IMPLEMENTATION CHOICE
# ------------------------------
# The papers you referenced build the ANN in Keras/TensorFlow (Sequential + Dense
# layers, Adam optimizer, manual epoch loop). This repo's other frameworks are all
# scikit-learn based with no deep-learning dependency in requirements.txt, so this
# file uses sklearn.neural_network.MLPRegressor instead — same feedforward /
# backprop architecture, but it plugs directly into the existing Pipeline +
# RandomizedSearchCV pattern used by knn_framework.py, with no new dependency.
#
# If you specifically need epoch-by-epoch loss curves (val_loss plots) for the
# paper's figures, or want to match the Keras architecture 1:1 (Dense(128)->
# Dropout(0.2)->Dense(64)->Dense(32)->Dense(1)) for a direct comparison against
# those five papers, say so and I'll write a keras_ann_framework.py variant
# instead/in addition — that needs tensorflow added to requirements.txt.


class ANNModel(BaseModel):
    def __init__(self, random_state=42, hidden_layer_sizes=(128, 64, 32),
                 activation="relu", alpha=1e-4, learning_rate_init=1e-3,
                 max_iter=2000):
        super().__init__(name="ANN", random_state=random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        # ANN needs feature scaling — store scaler separately (same as KNNModel)
        self.scaler = StandardScaler()

    def _build_model(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=self.random_state,
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

    # MLPRegressor has no feature_importances_ — use permutation importance
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
        ax.set_title(f"ANN - {self.target_name} Permutation Feature Importance")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return importances

    # MLPRegressor is not tree-based — use KernelExplainer with background sample
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
            plt.close(fig)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"SHAP failed: {e}", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG = {
        "dataset_path": os.path.join(BASE_DIR, "datasets", "CsSnI3_dataset.csv"),
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "search_iterations": 10,
        "output_dir": os.path.join(BASE_DIR, "output", "ann"),
        "output_file": "results_ann.csv"
    }

    OUTPUT_DIR = CONFIG["output_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    feature_cols = [col for col in feature_cols if col not in cols_to_remove]
    print("\nFinal Feature List after removing zero-variance columns:")
    print(feature_cols)

    # ANN pipeline uses StandardScaler internally via the class.
    # For RandomizedSearchCV we build a sklearn Pipeline so scaling
    # is applied correctly inside each CV fold (avoids leakage).
    param_dist = {
        "ann__hidden_layer_sizes": [
            (32,), (64,), (128,),
            (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)
        ],
        "ann__activation": ["relu", "tanh"],
        "ann__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "ann__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
    }

    results = []

    for target in targets:
        print("\n" + "=" * 60)
        print("TARGET:", target)
        print("=" * 60)

        X = df[feature_cols].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ann", MLPRegressor(
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=CONFIG["random_state"],
            ))
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

        # Permutation importance (MLPRegressor has no native feature_importances_)
        from sklearn.inspection import permutation_importance
        scaler_fit = best_pipe.named_steps["scaler"]
        ann_fit = best_pipe.named_steps["ann"]
        X_test_scaled = scaler_fit.transform(X_test)

        perm = permutation_importance(
            ann_fit, X_test_scaled, y_test,
            n_repeats=10, random_state=CONFIG["random_state"], n_jobs=-1
        )
        imp_path = os.path.join(OUTPUT_DIR, f"{target}_ann_feature_importance.png")
        plt.figure(figsize=(8, 5))
        importance = pd.Series(perm.importances_mean, index=feature_cols)
        importance.sort_values().plot.barh()
        plt.title(f"{target} Permutation Feature Importance (ANN)")
        plt.tight_layout()
        plt.savefig(imp_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {imp_path}")

        # Loss curve (ANN-specific diagnostic)
        loss_path = os.path.join(OUTPUT_DIR, f"{target}_ann_loss_curve.png")
        plt.figure(figsize=(6, 4))
        plt.plot(ann_fit.loss_curve_, label="Training loss")
        if hasattr(ann_fit, 'validation_scores_') and ann_fit.validation_scores_:
            plt.plot(ann_fit.validation_scores_, label="Validation score")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(f"{target} ANN Training Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {loss_path}")

        # Parity plot
        parity_path = os.path.join(OUTPUT_DIR, f"{target}_ann_parity_plot.png")
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, pred_test, alpha=0.7)
        mn = min(y_test.min(), pred_test.min())
        mx = max(y_test.max(), pred_test.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target} Parity Plot (ANN)")
        plt.tight_layout()
        plt.savefig(parity_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {parity_path}")

        # SHAP via KernelExplainer (model-agnostic)
        shap_path = os.path.join(OUTPUT_DIR, f"{target}_ann_shap_summary.png")
        try:
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
            background = shap.sample(X_test_df, min(50, len(X_test_df)))
            explainer = shap.KernelExplainer(ann_fit.predict, background)
            shap_values = explainer.shap_values(X_test_df.iloc[:100], silent=True)
            shap.summary_plot(shap_values, X_test_df.iloc[:100],
                              feature_names=feature_cols, show=False)
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close(plt.gcf())
            print(f"Saved: {shap_path}")
        except Exception as e:
            print(f"SHAP skipped for {target}: {e}")

        # Residual plot
        resid_path = os.path.join(OUTPUT_DIR, f"{target}_ann_residual_plot.png")
        plt.figure(figsize=(5, 5))
        residuals = y_test - pred_test
        plt.scatter(pred_test, residuals, alpha=0.7)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"{target} Residual Plot (ANN)")
        plt.tight_layout()
        plt.savefig(resid_path, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        print(f"Saved: {resid_path}")

    results_df = pd.DataFrame(
        results,
        columns=["Target", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                 "Train_MAE", "Test_MAE", "Best_Parameters"]
    )

    print("\nFINAL RESULTS")
    print(results_df)

    results_df.to_csv(os.path.join(OUTPUT_DIR, CONFIG["output_file"]), index=False)
    print(f"\nResults saved to {os.path.join(OUTPUT_DIR, CONFIG['output_file'])}")