import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap


class BaseModel:
    def __init__(self, name, random_state=42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.target_name = None

    def _build_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        if self.model is None:
            self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X)

    def feature_importance(self, feature_names, save_path):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        importances = self.model.feature_importances_
        idx = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        ax.barh(range(len(feature_names)), importances[idx])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_xlabel("Importance")
        ax.set_title(f"{self.name} - {self.target_name} Feature Importance")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def shap_analysis(self, X_test, feature_names, save_path):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            X_df = pd.DataFrame(X_test, columns=feature_names)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_df, check_additivity=False)
            fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
            shap.summary_plot(shap_values, X_df, show=False)
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_samples = min(100, X_df.shape[0])
            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
            shap.summary_plot(shap_values[:n_samples], X_df.iloc[:n_samples], plot_type="bar", show=False)
            plt.tight_layout()
            bar_path = save_path.replace(".png", "_bar.png")
            fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig2)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"SHAP failed: {e}", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
