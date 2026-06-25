import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from base_model import BaseModel

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostModel(BaseModel):
    def __init__(self, random_state=42, n_estimators=300, learning_rate=0.05, max_depth=5):
        super().__init__(name='XGBoost', random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _build_model(self):
        if xgb is None:
            raise ImportError('XGBoost is not installed. Install it with: pip install xgboost')
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )

    def feature_importance(self, feature_names, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importances = self.model.feature_importances_
        idx = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
        ax.barh(range(len(feature_names)), importances[idx])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in idx])
        ax.set_xlabel('Importance')
        ax.set_title(f'{self.name} - {self.target_name} Feature Importance')
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def shap_analysis(self, X_test, feature_names, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            X_df = pd.DataFrame(X_test, columns=feature_names)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_df, check_additivity=False)
            fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
            shap.summary_plot(shap_values, X_df, show=False)
            plt.tight_layout()
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
            shap.summary_plot(shap_values[:100], X_df.iloc[:100], plot_type='bar', show=False)
            plt.tight_layout()
            bar_path = save_path.replace('.png', '_bar.png')
            fig2.savefig(bar_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
        except Exception as e:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f'SHAP failed: {e}', ha='center', va='center', transform=ax.transAxes)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
