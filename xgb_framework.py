from base_model import BaseModel

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostModel(BaseModel):
    def __init__(self, random_state=42, n_estimators=300, learning_rate=0.05, max_depth=5):
        super().__init__(name="XGBoost", random_state=random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _build_model(self):
        if xgb is None:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
