from base_model import BaseModel

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
