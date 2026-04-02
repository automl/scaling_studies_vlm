import numpy as np
from tabpfn import TabPFNRegressor

from surrogate_benchmark.performance_surrogate.predictors.base import SurrogateModel


class TabPFNModel(SurrogateModel):
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = TabPFNRegressor(device=device, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
