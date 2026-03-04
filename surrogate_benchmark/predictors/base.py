import numpy as np

from surrogate_benchmark.data import SurrogateDataset
from surrogate_benchmark.metrics import compute_regression_metrics


class SurrogateModel:
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def validate(self, dataset: SurrogateDataset, sizes):
        X_test, y_test = dataset.get_test_data()
        results = {}

        for n_cfg in sizes:
            X_sub, y_sub = dataset.get_train_subset(n_cfg)
            self.fit(X_sub, y_sub)

            y_pred = self.predict(X_test)

            results[int(n_cfg)] = compute_regression_metrics(y_test, y_pred)

        return results
