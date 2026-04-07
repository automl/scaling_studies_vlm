import numpy as np
import pandas as pd


class SurrogateDataset:
    DEFAULT_TARGETS = ["val/val_loss"]

    DEFAULT_FEATURES = [
        "total_cells", "config/effective_batch_size", "config/lr", "config/max_features",
        "config/model_config.embedding_size", "config/model_config.num_layers",
        "config/num_datapoints_max", "config/weight_decay"]

    DEFAULT_LOG_COLUMNS = []

    def __init__(
            self,
            train_csv_path: str,
            test_csv_path: str,
            features: list[str] | None = None,
            targets: list[str] | None = None,
            seed: int = 42,
            apply_log_transform: bool = True,
    ):
        self.features = features if features is not None else self.DEFAULT_FEATURES
        self.targets = targets if targets is not None else self.DEFAULT_TARGETS
        self.seed = int(seed)
        self.apply_log_transform = apply_log_transform

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        if self.apply_log_transform:
            self._apply_log_transform()

    def _apply_log_transform(self):
        for col in self.DEFAULT_LOG_COLUMNS:
            if col in self.train_df.columns:
                self.train_df[col] = np.log(self.train_df[col])
            if col in self.test_df.columns:
                self.test_df[col] = np.log(self.test_df[col])

    def get_test_data(self):
        X_test = self.test_df[self.features].to_numpy()
        y_test = self.test_df[self.targets].to_numpy()
        return X_test, y_test

    def get_train_data(self):
        X_train = self.train_df[self.features].to_numpy()
        y_train = self.train_df[self.targets].to_numpy()
        return X_train, y_train

    def get_all_data(self):
        all_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)

        X_all = all_df[self.features].to_numpy()
        y_all = all_df[self.targets].to_numpy()
        return X_all, y_all
