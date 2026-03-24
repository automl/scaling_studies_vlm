import numpy as np

from surrogate_benchmark.config import VLMConfig, Target
from surrogate_benchmark.performance_surrogate.data import SurrogateDataset
from surrogate_benchmark.performance_surrogate.predictors.base import MultiLabelSurrogateModel
from surrogate_benchmark.performance_surrogate.predictors.pfn import TabPFNModel


class VLMBenchmark:
    def __init__(
            self,
            targets: list[Target] | None = None,
            device: str = "auto",
    ):
        targets = Target.all() if targets is None else [t.value for t in targets]
        self.dataset = SurrogateDataset(
            train_csv_path="performance_surrogate/splits/train.csv",
            test_csv_path="performance_surrogate/splits/test.csv",
            targets=targets,
            seed=42,
            include_intermediate_points=True,
        )

        self.feature_order = self.dataset.features
        self.targets = self.dataset.targets

        self.surrogate = MultiLabelSurrogateModel(
            labels=self.targets,
            model_factory=lambda label: TabPFNModel(seed=42, device=device),
        )
        self.fit_models()

    def fit_models(self):
        X, y = self.dataset.get_train_data()
        self.surrogate.fit(X, y)

    def query(self, config: VLMConfig) -> dict:
        config_dict = config.to_dict()

        row = self.config_to_feature_row(config_dict).reshape(1, -1)
        preds = self.surrogate.predict(row)[0]

        all_results = {
            label: round(float(pred), 4)
            for label, pred in zip(self.targets, preds)
        }

        return all_results

    def config_to_feature_row(self, config: dict) -> np.ndarray:
        missing = [name for name in self.feature_order if name not in config]
        if missing:
            raise ValueError(f"Missing features in config: {missing}")

        row = []
        for name in self.feature_order:
            value = config[name]

            if self.dataset.apply_log_transform and name in self.dataset.DEFAULT_LOG_COLUMNS:
                value = np.log(value)

            row.append(value)

        return np.array(row, dtype=float)


if __name__ == "__main__":
    vlm_bench = VLMBenchmark(targets=[Target.VAL_LOSS])

    config = VLMConfig(
        lr=1e-4,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=384,
        text_width=256,
        total_samples_planned=128000,
    )

    result = vlm_bench.query(config)
    print(result)
