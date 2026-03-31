import json

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
        X, y = self.dataset.get_all_data()
        self.surrogate.fit(X, y)

    def query(self, config: VLMConfig) -> dict:
        config_dict = config.to_dict()

        row = self.config_to_feature_row(config_dict).reshape(1, -1)
        preds = self.surrogate.predict(row)[0]

        predictions = {
            label: round(float(pred), 3)
            for label, pred in zip(self.targets, preds)
        }

        text_width = config.text_width
        vision_width = config.vision_width
        stats_key = f"text_{text_width}_vision_{vision_width}"

        with open("gflops_mparams.json", "r", encoding="utf-8") as f:
            models_stats = json.load(f)

        model_stats = models_stats.get(stats_key)

        stats = {}
        if model_stats is not None:
            stats = {
                "gflops_per_sample": model_stats.get("gflops_per_sample"),
                "total_gflops": model_stats.get(
                    "gflops_per_sample") * config.total_samples_planned * config.training_progress,
                "mparams": model_stats.get("mparams"),
                "vision_mparams": model_stats.get("vision_mparams"),
                "text_mparams": model_stats.get("text_mparams"),
                "text_mparams_non_emb": model_stats.get("text_mparams_non_emb"),
            }

        return {
            "predictions": predictions,
            "model_stats": stats
        }

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
        training_progress=1.0
    )
    config2 = VLMConfig(
        lr=1e-4,
        wd=0.01,
        beta1=0.9,
        beta2=0.98,
        warmup_fraction=0.05,
        eps=1e-8,
        vision_width=384,
        text_width=256,
        total_samples_planned=128000,
        training_progress=0.5
    )

    result = vlm_bench.query(config)
    result2 = vlm_bench.query(config2)
    print(result)
    print(result2)
