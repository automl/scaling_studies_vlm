import json

from surrogate_benchmark.config import VLMConfig, Target
from surrogate_benchmark.divergence_surrogate.data import DivergenceDataset
from surrogate_benchmark.divergence_surrogate.predictors.xgb import BinaryBaggingEnsemble
from surrogate_benchmark.feature_encoder import FeatureEncoder
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

        self.encoder = FeatureEncoder(
            surrogate_feature_order=self.dataset.features,
            divergence_feature_order=DivergenceDataset.DEFAULT_FEATURES,
            apply_log=self.dataset.apply_log_transform,
            log_columns=self.dataset.DEFAULT_LOG_COLUMNS,
        )

        self.targets = self.dataset.targets

        self.divergence_surrogate = BinaryBaggingEnsemble()

        self.surrogate = MultiLabelSurrogateModel(
            labels=self.targets,
            model_factory=lambda label: TabPFNModel(device=device),
        )
        self.fit_models()

    def fit_models(self):
        X, y = self.dataset.get_all_data()
        self.surrogate.fit(X, y)

    def query(self, config: VLMConfig) -> dict:
        divergence_row = self.encoder.transform_divergence(config)

        divergence_prob = float(self.divergence_surrogate.predict_proba(divergence_row)[0])
        failed = divergence_prob >= 0.5

        text_width = config.text_width
        vision_width = config.vision_width
        stats_key = f"text_{text_width}_vision_{vision_width}"

        with open("gflops_mparams.json", "r", encoding="utf-8") as f:
            models_stats = json.load(f)

        model_stats = models_stats.get(stats_key)

        stats = {}
        if model_stats is not None:
            gflops_per_sample = model_stats.get("gflops_per_sample")
            stats = {
                "gflops_per_sample": gflops_per_sample,
                "total_gflops": gflops_per_sample * config.total_samples_planned * config.training_progress,
                "mparams": model_stats.get("mparams"),
                "vision_mparams": model_stats.get("vision_mparams"),
                "text_mparams": model_stats.get("text_mparams"),
                "text_mparams_non_emb": model_stats.get("text_mparams_non_emb"),
            }

        if failed:
            return {
                "failed": True,
                "divergence_probability": round(divergence_prob, 3),
                "predictions": None,
                "model_stats": stats,
            }

        row = self.encoder.transform_surrogate(config)

        preds = self.surrogate.predict(row)[0]
        predictions = {
            label: round(float(pred), 3)
            for label, pred in zip(self.targets, preds)
        }

        return {
            "failed": False,
            "divergence_probability": round(divergence_prob, 3),
            "predictions": predictions,
            "model_stats": stats,
        }


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
        total_samples_planned=1280000,
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
        total_samples_planned=12800000,
        training_progress=0.5
    )

    result = vlm_bench.query(config)
    result2 = vlm_bench.query(config2)
    print(result)
    print(result2)
