import json
from importlib.resources import files

from tqdm import tqdm

from surrogate_benchmark.config import VLMConfig, Target
from surrogate_benchmark.divergence_surrogate.data import DivergenceDataset
from surrogate_benchmark.divergence_surrogate.predictors.xgb import BinaryBaggingEnsemble
from surrogate_benchmark.feature_encoder import ConfigFeatureMapper
from surrogate_benchmark.performance_surrogate.data import SurrogateDataset
from surrogate_benchmark.performance_surrogate.predictors.pfn import TabPFNModel


class VLMBenchmark:
    def __init__(
            self,
            targets: list[Target] | None = None,
            device: str = "auto",
    ):
        targets = Target.all() if targets is None else [t.value for t in targets]

        train_path = files("surrogate_benchmark.performance_surrogate") \
            .joinpath("splits/train.csv")

        test_path = files("surrogate_benchmark.performance_surrogate") \
            .joinpath("splits/test.csv")

        self.surrogate_dataset = SurrogateDataset(
            train_csv_path=str(train_path),
            test_csv_path=str(test_path),
            targets=targets,
            seed=42,
            include_intermediate_points=True,
        )

        self.config_feature_mapper = ConfigFeatureMapper(
            surrogate_feature_order=self.surrogate_dataset.features,
            divergence_feature_order=DivergenceDataset.DEFAULT_FEATURES,
            apply_log=self.surrogate_dataset.apply_log_transform,
            log_columns=self.surrogate_dataset.DEFAULT_LOG_COLUMNS,
        )

        self.targets = self.surrogate_dataset.targets

        model_dir = files("surrogate_benchmark.divergence_surrogate") \
            .joinpath("xgb_models")

        self.divergence_surrogate = BinaryBaggingEnsemble(model_dir=str(model_dir))

        self.performance_surrogate = TabPFNModel(device=device)

    def query(self, config: VLMConfig) -> dict:
        divergence_prob, failed = self._predict_divergence(config)
        stats = self._get_model_stats(config)

        if failed:
            return {
                "failed": True,
                "divergence_probability": round(divergence_prob, 3),
                "predictions": None,
                "model_stats": stats,
            }

        predictions = self._predict_performance(config)

        return {
            "failed": False,
            "divergence_probability": round(divergence_prob, 3),
            "predictions": predictions,
            "model_stats": stats,
        }

    def _get_model_stats(self, config: VLMConfig) -> dict:
        stats_key = f"text_{config.text_width}_vision_{config.vision_width}"

        path = files("surrogate_benchmark").joinpath("gflops_mparams.json")

        with open(str(path), "r", encoding="utf-8") as f:
            models_stats = json.load(f)

        model_stats = models_stats.get(stats_key)
        if model_stats is None:
            return {}

        gflops_per_sample = model_stats.get("gflops_per_sample")

        return {
            "gflops_per_sample": gflops_per_sample,
            "total_gflops": gflops_per_sample * config.total_samples_planned * config.training_progress,
            "mparams": model_stats.get("mparams"),
            "vision_mparams": model_stats.get("vision_mparams"),
            "text_mparams": model_stats.get("text_mparams"),
            "text_mparams_non_emb": model_stats.get("text_mparams_non_emb"),
        }

    def _predict_divergence(self, config: VLMConfig) -> tuple[float, bool]:
        row = self.config_feature_mapper.to_divergence_features(config)
        prob = float(self.divergence_surrogate.predict_proba(row)[0])
        return prob, prob >= 0.5

    def _predict_performance(self, config: VLMConfig) -> dict:
        row = self.config_feature_mapper.to_surrogate_features(config)
        X, y = self.surrogate_dataset.get_all_data()

        predictions = {}

        for i, target in enumerate(tqdm(self.targets, desc="Evaluating targets")):
            y_target = y[:, i]

            self.performance_surrogate.fit(X, y_target)
            pred = self.performance_surrogate.predict(row)[0]

            predictions[target] = round(float(pred), 3)

        return predictions


if __name__ == "__main__":
    vlm_bench = VLMBenchmark()

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
        lr=0.03,
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
