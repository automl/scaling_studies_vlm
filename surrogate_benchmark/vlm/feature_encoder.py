import numpy as np

from surrogate_benchmark.vlm.config import VLMConfig


class ConfigFeatureMapper:
    def __init__(
        self,
        surrogate_feature_order,
        divergence_feature_order,
        apply_log=False,
        log_columns=None,
    ):
        self.surrogate_feature_order = surrogate_feature_order
        self.divergence_feature_order = divergence_feature_order
        self.apply_log = apply_log
        self.log_columns = set(log_columns or [])

    def _build_row(self, config: VLMConfig, feature_order: list[str]) -> np.ndarray:
        config_dict = config.to_dict()

        missing = [name for name in feature_order if name not in config_dict]
        if missing:
            raise ValueError(f"Missing features in config: {missing}")

        row = []
        for name in feature_order:
            value = config_dict[name]

            if self.apply_log and name in self.log_columns:
                value = np.log(value)

            row.append(value)

        return np.array(row, dtype=float).reshape(1, -1)

    def to_surrogate_features(self, config: VLMConfig) -> np.ndarray:
        return self._build_row(config, self.surrogate_feature_order)

    def to_divergence_features(self, config: VLMConfig) -> np.ndarray:
        return self._build_row(config, self.divergence_feature_order)