from dataclasses import dataclass, asdict

from surrogate_benchmark.base_config import BaseConfig


@dataclass
class TabPFNConfig(BaseConfig):
    total_cells: int
    effective_batch_size: int
    lr: float
    max_features: int
    embedding_size: int
    num_layers: int
    num_datapoints_max: int
    weight_decay: float

    def to_dict(self) -> dict:
        key_mapping = {
            "total_cells": "total_cells",
            "effective_batch_size": "config/effective_batch_size",
            "lr": "config/lr",
            "max_features": "config/max_features",
            "embedding_size": "config/model_config.embedding_size",
            "num_layers": "config/model_config.num_layers",
            "num_datapoints_max": "config/num_datapoints_max",
            "weight_decay": "config/weight_decay",
        }

        raw = asdict(self)
        return {
            key_mapping.get(k, k): v
            for k, v in raw.items()
        }
