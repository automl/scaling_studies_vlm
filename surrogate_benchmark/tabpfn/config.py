from dataclasses import dataclass


@dataclass
class TabPFNConfig:
    effective_batch_size: int
    lr: float
    max_features: int
    embedding_size: int
    num_layers: int
    num_datapoints_max: int
    weight_decay: float
