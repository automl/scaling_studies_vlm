import math
from dataclasses import dataclass, asdict, field
from enum import Enum


@dataclass
class VLMConfig:
    lr: float
    wd: float
    beta1: float
    beta2: float
    warmup_fraction: float
    eps: float
    vision_width: int
    text_width: int
    total_samples_planned: int
    training_progress: float

    lr_ratio: float = field(init=False)
    global_batch_size: int = field(init=False)

    BATCH_SIZE_RULES = [
        {"min": 0.0, "max": 3404800.0, "bs": 512},
        {"min": 3404800.0, "max": 14592000.0, "bs": 2048},
        {"min": 14592000.0, "max": 70400000.0, "bs": 4096},
        {"min": 70400000.0, "max": 145920000.0, "bs": 8192},
        {"min": 145920000.0, "max": 340480000.0, "bs": 16384},
        {"min": 340480000.0, "max": 640000000.0, "bs": 32768},
        {"min": 640000000.0, "max": 704000000.0, "bs": 45056},
        {"min": 704000000.0, "max": 1459200000.0, "bs": 65536},
        {"min": 1459200000.0, "max": None, "bs": 90112},
    ]

    def __post_init__(self):
        self.global_batch_size = self._derive_global_batch_size()
        self.lr_ratio = self.compute_lr_from_progress()

    def _derive_global_batch_size(self) -> int:
        n = self.total_samples_planned

        for rule in self.BATCH_SIZE_RULES:
            lower = rule["min"]
            upper = rule["max"]

            if upper is None:
                if n >= lower:
                    return rule["bs"]
            else:
                if lower <= n < upper:
                    return rule["bs"]

        raise ValueError(
            f"Could not determine global_batch_size for total_samples_planned={n}"
        )

    def compute_lr_from_progress(self):
        p = self.training_progress
        w = self.warmup_fraction
        base_lr = self.lr

        if p < w:
            current_lr = base_lr * (p / w)
        else:
            x = (p - w) / (1 - w)
            current_lr = base_lr * 0.5 * (1 + math.cos(math.pi * x))

        lr_ratio = current_lr / base_lr

        return lr_ratio

    def to_dict(self) -> dict:
        return asdict(self)


class Target(str, Enum):
    VAL_LOSS = "val_loss"
    TEST_LOSS = "test_loss"

    # TODO add downstream tasks

    @classmethod
    def all(cls):
        return [t.value for t in cls]
