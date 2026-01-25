from dataclasses import dataclass
from enum import Enum

import neps
from neps.space import Parameter

from train.spaces import (__HP_SPACE, __SCALE_SPACE, __HP_SPACE_LR_CONSTANT, __SCALE_SPACE_CONSTANT_ViT_S_32,
                          __LR_WD_WARMUP_CONSTANT)


class Optimizer(Enum):
    Grid = "grid_search"
    Random = "random_search"


@dataclass
class NePSConfig:
    """
    NePS configuration.

    Args:
        optimizer: optimizer instance.
        evaluations_to_spend: The total number of evaluations to run.
        pipeline_space: The search space for the pipeline.
    """

    optimizer: Optimizer = None
    pipeline_space: dict[str, Parameter] = None
    evaluations_to_spend: int | None = None

    @property
    def run_args(self) -> dict[str, any]:
        args = {
            "optimizer": self.optimizer.value,
            "evaluations_to_spend": self.evaluations_to_spend,
            "pipeline_space": self.pipeline_space,
        }

        return args


HPO_CONFIGS = {
    "grid": NePSConfig(
        optimizer=Optimizer.Grid,
        evaluations_to_spend=2,
        pipeline_space={
            **__HP_SPACE,
            **__SCALE_SPACE
        },
    ),
    "random": NePSConfig(
        optimizer=Optimizer.Random,
        evaluations_to_spend=30,
        pipeline_space={
            **__HP_SPACE,
            **__SCALE_SPACE
        },
    ),
    "experimental": NePSConfig(
        optimizer=Optimizer.Random,
        evaluations_to_spend=1,
        pipeline_space={
            **__HP_SPACE,
            **__SCALE_SPACE
        },
    ),
    "random_ViT-S-32_fixed_lr_307e4_samples": NePSConfig(optimizer=Optimizer.Random,
                                                         evaluations_to_spend=20,
                                                         pipeline_space={
                                                             "lr": neps.Constant(
                                                                 value=5e-4
                                                             ),
                                                             "train_num_samples": neps.Constant(value=3.07),
                                                             **__HP_SPACE_LR_CONSTANT,
                                                             **__SCALE_SPACE_CONSTANT_ViT_S_32
                                                         }),
    "random_ViT-S-32_fixed_lr_128e5_samples": NePSConfig(optimizer=Optimizer.Random,
                                                         evaluations_to_spend=20,
                                                         pipeline_space={
                                                             **__HP_SPACE_LR_CONSTANT,
                                                             **__SCALE_SPACE_CONSTANT_ViT_S_32,
                                                             "lr": neps.Constant(
                                                                 value=1e-3
                                                             ),
                                                             "train_num_samples": neps.Constant(value=12.8),
                                                         }),
    "rs_over_optimizer_params_ViT-S-32_307e4_samples": NePSConfig(optimizer=Optimizer.Random,
                                                                  evaluations_to_spend=20,
                                                                  pipeline_space={
                                                                      **__LR_WD_WARMUP_CONSTANT,
                                                                      "train_num_samples": neps.Constant(value=3.07),
                                                                      **__SCALE_SPACE_CONSTANT_ViT_S_32,
                                                                      "beta2": neps.Float(lower=0.98, upper=0.999),
                                                                      "eps": neps.Float(lower=1e-9, upper=1e-6,
                                                                                        log=True),
                                                                  }),
    "rs_over_optimizer_params_ViT-S-32_128e5_samples": NePSConfig(optimizer=Optimizer.Random,
                                                                  evaluations_to_spend=20,
                                                                  pipeline_space={
                                                                      "lr": neps.Constant(value=1e-3),
                                                                      "warmup_fraction": neps.Constant(value=0.64),
                                                                      "wd": neps.Constant(value=1e-6),
                                                                      "train_num_samples": neps.Constant(value=12.8),
                                                                      **__SCALE_SPACE_CONSTANT_ViT_S_32,
                                                                      "beta2": neps.Float(lower=0.98, upper=0.999),
                                                                      "eps": neps.Float(lower=1e-9, upper=1e-6,
                                                                                        log=True),
                                                                  })
}


@dataclass
class TrainConfig:
    """
    Train configuration.

    Args:
        model: vision backbone to train on.
        train_num_samples: number of training samples to use(this number defines epoch length).
        epochs: number of epochs to train for.
        name: experiment name.
        logs: string pointing to relative path where to store logs.
        seed: random seed for reproducibility.
    """

    model: str
    train_num_samples: str
    epochs: int
    name: str
    logs: str
    seed: int
    batch_size: int

    # specific to NePs
    pipeline_id: int
    sync_neps: bool

    # hyperparameters from NePs
    lr: float
    wd: float
    warmup_fraction: float
    beta2: float = 0.98
    eps: float = 1e-6

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")


@dataclass
class EvalConfig:
    """
        Evaluation configuration.

        Args:
            model: vision backbone to evaluate on.
            seed: random seed for reproducibility.
            name: experiment name.
    """

    model: str
    seed: int
    name: str
    logs: str

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")
