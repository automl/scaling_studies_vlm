from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import neps
from neps.space import Parameter

from optimizers import chinchilla_optimizer, ChichillaPipeline, \
    chinchilla_rs_optimizer, chinchilla_optimizer_fine_grained_dp, \
    bo_guided_scaling_optimizer, chinchilla_optimizer_fine_grained_dp, \
    chinchilla_optimizer_aproach1
from spaces import (__SCALE_SPACE_CONSTANT_ViT_S_32,
                    ScanPipeline)


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

    optimizer: Union[Optimizer, callable] = None
    pipeline_space: dict[str, Parameter] = None
    evaluations_to_spend: int | None = None
    cost_to_spend: int | None = None

    def __post_init__(self):
        if isinstance(self.optimizer, Optimizer):
            self.optimizer = self.optimizer.value

    @property
    def run_args(self) -> dict[str, any]:
        args = {
            "optimizer": self.optimizer,
            "evaluations_to_spend": self.evaluations_to_spend,
            "pipeline_space": self.pipeline_space,
            "cost_to_spend": self.cost_to_spend,
        }
        return args


HPO_CONFIGS = {
    "grid": NePSConfig(
        optimizer=Optimizer.Grid,
        evaluations_to_spend=2,
        pipeline_space=ScanPipeline.with_overrides()
    ),
    "random": NePSConfig(
        optimizer=Optimizer.Random,
        evaluations_to_spend=30,
        pipeline_space=ScanPipeline.with_overrides()
    ),
    "experimental": NePSConfig(
        optimizer=chinchilla_rs_optimizer,
        evaluations_to_spend=20,
        pipeline_space=ScanPipeline.with_overrides()
    ),
    "random_ViT-S-32_fixed_lr_307e4_samples": NePSConfig(optimizer=Optimizer.Random,
                                                         evaluations_to_spend=20,
                                                         pipeline_space=ScanPipeline.with_overrides(lr=5e-4,
                                                                                                    train_num_samples=3.07,
                                                                                                    **__SCALE_SPACE_CONSTANT_ViT_S_32)
                                                         ),

    "random_ViT-S-32_fixed_lr_128e5_samples": NePSConfig(optimizer=Optimizer.Random,
                                                         evaluations_to_spend=20,
                                                         pipeline_space=ScanPipeline.with_overrides(lr=1e-3,
                                                                                                    train_num_samples=12.8,
                                                                                                    **__SCALE_SPACE_CONSTANT_ViT_S_32)
                                                         ),
    "rs_over_optimizer_params_ViT-S-32_307e4_samples": NePSConfig(optimizer=Optimizer.Random,
                                                                  evaluations_to_spend=20,
                                                                  pipeline_space=
                                                                  ScanPipeline.with_overrides(lr=5e-4,
                                                                                              warmup_fraction=0.5,
                                                                                              wd=1.0e-6,
                                                                                              train_num_samples=3.07,
                                                                                              beta2=neps.Float(
                                                                                                  lower=0.98,
                                                                                                  upper=0.999),
                                                                                              eps=neps.Float(lower=1e-9,
                                                                                                             upper=1e-6,
                                                                                                             log=True),
                                                                                              **__SCALE_SPACE_CONSTANT_ViT_S_32)
                                                                  ),
    "rs_over_optimizer_params_ViT-S-32_128e5_samples": NePSConfig(optimizer=Optimizer.Random,
                                                                  evaluations_to_spend=20,
                                                                  pipeline_space=
                                                                  ScanPipeline.with_overrides(lr=1e-3,
                                                                                              warmup_fraction=0.64,
                                                                                              wd=1.0e-6,
                                                                                              train_num_samples=12.8,
                                                                                              beta2=neps.Float(
                                                                                                  lower=0.98,
                                                                                                  upper=0.999),
                                                                                              eps=neps.Float(lower=1e-9,
                                                                                                             upper=1e-6,
                                                                                                             log=True),
                                                                                              **__SCALE_SPACE_CONSTANT_ViT_S_32)
                                                                  ),
    "scale_space_only_search": NePSConfig(lambda p: neps.algorithms.random_search(p, use_priors=True),
                                          evaluations_to_spend=50,
                                          pipeline_space=
                                          ScanPipeline.with_overrides(lr=2e-3, warmup_fraction=0.5, wd=1.0e-6)
                                          ),
    "scale_space_only_search_biased_scale_space": NePSConfig(
        optimizer=lambda p: neps.algorithms.random_search(p, use_priors=True),
        evaluations_to_spend=50,
        pipeline_space=
        ScanPipeline.with_overrides(lr=2e-3, warmup_fraction=0.5, wd=1.0e-6)),
    "scale_space_only_search_biased_scale_space_alpha_0_5": NePSConfig(
        optimizer=lambda p: neps.algorithms.random_search(p, use_priors=True),
        evaluations_to_spend=40,
        pipeline_space=
        ScanPipeline.with_overrides(lr=2e-3, warmup_fraction=0.5, wd=1.0e-6)),
    "gs_ViT-XS-32": NePSConfig(
        optimizer=chinchilla_optimizer,
        evaluations_to_spend=100,
        pipeline_space=ChichillaPipeline.with_overrides(
            wd=1.0e-6,
            vision_width=256 / 64,
            text_width=256 / 8,
            text_heads=4,
        )
    ),
    "gs_ViT-Micro-32": NePSConfig(
        optimizer=chinchilla_optimizer,
        evaluations_to_spend=27,
        pipeline_space=ChichillaPipeline.with_overrides(
            wd=1.0e-6,
            vision_width=256 / 64,
            text_width=192 / 8,
            text_heads=4,
        )
    ),
    "gs_ViT-XXS-32": NePSConfig(
        optimizer=chinchilla_optimizer,
        evaluations_to_spend=0,
        pipeline_space=ChichillaPipeline.with_overrides(
            wd=1.0e-6,
            vision_width=192 / 64,
            text_width=192 / 8,
            text_heads=4,
        )
    ),
    "gs_ViT-Tiny-32": NePSConfig(
        optimizer=chinchilla_optimizer_fine_grained_dp,
        evaluations_to_spend=0,
        pipeline_space=ChichillaPipeline.with_overrides(
            wd=1.0e-6,
            vision_width=128 / 64,
            text_width=128 / 8,
            text_heads=4,
        )
    ),
    "scale_space_only_minimal_capacity_probe": NePSConfig(
        optimizer=chinchilla_optimizer_aproach1,
        evaluations_to_spend=100,
        pipeline_space=
        ScanPipeline.with_overrides(
            lr=2e-3,
            warmup_fraction=0.5,
            wd=1.0e-6,
            train_num_samples=neps.Float(
                lower=1,
                upper=80.0,
            ),
            vision_width=neps.Integer(
                lower=1, upper=4,
            ),
            text_width=neps.Integer(
                lower=1, upper=4,
            )
        ),
    ),
    "constrained_bo": NePSConfig(
        optimizer=bo_guided_scaling_optimizer(max_evaluation_flops=209161783),
        cost_to_spend=195017516,  # GFLOPS
        pipeline_space=ScanPipeline.with_overrides(
            train_num_samples=neps.Float(
                lower=1,
                upper=80.0,
            ),
            vision_width=neps.Integer(
                lower=1, upper=4,
            ),
            text_width=neps.Integer(
                lower=1, upper=4,
            )
        ),
    ),
    "constrained_bo_cost_gp": NePSConfig(
        optimizer=bo_guided_scaling_optimizer(max_evaluation_flops=209161783, cost_aware="log"),
        cost_to_spend=195017516,  # GFLOPS
        pipeline_space=ScanPipeline.with_overrides(
            train_num_samples=neps.Float(
                lower=1,
                upper=80.0,
            ),
            vision_width=neps.Integer(
                lower=1, upper=4,
            ),
            text_width=neps.Integer(
                lower=1, upper=4,
            )
        ),
    ),
    "constrained_rs": NePSConfig(
        optimizer=chinchilla_rs_optimizer,
        cost_to_spend=209161783,  # GFLOPS
        pipeline_space=
        ScanPipeline.with_overrides(
            train_num_samples=neps.Float(
                lower=1,
                upper=80.0,
            ),
            vision_width=neps.Integer(
                lower=1, upper=4,
            ),
            text_width=neps.Integer(
                lower=1, upper=4,
            )
        ),
    ),
    "scan_space_juwels_biased_alpha_0_5": NePSConfig(lambda p: neps.algorithms.random_search(p),
                                                     evaluations_to_spend=100,
                                                     pipeline_space=
                                                     ScanPipeline.with_overrides()
                                                     )
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
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6
    config_id_to_train_num_samples: dict[int, int] = None

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
    pipeline_id: int

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")
