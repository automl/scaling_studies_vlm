import math
from dataclasses import dataclass
from enum import Enum
from typing import Union

import neps
from neps.space import Parameter
import torch
from optimizers import primo_space, prior_confidence, prior_centers, primo_space_80M
from spaces import (__SCALE_SPACE_CONSTANT_ViT_S_32,
                    ScanPipeline)


class Optimizer(Enum):
    Grid = "grid_search"
    Random = "random_search"
    BO_RW = "bo_rw"
    PriMO = "primo"



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
    fidelities_to_spend: int | None = None

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
        if self.fidelities_to_spend is not None:
            args["fidelities_to_spend"] = self.fidelities_to_spend
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
    # "experimental": NePSConfig(
    #     optimizer=chinchilla_rs_optimizer,
    #     evaluations_to_spend=20,
    #     pipeline_space=ScanPipeline.with_overrides()
    # ),
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
  
    "scan_space_juwels_biased_alpha_0_5": NePSConfig(lambda p: neps.algorithms.random_search(p),
                                                     evaluations_to_spend=100,
                                                     pipeline_space=
                                                     ScanPipeline.with_overrides()
                                                     ),
    "scan_space_final_biased_alpha_0_5": NePSConfig(lambda p: neps.algorithms.random_search(p),
                                                    evaluations_to_spend=100,
                                                    pipeline_space=
                                                    ScanPipeline.with_overrides()
                                                    ),
    "scan_space_large_scale": NePSConfig(lambda p: neps.algorithms.random_search(p),
                                         evaluations_to_spend=1000,
                                         pipeline_space=
                                         ScanPipeline.with_overrides(train_num_samples=neps.Float(
                                             lower=60,
                                             upper=120.0,
                                         ))
                                         ),
    "random_fixed_max_dp": NePSConfig(
        lambda p: neps.algorithms.random_search(p, ignore_fidelity="highest_fidelity"),
        evaluations_to_spend=20,
        pipeline_space=primo_space,
    ),
    "primo": NePSConfig(
        optimizer=lambda p: neps.algorithms.primo(space=p,
                prior_centers=prior_centers,
                mo_selector="epsnet",
                prior_confidences=prior_confidence,
                initial_design_size=5,
                cost_aware=False,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            ),
        fidelities_to_spend=200,
        pipeline_space=primo_space,
    ),
    "bo_rw": NePSConfig(
        optimizer=lambda p: neps.algorithms.bayesian_optimization(pipeline_space=p,
                initial_design_size=5,
                ignore_fidelity=True,
                reference_point=None,
                cost_aware=False,
                device=torch.device("cpu"),
            ),
        evaluations_to_spend=20,
        pipeline_space=primo_space,
    ),
    
    "random_80M": NePSConfig(
        lambda p: neps.algorithms.random_search(p, ignore_fidelity="highest_fidelity"),
        evaluations_to_spend=20,
        pipeline_space=primo_space_80M,
    ),
    "primo_80M": NePSConfig(
        optimizer=lambda p: neps.algorithms.primo(space=p,
                prior_centers=prior_centers,
                mo_selector="epsnet",
                prior_confidences=prior_confidence,
                initial_design_size=5,
                cost_aware=False,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            ),
        fidelities_to_spend=200,
        pipeline_space=primo_space_80M,
    ),
    "bo_rw_80M": NePSConfig(
        optimizer=lambda p: neps.algorithms.bayesian_optimization(pipeline_space=p,
                initial_design_size=5,
                ignore_fidelity=True,
                reference_point=None,
                cost_aware=False,
                device=torch.device("cpu"),
            ),
        evaluations_to_spend=20,
        pipeline_space=primo_space_80M,
    ),


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
    train_num_samples: int
    epochs: int
    name: str
    logs: str
    seed: int
    batch_size: int

    # specific to NePs
    pipeline_id: str | int
    sync_neps: bool

    # hyperparameters from NePs
    lr: float
    wd: float
    warmup_fraction: float
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6

    config_id_to_train_num_samples: dict[str | int, int] = None

    def __post_init__(self):
        # epoch_max_samples = 100_000
        # min_epochs = 1
        max_epochs = 10
        total = self.train_num_samples
        # self.epochs = max(min_epochs, math.ceil(total / epoch_max_samples))
        epoch_num_samples = math.ceil(total * (self.epochs / max_epochs))
        self.train_num_samples = epoch_num_samples
        self.epochs = 1

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
    pipeline_id: str | int

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")
