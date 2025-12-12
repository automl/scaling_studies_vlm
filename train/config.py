from dataclasses import dataclass
from enum import Enum

import neps
from neps.space import Parameter


class Optimizer(Enum):
    Grid = "grid_search"


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
    grid_step_size: int = 10

    @property
    def run_args(self) -> dict[str, any]:
        args = {
            "optimizer": self.optimizer.value,
            "evaluations_to_spend": self.evaluations_to_spend,
            "pipeline_space": self.pipeline_space,
        }

        return args


GRID_SPACE = NePSConfig(
    optimizer=Optimizer.Grid,
    evaluations_to_spend=2,
    pipeline_space={
        "lr": neps.Float(
            lower=1e-6,
            upper=1,
            log=True,
        ),
    },
)


@dataclass
class TrainConfig:
    """
    Train configuration.

    Args:
        model: vision backbone to train on.
        train_num_samples: number of training samples to use(this number defines epoch length).
        epochs: number of epochs to train for.
        logs: string pointing to relative path where to store logs.
        seed: random seed for reproducibility.
    """

    # static fields from args
    model: str
    train_num_samples: str
    epochs: int
    logs: str
    seed: int

    # specific to NePs
    pipeline_id: int
    sync_neps: bool

    # hyperparameters from NePs
    lr: float

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")


class EvalTask(str, Enum):
    ZEROSHOT_CLASSIFICATION = "zeroshot_classification"
    RETRIEVAL = "zeroshot_retrieval"
    LINEAR_PROBE = "linear_probe"


class EvalDataset(str, Enum):
    pass


class ZeroShotDataset(EvalDataset):
    CIFAR10 = "cifar10"
    IMAGENET = "imagenet1k"


class RetrievalDataset(EvalDataset):
    MS_COCO = "mscoco_captions"
