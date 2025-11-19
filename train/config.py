from dataclasses import dataclass
from enum import Enum

import neps
from neps.optimizers import GridSearch
from neps.search_spaces import SearchSpace
from neps.search_spaces.parameter import Parameter


class Searcher(Enum):
    Grid = "grid_search"


@dataclass
class NePSConfig:
    """
    NePS configuration.

    Args:
        searcher: search instance.
        max_evaluations_total: The total number of evaluations to run.
        pipeline_space: The search space for the pipeline.
    """

    searcher: Searcher = None
    pipeline_space: dict[str, Parameter] = None
    max_evaluations_total: int | None = None
    grid_step_size: int = 10

    def _build_searcher(self):
        if self.searcher is None:
            return None

        if self.searcher == Searcher.Grid:
            space = SearchSpace(**self.pipeline_space)
            return GridSearch(
                pipeline_space=space,
                grid_step_size=self.grid_step_size,
            )

        raise ValueError(f"Unsupported searcher: {self.searcher}")

    @property
    def run_args(self) -> dict[str, any]:
        args = {
            "searcher": self._build_searcher(),
            "max_evaluations_total": self.max_evaluations_total,
        }

        return args


GRID_SPACE = NePSConfig(
    searcher=Searcher.Grid,
    max_evaluations_total=2,
    pipeline_space={
        "lr": neps.FloatParameter(
            lower=1e-6,
            upper=1,
            default=1e-3,
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

    # hyperparameters from NePs
    lr: float

    def __post_init__(self):
        if self.logs and "train/" in self.logs:
            self.logs = self.logs.replace("train/", "")
