import gc
import json
import os
import sys
from pathlib import Path

import neps
from filelock import FileLock
from neps.optimizers.algorithms import chinchilla_guided_scaling, grid_search, random_search, bo_guided_scaling

import torch

import iterator_params
from configs_iterator import normalize_scan_space, get_total_compute
from utils import get_local_model_str, create_default_model, build_model_cfg_temp


class PriMOPipeline(neps.PipelineSpace):
    lr = neps.Float(
        lower=1.0e-4,
        upper=1.0e-2,
        log=True,
    )
    warmup_fraction = neps.Float(
        lower=0.0,
        upper=0.75,
    )
    wd=1.0e-6
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-6
    train_num_samples = 1
    vision_width = neps.Integer(
        lower=4, upper=8,  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
        prior=4,
        prior_confidence="medium"
    )
    text_width = neps.Integer(
        lower=32, upper=64,  # assuming number of heads is fixed to 8
        prior=32,
        prior_confidence="high"
    )
    text_heads = 8
    epochs = 1
    batch_size = 512

    @classmethod
    def with_overrides(cls, **overrides):
        """Create an instance with field overrides."""

        class CustomPipeline(cls):
            pass

        # Copy all class attributes from parent first
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name != 'with_overrides':
                attr_value = getattr(cls, attr_name)
                # Skip methods
                if not callable(attr_value):
                    setattr(CustomPipeline, attr_name, attr_value)

        # Apply overrides
        for key, value in overrides.items():
            setattr(CustomPipeline, key, value)

        return CustomPipeline()


prior_confidence = {
    "lr": 0.75,
    "warmup_fraction": 0.75,
    "vision_width": 0.75,
    "text_width": 0.75,
}