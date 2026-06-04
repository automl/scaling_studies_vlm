import gc
import json
import os
import sys
from pathlib import Path

import neps
from filelock import FileLock
def set_workspace_dir(workspace_dir: str):
    """Set the workspace directory for disk cache location."""
    global _workspace_dir
    _workspace_dir = workspace_dir




primo_space = dict(
    lr=neps.HPOFloat(
        lower=1.0e-4,
        upper=1.0e-2,
        log=True,
    ),
    warmup_fraction=neps.HPOFloat(
        lower=0.0,
        upper=0.75,
    ),
    wd=neps.HPOConstant(1.0e-6),
    beta1 = neps.HPOConstant(0.9),
    beta2 = neps.HPOConstant(0.999),
    eps = neps.HPOConstant(1.0e-6),
    train_num_samples = neps.HPOConstant(1),
    vision_width = neps.HPOInteger(
        lower=0, upper=4, 
    ),
    text_width = neps.HPOInteger(
        lower=0, upper=4,  # assuming number of heads is fixed to 8
    ),
    epochs = neps.HPOInteger(
        lower=1,
        upper=10,
        is_fidelity=True,
    ),
    text_heads = neps.HPOConstant(8),
    batch_size = neps.HPOConstant(512),
)

 
prior_confidence = {
    "good":{
        "lr": 0.75,
        "warmup_fraction": 0.75,
        "vision_width": 0.75,
        "text_width": 0.75,
    }
}

prior_centers = {
    "good": {
        "lr": 0.001,
        "warmup_fraction": 0.75,
        "vision_width": 3,
        "text_width": 2,
    }
}