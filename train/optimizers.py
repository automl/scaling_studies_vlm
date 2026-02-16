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

# Global workspace directory for cache location (set via set_workspace_dir)
_workspace_dir = None

# In-memory cache for fast repeated lookups within same process
_flop_cache = {}


class FlopDiskCache:
    """Disk-based FLOP cache with file locking for multiprocess safety."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "flop_cache"
        self.cache_file = self.cache_dir / "flop_cache.json"
        self.lock_file = self.cache_dir / "flop_cache.lock"
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.cache_file.exists():
            with FileLock(self.lock_file):
                if not self.cache_file.exists():
                    self.cache_file.write_text("{}")
    
    def _key_to_str(self, key: tuple) -> str:
        """Convert tuple key to string for JSON storage."""
        return json.dumps(list(key), sort_keys=True)
    
    def _str_to_key(self, key_str: str) -> tuple:
        """Convert string key back to tuple."""
        return tuple(tuple(item) if isinstance(item, list) else item for item in json.loads(key_str))
    
    def get(self, key: tuple) -> float | None:
        """Get cached FLOP value for key. Returns None if not found."""
        key_str = self._key_to_str(key)
        with FileLock(self.lock_file):
            try:
                cache_data = json.loads(self.cache_file.read_text())
                return cache_data.get(key_str)
            except (json.JSONDecodeError, FileNotFoundError):
                return None
    
    def set(self, key: tuple, value: float):
        """Store FLOP value for key."""
        key_str = self._key_to_str(key)
        with FileLock(self.lock_file):
            try:
                cache_data = json.loads(self.cache_file.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                cache_data = {}
            cache_data[key_str] = value
            self.cache_file.write_text(json.dumps(cache_data, indent=2))


def set_workspace_dir(workspace_dir: str):
    """Set the workspace directory for disk cache location."""
    global _workspace_dir
    _workspace_dir = workspace_dir


def get_disk_cache() -> FlopDiskCache | None:
    """Get disk cache instance if workspace_dir is set."""
    if _workspace_dir is not None:
        return FlopDiskCache(_workspace_dir)
    return None


class ChichillaPipeline(neps.PipelineSpace):
    lr = neps.Float(
        lower=1.0e-4,
        upper=1.0e-2,
        log=True,
    )
    wd = neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
        log=True,
    )
    warmup_fraction = neps.Float(
        lower=0.0,
        upper=0.75,
    )
    beta2 = 0.98
    eps = 1.0e-6
    train_num_samples = neps.Integer(
        # inverse mapping from uniform distribution to either warped log distribution or uniform distribution with ranges 0.6-60m
        lower=60_000,
        upper=60_000_000,
    )
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


def flop_estimator(**kwargs) -> float:
    cache_key = tuple(sorted(kwargs.items()))
    
    # Check in-memory cache first (fastest)
    if cache_key in _flop_cache:
        print(f"Using in-memory cached FLOP estimation")
        return _flop_cache[cache_key]
    
    # Check disk cache
    disk_cache = get_disk_cache()
    if disk_cache is not None:
        print(f"Using disk cached FLOP estimation")
        cached_value = disk_cache.get(cache_key)
        if cached_value is not None:
            print(f"Cache hit on disk for FLOP estimation with key: {cache_key} -> {cached_value}")
            _flop_cache[cache_key] = cached_value  # Store in memory for faster access
            return cached_value
    
    # Compute FLOPs
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0)
    model_cfg_dir = build_model_cfg_temp(**configs_denormalized)
    model_cfg = get_local_model_str(model_cfg_dir)
    result = get_total_compute(
        model_str=model_cfg,
        train_num_samples=kwargs["train_num_samples"],
        precision="amp",
    )
    
    # Cache the result in memory
    _flop_cache[cache_key] = result
    
    # Cache the result on disk
    if disk_cache is not None:
        disk_cache.set(cache_key, result)
    
    return result


def seen_datapoints_estimator(**kwargs) -> int:
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0)
    return configs_denormalized["epochs"] * configs_denormalized["train_num_samples"]


def parameters_estimator(**kwargs) -> int:
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0)
    model_cfg_dir = build_model_cfg_temp(**configs_denormalized)
    model_cfg = get_local_model_str(model_cfg_dir)
    model = create_default_model(model_cfg)
    if torch.cuda.is_available():
        model = model.cuda()
    try:
        return sum(p.numel() for p in model.parameters())
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def evaluate_pipeline(pipeline_id, pipeline_directory, previous_pipeline_directory, **kwargs):
    return None



chinchilla_optimizer = lambda space: chinchilla_guided_scaling(
    space,
    params_estimator=parameters_estimator,
    seen_datapoints_estimator=seen_datapoints_estimator,
    flops_estimator=flop_estimator,
    base_optimizer=grid_search(
        space,
        size_per_numerical_dimension=3,
    )
)

chinchilla_rs_optimizer = lambda space: chinchilla_guided_scaling(space,
                                                                  params_estimator=None,
                                                                  seen_datapoints_estimator=None,
                                                                  flops_estimator=flop_estimator,
                                                                  max_evaluation_flops=209161783,
                                                                  base_optimizer=random_search(space, use_priors=False))


chinchilla_optimizer_fine_grained_dp = lambda space: chinchilla_guided_scaling(
    space,
    params_estimator=parameters_estimator,
    seen_datapoints_estimator=seen_datapoints_estimator,
    flops_estimator=flop_estimator,
    base_optimizer=grid_search(
        space,
        size_per_numerical_dimension=dict(train_num_samples=100, warmup_fraction=3, lr=3),
    )
)

chinchilla_optimizer_aproach1 = lambda space: chinchilla_guided_scaling(
    space,
    params_estimator=parameters_estimator,
    seen_datapoints_estimator=seen_datapoints_estimator,
    flops_estimator=flop_estimator,
    base_optimizer=grid_search(
        space,
        size_per_numerical_dimension=dict(train_num_samples=4, vision_width=4, text_width=4),
    )
)

def bo_guided_scaling_optimizer(max_evaluation_flops: float, cost_aware: str | bool = False):
    return lambda pipeline_space: bo_guided_scaling(
    pipeline_space,
    params_estimator=parameters_estimator,
    seen_datapoints_estimator=seen_datapoints_estimator,
    flops_estimator=flop_estimator,
    max_evaluation_flops=max_evaluation_flops,
    max_target_flops=max_evaluation_flops,
    device='cpu',
    cost_aware=cost_aware,
)

if __name__ == "__main__":
    # test run
    args = iterator_params.parse_args(sys.argv[1:])

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=ChichillaPipeline(),
        root_directory=args.root_dir,
        cost_to_spend=100000000,
        overwrite_root_directory=True,
        optimizer=chinchilla_optimizer,
    )
