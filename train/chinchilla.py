import gc
import sys

import neps
import torch

import iterator_params
from configs_iterator import normalize_scan_space, get_total_compute
from utils import get_local_model_str, create_default_model, build_model_cfg_temp


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
    # create the model config path temporarily
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0, normalize_train_num_samples=False)
    model_cfg_dir = build_model_cfg_temp(**configs_denormalized)
    model_cfg = get_local_model_str(model_cfg_dir)
    return get_total_compute(
        model_str=model_cfg,
        train_num_samples=kwargs["train_num_samples"],
        precision="amp",
    )


def seen_datapoints_estimator(**kwargs) -> int:
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0, normalize_train_num_samples=False)
    return configs_denormalized["epochs"] * configs_denormalized["train_num_samples"]


def parameters_estimator(**kwargs) -> int:
    configs_denormalized = normalize_scan_space(kwargs, bias_scale_sampling=False, alpha=0, normalize_train_num_samples=False)
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


from neps.optimizers.algorithms import chinchilla_guided_scaling, grid_search, random_search

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
                                                                  flops_estimator=None,
                                                                  base_optimizer=random_search(space, use_priors=True))


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
