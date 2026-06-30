import neps
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

primo_space_80M = primo_space.copy()
primo_space_80M.update({
    "train_num_samples": neps.HPOConstant(80),
})

prior_confidence = {
    "good-n_param":{
        "lr": 0.75,
        "warmup_fraction": 0.75,
        "vision_width": 0.75,
        "text_width": 0.75,
    },
    "good-cifar10_acc5":{
        "lr": 0.75,
        "warmup_fraction": 0.75,
        "vision_width": 0.75,
        "text_width": 0.75,
    },
    "good-imagenet1k_acc5":{
        "lr": 0.75,
        "warmup_fraction": 0.75,
        "vision_width": 0.75,
        "text_width": 0.75,
    },
}

prior_centers = {
    "good-n_param": {
        "lr": 0.01,
        "warmup_fraction": 0.31,
        "vision_width": 0,
        "text_width": 0,
    },
    "good-cifar10_acc5": {
        "lr": 2.13e-5,
        "warmup_fraction": 0.68,
        "vision_width": 1,
        "text_width": 0,
    },
    "good-imagenet1k_acc5": {
        "lr": 1.12e-5,
        "warmup_fraction": 0.36,
        "vision_width": 2,
        "text_width": 1,
    }
}