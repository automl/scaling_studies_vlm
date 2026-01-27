import neps

__HP_SPACE = {
    "lr": neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
        log=True,
    ),
    "wd": neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
        log=True,
    ),
    "warmup_fraction": neps.Float(
        lower=0.0,
        upper=0.5,
    ),
    "beta2": neps.HPOConstant(value=0.98),
    "eps": neps.HPOConstant(value=1.0e-6)
}

__SCALE_SPACE = {
    "train_num_samples": neps.Float(
        lower=0.6,
        upper=60,  # in million
    ),
    "vision_width": neps.Integer(
        lower=4, upper=8  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
    ),
    "text_width": neps.Integer(
        lower=32, upper=64  # assuming number of heads is fixed to 8
    ),
    "text_heads": neps.HPOConstant(value=8)

}

__HP_SPACE_LR_CONSTANT = {
    "wd": neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
        log=True,
    ),
    "warmup_fraction": neps.Float(
        lower=0.1,
        upper=0.65,
    ),
}

__LR_WD_WARMUP_CONSTANT = {
    "lr": neps.HPOConstant(
        value=5e-4
    ),
    "wd": neps.HPOConstant(
        value=1.0e-6,

    ),
    "warmup_fraction": neps.HPOConstant(
        value=0.5,
    ),

}

__SCALE_SPACE_CONSTANT_ViT_S_32 = {
    "vision_width": neps.HPOConstant(
        value=6,
    ),
    "text_width": neps.HPOConstant(
        value=48,
    ),
    "text_heads": neps.HPOConstant(value=6)
}
