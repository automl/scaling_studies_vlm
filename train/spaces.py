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
}

__SCALE_SPACE = {
    "train_num_samples": neps.Float(
        lower=0.6,
        upper=60, # in million
    ),
    "vision_width": neps.Integer(
        lower=4, upper=8  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
    ),
    "text_width": neps.Integer(
        lower=32, upper=64  # assuming number of heads is fixed to 8
    ),
    "text_heads": neps.Constant(value=8)

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

__SCALE_SPACE_CONSTANT_ViT_S_32 = {
    "vision_width": neps.Constant(
        value=6,
    ),
    "text_width": neps.Constant(
        value=48,
    ),
    "text_heads": neps.Constant(value=6)
}
