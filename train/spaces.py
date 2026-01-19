import neps

__HP_SPACE = {
    "lr": neps.Float(
        lower=1.0e-6,
        upper=1,
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
        upper=60,
    ),
    "vision_width": neps.Categorical(
        [256, 320, 384, 448, 512] # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
    ),
    "text_width": neps.Integer(
        lower=32, upper=64  # assuming number of heads is fixed to 8
    ),

}
