import neps

__HP_SPACE = {
    "lr": neps.Float(
        lower=1.0e-6,
        upper=1,
        log=True,
        prior=5.0e-4
    ),
    "weight_decay": neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
        log=True,
        prior=2.0e-1
    ),
    "gradient_clipping": neps.Categorical(
        choices=[True, False],
        prior=False
    ),
    "warmup_fraction": neps.Float(
        lower=0.0,
        upper=0.5,
        prior=0.2,
    ),
}

__FIXED_SCALE_SPACE = {
    "img_patch_size": 32,
    "emb_projection": 512,
    "precision": "bf16"
}
