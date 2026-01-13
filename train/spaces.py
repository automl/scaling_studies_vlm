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
    "samples_seen": neps.Float(
        lower=0.6,
        upper=60,
    ),

}
