import neps

__SCALE_SPACE_CONSTANT_ViT_S_32 = {
    "vision_width": 6,
    "text_width": 48,
    "text_heads": 6
}


class ScanPipeline(neps.PipelineSpace):
    # hps
    lr = neps.Float(
        lower=1.0e-6,
        upper=1.0e-1,
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
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-6

    # scale space
    train_num_samples = neps.Float(
        # inverse mapping from uniform distribution to either warped log distribution or uniform distribution with ranges 0.6-60m
        lower=0.0,
        upper=1.0,
    )
    vision_width = neps.Integer(
        lower=1, upper=8,  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
        prior=3,
        prior_confidence="medium"
    )
    text_width = neps.Integer(
        lower=1, upper=8,
        prior=3,
        prior_confidence="medium"
    )

    @classmethod
    def with_overrides(cls, **overrides):
        class CustomPipeline(cls):
            pass

        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name != 'with_overrides':
                attr_value = getattr(cls, attr_name)
                if not callable(attr_value):
                    setattr(CustomPipeline, attr_name, attr_value)

        for key, value in overrides.items():
            setattr(CustomPipeline, key, value)

        return CustomPipeline()
