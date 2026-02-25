import neps

__SCALE_SPACE_CONSTANT_ViT_S_32 = {
    "vision_width": 6,
    "text_width": 48,
    "text_heads": 6
}


class ScanPipeline(neps.PipelineSpace):
    # hps
    lr = neps.Float(
        lower=1.0e-5,
        upper=5.0e-2,
        log=True,
    )
    wd = neps.Float(
        lower=1.0e-6,
        upper=2.0e-1,
        log=True,
    )
    warmup_fraction = neps.Float(
        lower=0.0,
        upper=0.75,
    )
    beta1 = neps.Float(lower=0.9, upper=0.99, log=True)
    beta2 = neps.Float(lower=0.95, upper=0.999, log=True)
    eps = neps.Float(lower=1e-8, upper=1e-6, log=True)

    # scale space
    train_num_samples = neps.Float(
        lower=0.6,
        upper=60.0,
    )
    vision_width = neps.Integer(
        lower=0, upper=8,  # head width is set to 64 by default. Emb. dim. needs to be divisible by 64!
    )
    text_width = neps.Integer(
        lower=0, upper=8,
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
