class BaseConfig:

    def to_dict(self) -> dict:
        raise NotImplementedError("Subclasses must implement to_dict()")

    def validate(self) -> None:
        raise NotImplementedError("Subclasses must implement validate()")
