import sys

from train.open_clip_train import params
from train.open_clip_train.main import main as train_model


if __name__ == "__main__":
    args = params.parse_args(sys.argv[1:])
    train_model(args)
