#!/bin/bash

SCRIPT="train.py"

export PYTHONPATH="C:\Users\Donat\Documents\Master\Thesis\vlm scaling laws repository;$PYTHONPATH"

python "$SCRIPT" \
      --model autogluon \
      --seed 42 \
      --include_intermediate_points