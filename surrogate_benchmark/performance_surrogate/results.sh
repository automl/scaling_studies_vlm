#!/bin/bash

SCRIPT="train.py"

export PYTHONPATH="C:\Users\Donat\Documents\Master\Thesis\vlm scaling laws repository;$PYTHONPATH"
for ensemble_type in tabpfn
do
  for seed in 42 73 94
  do
    echo "Running: $ensemble_type, seed=$seed, no_intermediate"
    python "$SCRIPT" \
      --model tabpfn \
      --seed "$seed"

    echo "Running: $ensemble_type, seed=$seed, with_intermediate"
    python "$SCRIPT" \
      --model tabpfn \
      --seed "$seed" \
      --include_intermediate_points
  done
done