#!/bin/bash

python -m train.scaling_law_plots.pareto_vs_hyperparameters
python -m train.scaling_law_plots.models_comparison
python -m train.comparison_across_budgets
python -m train.scaling_law_plots.plotting