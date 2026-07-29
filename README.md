# Vision Language Models scaling laws

This project defines the pipeline to train VLM models based on opensource implementation of CLIP style models.

## Requirements

We recommend using a conda environment.

```bash
conda create -n vlm_training python=3.11
conda activate vlm_training
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirments.txt
```

## Training Pipeline

This pipeline consists of three main stages:

1. Configuration generation
2. Individual train/evaluation scripts generation
3. Aggregate job submission in array jobs for training and evaluation

### 1. Configuration Generation

Generate configurations using NePS:

```
python configs_generator.py \
  --hpo-optimizer scan_space_final_biased_alpha_0_5 \
  --seed 94 \
  --workspace_dir "/path/to/workspace"
```

This creates sampled configurations under:

```
/path/to/workspace/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94
```

### 2. Individual training jobs

Iterate over the sampled configurations and prepare training commands:

```
python configs_iterator.py \
  --root_dir "results/optimizer=scan_space_final_biased_alpha_0_5/seed=94" \ # keep relative
  --seed 94 \
  --workspace_dir "/path/to/workspace" \
  --cluster meta \
  --partition alldlc_gpu-rtx3080 \
  --wandb_project_name vlm_training \
  --command_dir "/path/to/command/dir" \
  --bias_scale_sampling \
  --alpha 0.5
```

### 3. Array jobs

Generate cluster ready job scripts:

```
python array_jobs_generator.py \
  --seed 94 \
  --root_dir "results/optimizer=scan_space_final_biased_alpha_0_5/seed=94" \
  --wandb_project_name vlm_training \
  --cluster meta \
  --command_dir "/path/to/command/dir" \
  --workspace_dir "/path/to/workspace" \
  --partition alldlc_gpu-rtx3080
```

### Generated Scripts

All scripts are created under:

```
workspace_dir/root_dir
```

### Training Scripts (train_scripts/)

- train_configX.sh for single configuration training
- train_p{partition}_bs{batch_size}_g{nr_gpus}_n{nr_nodes}.sh for grouped jobs

### Upstream Evaluation Scripts (evaluation_scripts/)

- eval_configX.sh for single configuration evaluation
- array_job.sh for batched evaluation

### 4. Job Execution Order

Submit jobs to the cluster in the following order:

1. Training
2. Upstream evaluation

### 5. Aggregate Results

After all jobs have finished, aggregate the results:

```
python neps_status.py \
  --root_dir "results/optimizer=scan_space_final_biased_alpha_0_5/seed=94" \
  --start_from 1 \
  --end_at 1001
```

This generates a CSV file containing the aggregated results.

## Notes

- Training must complete before evaluation starts
- Designed for cluster execution (e.g. Slurm based systems). Check [`slurm.py`](train/slurm.py) for more details.
- Downstream evaluation is performed using: https://github.com/mlfoundations/datacomp. See their code on how to evaluate
  each model.

## Download Pre-training dataset

A cluster script used to download the data: [`download_train_data.sh`](cluster_scripts/download_train_data.sh)

## Additional

Full information on the trained configurations is found [`here`](train/optimizer=scale_space_biased_0_5/seed=94)

## Figures reproducibility

Run the following script from the main source-tree to recreate the plottings:

```bash
bash train/figures.sh
```

## Acknowledgments

This project builds upon existing open-source software. We gratefully acknowledge the contributions of the original
authors.

Our implementation partially reuses and adapts code from:

- OpenClip (https://github.com/mlfoundations/open_clip)
- Clip benchmark (https://github.com/LAION-AI/CLIP_benchmark)
- NePs(https://github.com/automl/neps)

All third-party components remain under their respective licenses.  
Full license texts are provided in `THIRD_PARTY_LICENSES.md`.
