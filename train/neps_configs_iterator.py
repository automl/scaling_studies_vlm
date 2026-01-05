import os
import sys
from pathlib import Path

import neps

from train import params
from train.config import TrainConfig
from train.slurm import SlurmManager, build_srun_cmd


def submit_job_per_config(trial_id, row, root_dir, args):
    hps = {
        col.replace("config.", ""): val
        for col, val in row.items()
        if col.startswith("config.")
    }

    logs_dir = os.path.join(root_dir, "individual_training_logs")
    scripts_dir = os.path.join(root_dir, "slurm_scripts")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    config = TrainConfig(
        model=args.model,
        train_num_samples=args.train_num_samples,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=512,  # TODO to be set based on bs_ranges.json
        pipeline_id=trial_id,
        sync_neps=args.sync_neps,
        logs=logs_dir,
        **hps,
    )

    cmd = build_srun_cmd(config)
    job_name = f"{config.model}_ns{config.train_num_samples}_s{config.seed}"

    slurm = SlurmManager(job_name=job_name, partition=args.partition, nodes=args.nodes,
                         gpus_per_node=args.gpus_per_node,
                         root_dir=root_dir)
    script_name = job_name + ".sh"
    script_path = Path(scripts_dir + "/" + script_name)
    slurm.build_script(cmd, save_to=script_path)
    # slurm.submit_without_build(script_path)


if __name__ == "__main__":
    args = params.parse_args(sys.argv[1:])

    root_dir = f"./train/results/model={args.model}/seed={args.seed}"

    full_df, result = neps.status(root_dir)
    for trial_id, row in full_df.iterrows():
        try:
            # if row.state != "success":
                submit_job_per_config(trial_id, row, root_dir, args)
                # TODO post hoc evaluation

        except Exception as e:
            print(e)
