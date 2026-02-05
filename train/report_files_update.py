import json
import os
import re
from pathlib import Path

import yaml
from open_clip import create_model

from clip_train.profiler import get_model_flops
from utils import read_last_json_line, precision_to_dtype


def create_yaml_file(checkpoint_path, n_data, total_mparams, image_mparams, text_mparams, compute, save_to):
    training_loss = read_last_json_line(checkpoint_path / "train_loss.jsonl")["train_loss"]

    training_loss_over_datapoints = None
    loss_over_data_path = os.path.join(checkpoint_path, "train_loss_over_data_seen.jsonl")
    if os.path.exists(loss_over_data_path):
        with open(loss_over_data_path, "r") as f:
            training_loss_over_datapoints = [json.loads(line) for line in f if line.strip()]

    training_duration_seconds = read_last_json_line(checkpoint_path / "train_time.jsonl")[
        "epoch_training_duration_seconds"]

    # currently these don't contribute to the plots
    evaluation_results = read_last_json_line(checkpoint_path / "results.jsonl")
    validation_loss = evaluation_results["clip_val_loss"]
    test_loss = evaluation_results.get("clip_test_loss", None)
    test_duration_seconds = evaluation_results.get("test_time_sec", None)

    result = {
        "objective_to_minimize": validation_loss,
        "cost": compute,
        "extra": {
            "train_loss": training_loss,
            "training_loss_over_datapoints": training_loss_over_datapoints,
            "train_duration_seconds": training_duration_seconds,
            "total_mparams": total_mparams,
            "image_mparams": image_mparams,
            "text_mparams": text_mparams,
            "n_param": total_mparams,
            "flops": compute,
            "n_data": n_data,
        },
        "learning_curve": training_loss,
        "reported_as": "success",
        "tb": "null",
        "err": "null",
        "evaluation_duration": compute
    }

    output_path = save_to / "report.yaml"
    print(f"saving to {str(output_path)}")
    with open(output_path, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    return training_duration_seconds


def fetch_ndata(script_path):
    text = script_path.read_text()

    match = re.search(r"--train-num-samples\s+(\d+)", text)
    if match:
        train_num_samples = int(match.group(1))
    else:
        raise ValueError("--train-num-samples not found")

    return train_num_samples


def count_params(model):
    return sum(m.numel() for m in model.parameters())


def modify_metadata_json(json_file_path, compute, training_duration):
    with open(json_file_path, "r") as f:
        metadata = json.load(f)

    if metadata.get("time_started") is None:
        raise ValueError("time_started is missing, cannot compute duration")

    time_started = metadata["time_started"]

    metadata["state"] = "success"
    metadata["evaluation_duration"] = compute
    metadata["time_end"] = time_started + training_duration

    with open(json_file_path, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    neps_path = Path(
        "/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/"
        "optimizer=scale_space_only_search_biased_scale_space/seed=42"
    )

    config_dirs = sorted(
        [p for p in (neps_path / "configs").glob("config_*") if p.is_dir()]
    )

    print(f"[main] found {len(config_dirs)} config directories")

    for config_path in config_dirs:
        print("\n" + "=" * 80)
        config_dir = config_path.name
        config_id = int(config_dir.replace("config_", ""))

        print(f"[main] config_id = {config_id}")

        train_script_path = neps_path / "train_scripts" / f"train_config{config_id}.sh"
        config_checkpoint_path = (neps_path / "individual_training_logs" / f"train_config{config_id}" / "checkpoints")
        model_cfg_path = neps_path / "model_cfg" / f"config_{config_id}"
        metadata_json = neps_path / "configs" / f"config_{config_id}" / "metadata.json"

        print(f"[main] train_script_path = {train_script_path}")
        print(f"[main] checkpoint_path   = {config_checkpoint_path}")
        print(f"[main] metadata_json     = {metadata_json}")

        if not train_script_path.exists():
            print("[SKIP] train script missing")
            continue
        if not config_checkpoint_path.exists():
            print("[SKIP] checkpoints missing")
            continue
        if not metadata_json.exists():
            print("[SKIP] metadata.json missing")
            continue

        n_data = fetch_ndata(train_script_path)
        print(f"[main] fetched n_data = {n_data}")

        model = create_model(
            f"local-dir:results/optimizer=scale_space_only_search_biased_scale_space/seed=42/model_cfg/config_{config_id}",
            precision="amp")
        print("created model")
        compute = get_model_flops(model, n_data, dtype=precision_to_dtype("amp"))

        total_mparams = round(count_params(model) / 1e6, 2)
        image_mparams = round(count_params(model.visual) / 1e6, 2)
        text_mparams = round(total_mparams - image_mparams, 2)

        training_duration_seconds = create_yaml_file(config_checkpoint_path, n_data, total_mparams, image_mparams,
                                                     text_mparams, compute,
                                                     neps_path / "configs" / config_dir)
        modify_metadata_json(metadata_json, compute, training_duration_seconds)
