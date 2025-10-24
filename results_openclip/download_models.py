import argparse
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

trained_models_info= pd.read_csv("trained_models_info.csv")

base_cache_dir = Path("models--laion--scaling-laws-openclip")
snapshot_dir = base_cache_dir / "snapshots"

def download_model(model, samples_seen, dataset):
    res = trained_models_info[
        (trained_models_info.arch==model) & (trained_models_info.samples_seen_pretty==samples_seen) & (trained_models_info.data==dataset)
    ]
    if len(res) == 1:
        filename = res.name.tolist()[0]
        hf_hub_download("laion/scaling-laws-openclip", filename, cache_dir=".", force_filename=filename)
        print(f"'{filename}' downloaded.")
        return filename
    else:
        print("The model is not available in the repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to download pre-trained models")
    parser.add_argument("--samples_seen", type=str, nargs="+", default=["3B", "13B", "34B"])
    parser.add_argument("--dataset", type=str, nargs="+",default=["80M", "400M", "2B"])
    parser.add_argument("--model", type=str, nargs="+", default=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-g-14"])
    args = parser.parse_args()
    for model in args.model:
        for samples_seen in args.samples_seen:
            for dataset in args.dataset:
                download_model(model, samples_seen, dataset)

    dest_dir = Path("openclip_checkpoints")
    dest_dir.mkdir(exist_ok=True)

    pt_files = list(snapshot_dir.rglob("*.pt"))

    for pt_file in pt_files:
        target_path = dest_dir / pt_file.name
        print(f"Moving {pt_file} → {target_path}")
        shutil.move(str(pt_file), str(target_path))

    print("Cleaning up the old cache directory...")
    shutil.rmtree(base_cache_dir)
    shutil.rmtree(".locks")
    print("Done.")
