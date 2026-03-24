from pathlib import Path

import open_clip
import torch


def create_model(model_arch, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model_path = str(model_path)

    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_arch,
        pretrained=None,
    )

    checkpoint = torch.load(model_path, map_location="cpu")

    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        if next(iter(sd)).startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}
    else:
        sd = checkpoint

    model.load_state_dict(sd, strict=True)

    model.eval()
    model = model.to(device)

    return model, transform, device


if __name__ == "__main__":
    individual_training_logs_dir = Path("/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/individual_training_logs")
    model_cfg_dir = Path("/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/results/optimizer=scan_space_final_biased_alpha_0_5/seed=94/model_cfg")
    all_possible_configs_dir = Path("/work/dlclarge1/sinanid-VLM-scaling-law/all_model_configs")

    for train_dir in individual_training_logs_dir.iterdir():
        if not train_dir.is_dir():
            continue

        ckpt_dir = train_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        pt_files = list(ckpt_dir.glob("*.pt"))
        if not pt_files:
            print(f"No .pt in {train_dir}")
            continue

        ckpt = pt_files[0]

        config_id = int(train_dir.name.replace("train_config", ""))
        if 600 > config_id > 701:
            continue
        cfg_dir = model_cfg_dir / f"config_{config_id}"

        try:
            create_model(f"local-dir:{cfg_dir}", ckpt)
            print(f"config works for {train_dir.name}")
            continue
        except Exception:
            print(f"Failed existing config for {train_dir.name}: {e}")
            pass

        for cfg in all_possible_configs_dir.iterdir():
            if not cfg.is_dir():
                continue

            try:
                create_model(f"local-dir:{cfg}", ckpt)

                cfg_dir.mkdir(parents=True, exist_ok=True)
                (cfg_dir / "open_clip_config.json").write_text(
                    (cfg / "open_clip_config.json").read_text()
                )
                print(f"matched {cfg.name} with {train_dir.name}")
                break

            except Exception as e:
                print(f"None of the configs suitable: {e}")
