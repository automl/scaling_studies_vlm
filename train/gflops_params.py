import json
import re
from pathlib import Path

import torch

from train.clip_train.profiler import get_raw_model_flops
from train.utils import create_default_model, precision_to_dtype


def count_params(model):
    return sum(m.numel() for m in model.parameters())


def count_non_emb_text_params(model) -> int:
    m = model.module if hasattr(model, "module") else model

    transformer = sum(p.numel() for p in m.transformer.parameters())
    ln_final = sum(p.numel() for p in m.ln_final.parameters()) if hasattr(m, "ln_final") else 0
    text_proj = m.text_projection.numel() if getattr(m, "text_projection", None) is not None else 0

    return transformer + ln_final + text_proj


def parse_widths(name: str):
    match = re.search(r"text_(\d+)_vision_(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse widths from {name}")
    return int(match.group(1)), int(match.group(2))


if __name__ == "__main__":
    all_possible_configs_dir = Path("/work/dlclarge1/sinanid-VLM-scaling-law/all_model_configs")

    results = {}

    for cfg in all_possible_configs_dir.iterdir():
        if not cfg.is_dir():
            continue

        try:
            text_width, vision_width = parse_widths(cfg.name)
            pair_key = f"text_{text_width}_vision_{vision_width}"

            model = create_default_model(model_str=f"local-dir:{cfg}")

            if torch.cuda.is_available():
                model = model.cuda()

            flops_per_sample = get_raw_model_flops(
                model,
                1,
                dtype=precision_to_dtype("amp"),
            )

            total_params = round(count_params(model), 2)

            model_for_counts = model.module if hasattr(model, "module") else model

            image_params = round(count_params(model_for_counts.visual), 2)
            text_params = round(total_params - image_params, 2)
            text_params_non_emb = round(count_non_emb_text_params(model_for_counts), 2)

            results.setdefault(pair_key, []).append({
                "arch": cfg.name,
                "flops_per_sample": flops_per_sample,
                "params": total_params,
                "vision_params": image_params,
                "text_params": text_params,
                "text_params_non_emb": text_params_non_emb,
            })

        except Exception as e:
            print(f"Failed for {cfg.name}: {e}")

    with open("flops_params.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved results to flops_params.json")
