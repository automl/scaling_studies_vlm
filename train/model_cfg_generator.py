import json
from pathlib import Path

if __name__ == "__main__":
    widths = [32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024]
    large_widths = [768, 1024]

    output_dir = Path("model_cfg_new")
    output_dir.mkdir(parents=True, exist_ok=True)

    combinations = {
        (text_width, vision_width)
        for text_width in widths
        for vision_width in large_widths
    } | {
        (text_width, vision_width)
        for text_width in large_widths
        for vision_width in widths
    }

    for text_width, vision_width in sorted(combinations):
        folder = output_dir / f"text_{text_width}_vision_{vision_width}"
        folder.mkdir(parents=True, exist_ok=True)

        text_head_width = 32 if text_width == 32 else 64

        config = {
            "model_cfg": {
                "embed_dim": 384,
                "vision_cfg": {
                    "image_size": 224,
                    "layers": 12,
                    "width": vision_width,
                    "head_width": 32 if vision_width == 32 else 64,
                    "patch_size": 32,
                },
                "text_cfg": {
                    "context_length": 77,
                    "vocab_size": 49408,
                    "width": text_width,
                    "heads": text_width // text_head_width,
                    "layers": 12,
                },
            }
        }

        with open(folder / "open_clip_config.json", "w") as f:
            json.dump(config, f, indent=2)
