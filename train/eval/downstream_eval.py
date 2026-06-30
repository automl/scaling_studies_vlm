import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, List, Union, Mapping

from eval.openclip_benchmark_cli import main_eval
from eval.openclip_benchmark_parser import get_parser
from utils import find_latest_checkpoint, _build_output_name


class EvalTask(str, Enum):
    ZEROSHOT_CLASSIFICATION = "zeroshot_classification"
    RETRIEVAL = "zeroshot_retrieval"
    LINEAR_PROBE = "linear_probe"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Optional[Union[Path, Mapping[str, Path]]] = None


class EvalDataset(Enum):
    pass


class ZeroShotDataset(EvalDataset):
    CIFAR10 = DatasetSpec("cifar10")
    IMAGENET = DatasetSpec(
        "imagenet1k",
        root={
            "meta": Path("/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/eval_data"),
            "juwels": Path("/p/data1/datasets/Imagenet2012_1k_tar"),
            "tu_dresden": Path("/data/cat/ws/naal364i-primo/runs/train/eval_data"),
        },
    )


class RetrievalDataset(EvalDataset):
    MS_COCO = DatasetSpec(
        "mscoco_captions",
        root={
            "meta": Path("/work/dlclarge1/sinanid-VLM-scaling-law/scaling_studies_vlm/train/eval_data"),
            "juwels": Path("/p/data1/datasets/scan/openclip/train/eval_data"),
        },
    )


class ClipBenchmarkEvaluator:
    def __init__(
            self,
            model: str = "ViT-S-32",
            root: str = "",
            default_batch_size: int = 128,
            default_seed: int = 0,
            use_amp: bool = False,
            cluster: str = "meta"
    ) -> None:
        """
        Wrapper around clip_benchmark eval.
        """

        self.model = model
        self.root = Path(root)
        self.pre_trained_path = self.root / "checkpoints"
        self.default_batch_size = default_batch_size
        self.default_seed = default_seed
        self.use_amp = use_amp

        self.eval_dir = self.root / "eval"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = self.eval_dir
        self.cluster = cluster

        self.parser = get_parser()

    def _resolve_dataset_root(self, dataset: EvalDataset) -> str:
        spec = dataset.value

        if spec.root is None:
            return "eval_data"

        if isinstance(spec.root, Path):
            return str(spec.root)

        if self.cluster in spec.root:
            return str(spec.root[self.cluster])

        raise KeyError(
            f"Cluster '{self.cluster}' has no root path configured for dataset '{spec.name}'. "
            f"Available clusters: {list(spec.root.keys())}"
        )

    def _build_argv(
            self,
            task: str,
            dataset: EvalDataset,
            pretrained: Optional[Path],
            output: Optional[Path],
            batch_size: Optional[int],
            seed: Optional[int],
            extra_args: Optional[Iterable[str]] = None,
    ) -> List[str]:

        bs = batch_size if batch_size is not None else self.default_batch_size
        sd = seed if seed is not None else self.default_seed
        pretrained = pretrained if pretrained is not None else self.pre_trained_path

        checkpoint = find_latest_checkpoint(pretrained)
        pretrained = pretrained / checkpoint

        epoch = int(checkpoint.split("_")[1].split(".")[0])
        filename = _build_output_name(dataset.value.name, epoch)

        out = (output / filename) if output is not None else (self.output_dir / filename)
        out.parent.mkdir(parents=True, exist_ok=True)

        dataset_root = self._resolve_dataset_root(dataset)

        argv = [
            "eval",
            f"--dataset={dataset.value.name}",
            "--task", task,
            "--model", self.model,
            "--pretrained", str(pretrained),
            "--batch_size", str(bs),
            "--output", str(out),
            "--seed", str(sd),
            "--dataset_root", dataset_root,
        ]

        if not self.use_amp:
            argv.append("--no_amp")
        if extra_args:
            argv.extend(list(extra_args))
        return argv

    def evaluate(
            self,
            task: EvalTask,
            dataset: EvalDataset,
            pretrained: Optional[Path] = None,
            output: Optional[Path] = None,
            batch_size: Optional[int] = None,
            seed: Optional[int] = None,
            extra_args: Optional[Iterable[str]] = None,
    ) -> Path:

        argv = self._build_argv(
            task=task.value,
            dataset=dataset,
            pretrained=pretrained,
            output=output,
            batch_size=batch_size,
            seed=seed,
            extra_args=extra_args,
        )
        args = self.parser.parse_args(argv)

        main_eval(args)

        return Path(args.output)

    def zero_shot_eval(self, datasets: ZeroShotDataset | list[ZeroShotDataset]):
        out = self.output_dir / "zero_shot"

        if isinstance(datasets, ZeroShotDataset):
            datasets = [datasets]

        results = []

        for ds in datasets:
            start = time.time()

            output_file = self.evaluate(
                task=EvalTask.ZEROSHOT_CLASSIFICATION,
                dataset=ds,
                output=out
            )

            duration = time.time() - start

            data = json.loads(Path(output_file).read_text())
            metrics = data["metrics"]

            results.append({
                "dataset": data["dataset"],
                "task": data["task"],
                "acc1": metrics.get("acc1"),
                "acc5": metrics.get("acc5"),
                "mean_per_class_recall": metrics.get("mean_per_class_recall"),
                "duration_sec": duration,
            })

        return results

    def linear_probe(
            self,
            datasets: ZeroShotDataset | list[ZeroShotDataset],
            lr: float = 0.1,
            epochs: int = 1,
            use_val_split: bool = False,
            fewshot_k: int | None = None,
    ):
        out = self.output_dir / "linear_probe"

        if isinstance(datasets, ZeroShotDataset):
            datasets = [datasets]

        results = []

        for ds in datasets:

            # TODO add val split for performing hyperparam sweep over weight decay
            extra_args = [
                "--train_split", "train",
                "--test_split", "test",
                "--fewshot_lr", str(lr),
                "--fewshot_epochs", str(epochs),
            ]

            if fewshot_k is not None:
                extra_args += ["--fewshot_k", str(fewshot_k)]

            if use_val_split:
                extra_args += ["--val_proportion", "0.1"]

            start = time.time()
            output_file = self.evaluate(
                task=EvalTask.LINEAR_PROBE,
                dataset=ds,
                output=out,
                extra_args=extra_args,
            )
            duration = time.time() - start

            data = json.loads(Path(output_file).read_text())
            metrics = data["metrics"]

            results.append({
                "dataset": data["dataset"],
                "task": data["task"],
                "lp_acc1": metrics.get("lp_acc1"),
                "lp_acc5": metrics.get("lp_acc5"),
                "mean_per_class_recall": metrics.get("lp_mean_per_class_recall"),
                "epochs": metrics.get("epochs"),
                "fewshot_k": metrics.get("fewshot_k"),
                "weight_decay": metrics.get("weight_decay"),
                "duration_sec": duration,
            })

        return results

    def zero_shot_retrieval(self, datasets: RetrievalDataset | list[RetrievalDataset]):
        out = self.output_dir / "retrieval"

        if isinstance(datasets, RetrievalDataset):
            datasets = [datasets]

        results = []

        for ds in datasets:
            start = time.time()
            try:
                output_file = self.evaluate(
                    task=EvalTask.RETRIEVAL,
                    dataset=ds,
                    output=out,
                )
            except (FileNotFoundError, RuntimeError, KeyError) as e:
                print(f"Skipping retrieval eval for {ds.value.name} on cluster '{self.cluster}': {e}")
                continue

            duration = time.time() - start

            data = json.loads(Path(output_file).read_text())
            metrics = data["metrics"]

            results.append({
                "dataset": data["dataset"],
                "task": data["task"],
                "image_retrieval": metrics.get("image_retrieval_recall@5"),
                "text_retrieval": metrics.get("text_retrieval_recall@5"),
                "duration_sec": duration,
            })

        return results
