import copy
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import neps
import numpy as np
import torch
import yaml
from clip_train.params import parse_args
from clip_train.profiler import get_model_flops
from eval.downstream_eval import ClipBenchmarkEvaluator
from eval.downstream_eval import ZeroShotDataset, RetrievalDataset
from torch import optim
from utils import read_last_json_line

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from clip_train.data import get_data
from clip_train.distributed import is_master, init_distributed_device, broadcast_object
from clip_train.logger import setup_logging
from clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from clip_train.train import train_one_epoch, evaluate
from clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from utils import precision_to_dtype

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    seed = seed + rank
    try:
        torch.manual_seed(seed)
    except IndexError as exc:
        logging.warning(
            "torch.manual_seed failed while seeding CUDA generators: %s. "
            "Seeding the CPU generator and continuing without CUDA RNG seeding.",
            exc,
        )
        torch.default_generator.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"lr_{args.lr:.4g}"
        ])

    resume_latest = args.resume == 'latest'
    args.logs = str(args.logs).replace("results", "train/results")
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path) and not resume_latest:
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        # FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        # FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model,
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        if args.train_data:
            with open(params_file, "w") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    logging.info(f"  {name}: {val}")
                    f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
            )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"] if "consecutive_nan_steps" not in checkpoint else checkpoint["model"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.warmup_fraction is not None:
            args.warmup = int(total_steps * args.warmup_fraction)
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None, \
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
            settings=wandb.Settings(symlink=False),
            mode="offline"
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        metrics = evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer, checkpoint=args.resume)

        neps_dir = Path(args.logs).parent

        config_dir = neps_dir / f"configs/config_{args.pipeline_id}"
        report_path = config_dir / "report.yaml"
        train_report_path = config_dir / "report_train.yaml"

        with open(report_path, "r") as f:
            state = yaml.safe_load(f) or {}

        eval_results = {
            "validation_loss": metrics["clip_val_loss"],
            "validation_duration_seconds": metrics["val_time_sec"],
            "test_loss": metrics["clip_test_loss"],
            "test_duration_seconds": metrics["test_time_sec"]
        }

        info_dict = state.pop("extra", {}) or {}
        state["info_dict"] = info_dict

        with open(train_report_path, "w") as f:
            yaml.safe_dump(state, f, sort_keys=False)

        info_dict["eval_results"] = eval_results

        neps.save_pipeline_results(
            user_result=state,
            pipeline_id=str(args.pipeline_id),
            root_directory=neps_dir,
        )

        return

    loss = create_loss(args)

    start_time = time.time()
    early_stop_info = None

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        epoch_start = time.time()
        early_stop_info = train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args,
                                          tb_writer=writer)

        stop_training = early_stop_info is not None
        if stop_training:
            if is_master(args):
                logging.info(f"Early stopping triggered: {early_stop_info}")
            break

        completed_epoch = epoch + 1
        if is_master(args):
            steps_per_epoch = data["train"].dataloader.num_batches
            global_steps_completed = steps_per_epoch * completed_epoch

            in_warmup = global_steps_completed <= args.warmup
            warmup_progress = min(global_steps_completed / args.warmup, 1.0) if args.warmup > 0 else 1.0
            training_progress = global_steps_completed / total_steps

            epoch_steps_info = {
                "epoch": completed_epoch,
                "global_steps_completed": global_steps_completed,
                "total_global_steps": total_steps,
                "in_warmup": in_warmup,
                "warmup_steps": args.warmup,
                "warmup_progress": warmup_progress,
                "training_progress": training_progress
            }

            path = os.path.join(args.checkpoint_path, "epoch_steps_info.jsonl")
            os.makedirs(args.checkpoint_path, exist_ok=True)

            with open(path, "a") as f:
                f.write(json.dumps(epoch_steps_info) + "\n")

        epoch_end = time.time()
        epoch_duration_seconds = epoch_end - epoch_start

        if is_master(args):
            append_jsonl(
                os.path.join(args.checkpoint_path, "train_time.jsonl"),
                {
                    "epoch": completed_epoch,
                    "epoch_training_duration_seconds": epoch_duration_seconds,
                }
            )

        if any(v in data for v in ('val', 'test', 'imagenet-val', 'imagenet-v2')) & args.eval:
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                    args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
                print(f"Saved checkpoint for epoch {completed_epoch} at {args.checkpoint_path}")
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    end_time = time.time()
    training_duration_seconds = end_time - start_time

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name),
            os.path.join(args.remote_sync, args.name),
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

    if is_master(args):
        if args.wandb:
            wandb.finish()

        evaluation_results = None
        validation_loss = None
        test_loss = None
        test_duration_seconds = None

        if args.eval:
            evaluation_results = read_last_json_line(args.checkpoint_path + "/results.jsonl")
            validation_loss = evaluation_results["clip_val_loss"]
            test_loss = evaluation_results.get("clip_test_loss", None)
            test_duration_seconds = evaluation_results.get("test_time_sec", None)

        downstream_evaluation_results = {}

        if args.downstream_eval:
            downstream_evaluation_results = evaluate_benchmarks(args.model, log_base_path, args.seed)

        training_loss = read_last_json_line(args.checkpoint_path + "/train_loss.jsonl")[
            "train_loss"] if early_stop_info is None else 1e6

        samples_per_epoch = data["train"].dataloader.num_samples

        compute_per_epoch = get_model_flops(model, samples_per_epoch, dtype=precision_to_dtype(args.precision))
        compute_per_sample = compute_per_epoch / samples_per_epoch

        if early_stop_info is None:
            total_samples = samples_per_epoch * args.epochs
            failed_epoch = None
            samples_in_failed_epoch = None
        else:
            failed_epoch = int(early_stop_info["epoch"]) # 0 based, logs are 1 based
            samples_in_failed_epoch = int(early_stop_info["samples_processed_current_epoch"])
            total_samples = failed_epoch * samples_per_epoch + samples_in_failed_epoch

        compute = compute_per_sample * total_samples

        compute_info = {
            "total_samples": total_samples,
            "epochs": args.epochs,
            "total_compute": compute,
            "compute_per_epoch": compute_per_epoch,
            "compute_per_sample": compute_per_sample,
        }

        if early_stop_info is not None:
            compute_info.update(
                {
                    "failed_epoch": failed_epoch,
                    "samples_processed_current_epoch": samples_in_failed_epoch,
                    "samples_per_epoch": samples_per_epoch,
                }
            )

        compute_path = os.path.join(args.checkpoint_path, "compute_info.json")

        with open(compute_path, "w") as f:
            json.dump(compute_info, f, indent=2)

        # Read training loss over datapoints if available
        training_loss_over_datapoints = None
        loss_over_data_path = os.path.join(args.checkpoint_path, "train_loss_over_data_seen.jsonl")
        if os.path.exists(loss_over_data_path):
            with open(loss_over_data_path, "r") as f:
                training_loss_over_datapoints = [json.loads(line) for line in f if line.strip()]

        total_mparams = round(count_params(model) / 1e6, 2)
        if args.world_size > 1:
            image_mparams = round(count_params(model.module.visual) / 1e6, 2)
        else:
            image_mparams = round(count_params(model.visual) / 1e6, 2)
        text_mparams = round(total_mparams - image_mparams, 2)
        text_mparams_non_emb = round(count_non_emb_text_params(model) / 1e6, 2)

        out_path = Path(args.checkpoint_path) / "model_params.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w") as f:
            f.write(f"total_mparams: {total_mparams}\n")
            f.write(f"image_mparams: {image_mparams}\n")
            f.write(f"text_mparams: {text_mparams}\n")
            f.write(f"text_mparams_non_emb: {text_mparams_non_emb}\n")

        result = {
            "objective_to_minimize": training_loss,
            "cost": compute,
            "info_dict": {
                "train_loss": training_loss,
                "training_loss_over_datapoints": training_loss_over_datapoints,
                "train_duration_seconds": training_duration_seconds,
                "validation_loss": validation_loss,
                "validation_duration_seconds": evaluation_results["val_time_sec"] if evaluation_results else None,
                "test_loss": test_loss,
                "test_duration_seconds": test_duration_seconds,
                "eval_results": downstream_evaluation_results,
                "total_mparams": total_mparams,
                "image_mparams": image_mparams,
                "text_mparams": text_mparams,
                "n_param": total_mparams,
                "flops": compute,
                "n_data": total_samples,
                "compute_per_epoch": compute_per_epoch,
                "compute_per_sample": compute_per_sample,
            },
        }

        logging.info(f"{str(result)}")

        if args.pipeline_id is None:
            return

        root_dir = Path(args.logs).parent

        if not args.sync_neps:
            # root_dir =Path(str(root_dir).replace("results", "train/results"))
            # Write result to report_train.yaml instead of calling neps.save_pipeline_results
            config_dir = root_dir / f"configs/config_{args.pipeline_id}"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            report_train_path = config_dir / "report_train.yaml"
            
            with open(report_train_path, "w") as f:
                yaml.safe_dump(result, f, sort_keys=False)
            
            logging.info(f"Saved result to {report_train_path}")
            
            # Import and call post_hoc_iterator.main() for downstream evaluation
            from post_hoc_iterator import main as post_hoc_main
            
            logging.info("Running downstream evaluation...")
            post_hoc_main(
                root=str(log_base_path),
                model=f"local-dir:{root_dir}/model_cfg/config_{args.pipeline_id}",
                seed=args.seed,
                pipeline_id=args.pipeline_id,
                cluster=args.cluster,
            )
            
            # Wait for report.yaml to be written
            report_path = root_dir / f"configs/config_{args.pipeline_id}/report_down.yaml"
            report_train_path = root_dir / f"configs/config_{args.pipeline_id}/report_train.yaml"
            max_wait_time = 300  # 5 minutes timeout
            wait_interval = 2  # Check every 2 seconds
            elapsed = 0
            
            while not report_train_path.exists() and elapsed < max_wait_time:
                logging.info(f"Waiting for report_train.yaml... ({elapsed}s)")
                time.sleep(wait_interval)
                elapsed += wait_interval
            
            if not report_path.exists():
                logging.error(f"Timeout waiting for report.yaml at {report_path}")
                raise TimeoutError(f"report.yaml not generated within {max_wait_time} seconds")
            
            logging.info(f"Found report.yaml, reading results...")
            with open(report_path, "r") as f:
                report_data = yaml.safe_load(f) or {}
            
            # Extract data from report. Older helpers used "extra"; the
            # downstream enrichment path writes "info_dict".
            info_dict = (
                report_data.get("info_dict")
                or report_data.get("extra")
                or report_data.get("extra_info")
                or {}
            )
            n_param = info_dict.get("n_param", total_mparams)
            downstream_results = info_dict.get("downstream_results", {})
            
            # Extract zero-shot and retrieval results
            zero_shot_results = downstream_results.get("zero_shot_results", [])
            
            # Build multi-objective tuple: minimize n_param, zero-shot acc5,
            # and MSCOCO text retrieval recall@5.
            cifar10_acc5 = zero_shot_results[0].get("acc5", 0) if zero_shot_results else 0
            imagenet1k_acc5 = zero_shot_results[1].get("acc5", 0) if zero_shot_results else 0
            
            result = {
                "objective_to_minimize": (
                    n_param,
                    -1 * cifar10_acc5,
                    -1 * imagenet1k_acc5,
                ),
                "cost": compute,
                "info_dict": info_dict,
                "extra": info_dict,
            }
            
            logging.info(f"Sync eval result: {result}")
            
            # Write result to report_train.yaml
            report_train_path = root_dir / f"configs/config_{args.pipeline_id}/report_final.yaml"
            with open(report_train_path, "w") as f:
                yaml.safe_dump(result, f, sort_keys=False)
            
            logging.info(f"Saved sync_eval result to {report_train_path}")
            return
        
        sync_dir = root_dir / "sync_results"
        sync_dir.mkdir(parents=True, exist_ok=True)

        pipeline_file = sync_dir / f"{args.pipeline_id}.json"
        with open(pipeline_file, "w") as f:
            json.dump(result, f, indent=2)


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def evaluate_benchmarks(model, root, seed):
    evaluator = ClipBenchmarkEvaluator(model=model, root=root, default_seed=seed)

    zero_shot_results = evaluator.zero_shot_eval([ZeroShotDataset.CIFAR10, ZeroShotDataset.IMAGENET])
    retrieval_results = evaluator.zero_shot_retrieval([RetrievalDataset.MS_COCO])

    return {
        "zero_shot": zero_shot_results,
        "retrieval": retrieval_results
    }


def count_params(model):
    return sum(m.numel() for m in model.parameters())


def count_non_emb_text_params(model) -> int:
    m = model.module if hasattr(model, "module") else model

    transformer = sum(p.numel() for p in m.transformer.parameters())
    ln_final = sum(p.numel() for p in m.ln_final.parameters()) if hasattr(m, "ln_final") else 0
    text_proj = m.text_projection.numel() if getattr(m, "text_projection", None) is not None else 0

    return transformer + ln_final + text_proj


if __name__ == "__main__":
    main(sys.argv[1:])
