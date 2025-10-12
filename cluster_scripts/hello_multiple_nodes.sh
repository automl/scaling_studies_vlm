#!/bin/bash
#SBATCH --job-name test_multi_node_setup
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

nvidia-smi

source ~/miniconda3/bin/activate
conda activate vlm_scaling_law

# Master setup (rank 0 node)
scontrol show hostnames "$SLURM_JOB_NODELIST"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

cat > ddp_ping.py << 'PY'
import os, socket, torch
import torch.distributed as dist

def main():
    rank       = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")

    host = socket.gethostname()
    x = torch.ones(1, device=device) * rank
    print(f"[rank {rank}/{world_size} | {host} | local_rank {local_rank}] start x={x.item()}")

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = (world_size - 1) * world_size / 2
    ok = "OK" if x.item() == expected else f"WRONG(expected {expected})"
    print(f"[rank {rank}] after all_reduce x={x.item()}  => {ok}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
PY

srun --mpi=none --accel-bind=gn python ddp_ping.py