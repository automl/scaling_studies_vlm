# OpenCLIP Checkpoints

These are the pretrained checkpoints considered for the reproducibility experiments.  
They should be downloaded locally before running the experiments.

## Checkpoints

- Model-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.pt
- Model-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.pt

## Instructions

Run [`open_clip.ipynb`](../open_clip.ipynb) notebook, which downloads the mentioned models, and runs zero-shot classification tasks on 5 datasets(MNIST, CIFAR10, CIFAR100, ImageNet1k, ImageNetv2), zero-shot retrieval(image/text) for MS-COCO dataset, and linear probing for MNIST, with full and few-shot setting, and hyperparameter sweep over the weight decay.