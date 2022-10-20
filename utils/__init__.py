# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : __init__.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:22 PM 
'''


from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from munkres import Munkres
from sklearn import metrics
import warnings
import matplotlib.pyplot as plt
import tqdm
import torch.distributed as dist

from utils.ops import convert_to_cuda, is_root_worker
from .knn_monitor import knn_monitor


@torch.no_grad()
def extract_features(extractor, loader):
    extractor.eval()

    local_features = []
    local_labels = []
    for inputs in tqdm.tqdm(loader, disable=not is_root_worker()):
        images, labels = convert_to_cuda(inputs)
        local_labels.append(labels)
        local_features.append(extractor(images))
    local_features = torch.cat(local_features, dim=0)
    local_labels = torch.cat(local_labels, dim=0)

    indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

    features = torch.zeros(len(loader.dataset), local_features.size(1)).cuda()
    all_labels = torch.zeros(len(loader.dataset)).cuda()
    counts = torch.zeros(len(loader.dataset)).cuda()
    features.index_add_(0, indices, local_features)
    all_labels.index_add_(0, indices, local_labels.float())
    counts[indices] = 1.

    if dist.is_initialized():
        dist.all_reduce(features, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_labels, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    # account for the few samples that are computed twice
    labels = (all_labels / counts).long()
    features /= counts[:, None]

    return features, labels


# @torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def shuffling_forward(inputs, encoder):
    # shuffle for making use of BN
    inputs, idx_unshuffle = _batch_shuffle_ddp(inputs)
    inputs = encoder(inputs)  # keys: NxC
    # undo shuffle
    inputs = _batch_unshuffle_ddp(inputs, idx_unshuffle)
    return inputs


@torch.no_grad()
def _batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def _batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


@torch.no_grad()
def _momentum_update(q_params, k_params, m):
    """
    Momentum update
    """
    if not isinstance(q_params, (list, tuple)):
        q_params, k_params = [q_params, ], [k_params, ]
    for param_q, param_k in zip(q_params, k_params):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform1, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform1 if transform2 is None else transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

    def __str__(self):
        return f'transform1 {str(self.transform1)} transform2 {str(self.transform2)}'