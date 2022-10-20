# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : ops.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 8:43 PM 
'''
import math

from torch import nn
import torch
from typing import Union
import torch.distributed as dist
from torch._six import string_classes
import torch.nn.functional as F
import collections.abc as container_abcs
from PIL import Image


@torch.no_grad()
def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res


def concat_all_gather(tensor):
    dtype = tensor.dtype
    tensor = tensor.float()
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    output = output.to(dtype)
    return output


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data = cls.__getitem__(self, index)
        return [data, index]

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [convert_to_cuda(d) for d in data]
    else:
        return data


def is_root_worker():
    verbose = True
    if dist.is_initialized():
        if dist.get_rank() != 0:
            verbose = False
    return verbose


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def convert_to_ddp(modules: Union[list, nn.Module], **kwargs):
    if isinstance(modules, list):
        modules = [x.cuda() for x in modules]
    else:
        modules = modules.cuda()
    if dist.is_initialized():
        device = torch.cuda.current_device()
        if isinstance(modules, list):
            modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                                 device_ids=[device, ],
                                                                 output_device=device,
                                                                 **kwargs) for
                       x in modules]
        else:
            modules = torch.nn.parallel.DistributedDataParallel(modules,
                                                                device_ids=[device, ],
                                                                output_device=device,
                                                                **kwargs)

    else:
        modules = torch.nn.DataParallel(modules)

    return modules
