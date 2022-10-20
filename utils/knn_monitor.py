# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : knn_monitor.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:23 PM 
'''

import torch
import tqdm
import torch.nn.functional as F
import torch.distributed as dist


# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# test using a knn monitor
@torch.no_grad()
def knn_monitor(memory_features,
                memory_labels,
                test_features,
                test_labels,
                knn_k,
                knn_t):
    classes = len(torch.unique(memory_labels))
    # generate feature bank
    # [D, N]
    # feature_bank = memory_features.t().contiguous()
    # [N]
    pred_labels = knn_predict(test_features, memory_features, memory_labels, classes, knn_k, knn_t)

    top1 = (pred_labels[:, 0] == test_labels).float().mean()

    return top1


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict_internal(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = feature.mm(feature_bank.T)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    split_size = 512
    pred_labels = []
    for f in feature.split(split_size, dim=0):
        pred_labels.append(knn_predict_internal(f, feature_bank, feature_labels, classes, knn_k, knn_t))
    return torch.cat(pred_labels, dim=0)
