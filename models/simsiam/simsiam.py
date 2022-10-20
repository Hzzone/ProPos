# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : simsiam.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:27 PM 
'''


import torch
import argparse
import copy

from .simsiam_wrapper import SimSiamWrapper
from utils.ops import convert_to_ddp
from network import backbone_dict
from models import model_dict
from models.basic_template import TrainTask

@model_dict.register('simsiam')
class SimSiam(TrainTask):
    def set_model(self):
        opt = self.opt
        self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        simsiam = SimSiamWrapper(encoder, in_dim=dim_in, feat_dim=opt.feat_dim,
                                 projection_num_layers=opt.projection_num_layers)
        self.feature_extractor_copy = copy.deepcopy(simsiam.encoder).cuda()
        optimizer = torch.optim.SGD(params=self.collect_params(simsiam, exclude_bias_and_bn=False),
                                    lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

        simsiam = convert_to_ddp(simsiam)
        self.logger.modules = [encoder, simsiam, optimizer]
        self.simsiam = simsiam
        self.optimizer = optimizer
        self.feature_extractor = simsiam.module.encoder

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--projection_num_layers', help='projection_num_layers, 2 for cifar10 and cifar100',
                            type=int, default=3)
        parser.add_argument('--fix_predictor_lr', help='fix the lr of predictor', action='store_true')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        images, labels = inputs
        self.simsiam.train()

        im_q, im_k = images

        # compute loss
        loss, q = self.simsiam(im_q, im_k)

        self.optimizer.zero_grad()
        # SGD
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            import math
            q_std = torch.std(q.detach(), dim=0).mean() * math.sqrt(q.size(1))

        self.logger.msg([loss, q_std], n_iter)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        lr = self.cosine_annealing_LR(n_iter)
        for i in range(len(self.optimizer.param_groups)):
            name = self.optimizer.param_groups[i]['name']
            if 'predictor' in name and opt.fix_predictor_lr:
                self.optimizer.param_groups[i]['lr'] = opt.learning_rate
            else:
                self.optimizer.param_groups[i]['lr'] = lr
        self.logger.msg([lr, ], n_iter)
