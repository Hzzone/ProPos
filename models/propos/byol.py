# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : byol.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:26 PM 
'''

import argparse
import copy

import torch
import numpy as np
from utils.ops import convert_to_ddp
from .byol_wrapper import BYOLWrapper
from models.basic_template import TrainTask
from network import backbone_dict
from models import model_dict


@model_dict.register('propos')
class BYOL(TrainTask):
    __BYOLWrapper__ = BYOLWrapper

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        byol = self.__BYOLWrapper__(encoder, in_dim=dim_in, num_cluster=self.num_cluster, temperature=opt.temperature,
                                    hidden_size=opt.hidden_size, fea_dim=opt.feat_dim, byol_momentum=opt.momentum_base,
                                    symmetric=opt.symmetric, shuffling_bn=opt.shuffling_bn, latent_std=opt.latent_std,
                                    queue_size=opt.queue_size)
        if opt.syncbn:
            if opt.shuffling_bn:
                byol.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.encoder_q)
                byol.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.projector_q)
                byol.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol.predictor)
            else:
                byol = torch.nn.SyncBatchNorm.convert_sync_batchnorm(byol)
        if opt.lars:
            from utils.optimizers import LARS
            optim = LARS
        else:
            optim = torch.optim.SGD
        optimizer = optim(params=self.collect_params(byol, exclude_bias_and_bn=opt.exclude_bias_and_bn),
                          lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

        self.logger.modules = [byol, optimizer]
        # Initialization
        self.feature_extractor_copy = copy.deepcopy(byol.encoder).cuda()
        byol = byol.cuda()
        self.feature_extractor = byol.encoder
        byol = convert_to_ddp(byol)
        self.byol = byol
        self.optimizer = optimizer

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        # SSL
        parser.add_argument('--symmetric', help='Symmetric contrastive loss', dest='symmetric', action='store_true')
        parser.add_argument('--hidden_size', help='hidden_size', type=int, default=4096)
        parser.add_argument('--fix_predictor_lr', help='fix the lr of predictor', action='store_true')
        parser.add_argument('--lambda_predictor_lr', help='lambda the lr of predictor', type=float, default=10.)
        parser.add_argument('--shuffling_bn', help='shuffling_bn', action='store_true')

        parser.add_argument('--momentum_base', help='ema momentum min', type=float, default=0.996)
        parser.add_argument('--momentum_max', help='ema momentum max', type=float, default=1.0)
        parser.add_argument('--momentum_increase', help='momentum_increase', action='store_true')

        parser.add_argument('--exclude_bias_and_bn', help='exclude_bias_and_bn', action='store_true')
        parser.add_argument('--lars', help='lars', action='store_true')
        parser.add_argument('--syncbn', help='syncbn', action='store_true')
        parser.add_argument('--byol_transform', help='byol_transform', action='store_true')

        # LOSS
        parser.add_argument('--cluster_loss_weight', type=float, default=1.0, help='weight for cluster loss')
        parser.add_argument('--latent_std', type=float, help='latent_std', default=0.0)
        parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
        parser.add_argument('--queue_size', type=int, help='queue_size', default=0)
        parser.add_argument('--v2', help='v2', action='store_true')

        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        images, labels = inputs
        self.byol.train()

        im_q, im_k = images

        update_params = (n_iter % opt.acc_grd_step == 0)

        # psedo_labels = self.psedo_labels[indices]
        self.byol.module.psedo_labels = self.psedo_labels

        is_warmup = not self.cur_epoch > opt.warmup_epochs
        self.byol.module.latent_std = opt.latent_std * float(not is_warmup)
        # compute loss
        with torch.autocast('cuda', enabled=opt.amp):
            contrastive_loss, cluster_loss_batch, q = self.byol(
                im_q, im_k, indices, update_params, opt.v2)

        loss = contrastive_loss + cluster_loss_batch * opt.cluster_loss_weight * float(not is_warmup)

        loss = loss / opt.acc_grd_step
        self.scaler(loss, optimizer=self.optimizer, update_grad=update_params)

        with torch.no_grad():
            q_std = torch.std(q.detach(), dim=0).mean()

        self.logger.msg([contrastive_loss, cluster_loss_batch, q_std, ], n_iter)

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        lr = self.cosine_annealing_LR(n_iter)
        if opt.fix_predictor_lr:
            predictor_lr = opt.learning_rate
        else:
            predictor_lr = lr * opt.lambda_predictor_lr
        flag = False
        for param_group in self.optimizer.param_groups:
            if 'predictor' in param_group['name']:
                flag = True
                param_group['lr'] = predictor_lr
            else:
                param_group['lr'] = lr
        assert flag

        ema_momentum = opt.momentum_base
        if opt.momentum_increase:
            ema_momentum = opt.momentum_max - (opt.momentum_max - ema_momentum) * (
                    np.cos(np.pi * n_iter / (opt.epochs * self.iter_per_epoch)) + 1) / 2
        self.byol.module.m = ema_momentum

        self.logger.msg([lr, predictor_lr, ema_momentum], n_iter)

    def train_transform(self, normalize):
        opt = self.opt
        if not opt.byol_transform:
            return super().train_transform(normalize)
        from torchvision import transforms
        from utils import TwoCropTransform

        '''
        byol transform
        https://github.com/yaox12/BYOL-PyTorch/blob/edefc01aa72716c5c59219883af1ff0ae1127053/data/byol_transform.py
        :param normalize:
        :return:
        '''
        base_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        train_transform1 = base_transform + [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
            transforms.RandomSolarize(128, p=0.0)
        ]
        train_transform2 = base_transform + [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(128, p=0.2)
        ]
        train_transform1 += [transforms.ToTensor(), normalize]
        train_transform2 += [transforms.ToTensor(), normalize]

        train_transform1 = transforms.Compose(train_transform1)
        train_transform2 = transforms.Compose(train_transform2)
        train_transform = TwoCropTransform(train_transform1, train_transform2)
        return train_transform
