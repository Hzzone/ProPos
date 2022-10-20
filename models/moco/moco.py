import torch
import argparse

from models.basic_template import TrainTask
from network import backbone_dict
from .moco_wrapper import MoCoWrapper
from utils.ops import convert_to_ddp
from models import model_dict


@model_dict.register('moco')
class MoCo(TrainTask):
    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder_q = encoder_type()
        encoder_k = encoder_type()
        moco = MoCoWrapper(encoder_q, encoder_k, in_dim=dim_in, fea_dim=opt.feat_dim,
                           mlp=opt.mlp, symmetric=opt.symmetric, m=opt.moco_momentum,
                           K=opt.queue_size, T=opt.moco_temp)
        optimizer = torch.optim.SGD(params=moco.parameters(),
                                    lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        moco = convert_to_ddp(moco)
        self.logger.modules = [moco, optimizer]
        self.moco = moco
        self.optimizer = optimizer
        self.feature_extractor = moco.module.encoder_k

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--mlp', help='Projection head for moco v2', dest='mlp', action='store_true')
        parser.add_argument('--symmetric', help='Symmetric contrastive loss', dest='symmetric', action='store_true')
        parser.add_argument('--moco_momentum', type=float, default=0.999, help='Moving Average Momentum')
        parser.add_argument('--queue_size', type=int, default=65536, help='Memory queue size')
        parser.add_argument('--moco_temp', type=float, default=0.07, help='temp for contrastive loss, 0.1 for cifar10')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        images, labels = inputs
        self.moco.train()

        im_q, im_k = images

        # compute loss
        loss = self.moco(im_q, im_k)

        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.msg([loss, ], n_iter)
