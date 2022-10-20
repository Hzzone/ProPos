import copy
import argparse
import torch
from utils.ops import convert_to_ddp
from .simclr_wrapper import SimCLRWrapper
from models.basic_template import TrainTask
from network import backbone_dict
from models import model_dict


@model_dict.register('simclr')
class SimCLR(TrainTask):

    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        self.num_cluster = self.num_classes if opt.num_cluster is None else opt.num_cluster
        simclr = SimCLRWrapper(encoder,
                               in_dim=dim_in, fea_dim=opt.feat_dim,
                               temperature=opt.temperature
                               )
        optimizer = torch.optim.SGD(params=self.collect_params(simclr),
                                    lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        self.feature_extractor_copy = copy.deepcopy(simclr.encoder).cuda()
        psedo_labels = self.psedo_labels
        self.logger.modules = [simclr, optimizer, psedo_labels]
        # Initialization
        simclr = simclr.cuda()
        simclr = convert_to_ddp(simclr)
        self.simclr = simclr
        self.optimizer = optimizer
        self.feature_extractor = simclr.module.encoder

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument('--temperature', help='temperature', type=float, default=1.)
        return parser

    def train(self, inputs, n_iter):
        opt = self.opt

        images, labels = inputs
        self.simclr.train()

        im_q, im_k = images

        # compute loss
        contrastive_loss = self.simclr(im_q, im_k)

        loss = contrastive_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.logger.msg([contrastive_loss, ], n_iter)
