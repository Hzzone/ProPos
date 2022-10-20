# -*- coding: UTF-8 -*-
'''
@Project : ICLR2022_Codes 
@File    : simsiam.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/1/24 7:23 PM 
'''

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.ops import topk_accuracy, concat_all_gather
from network import backbone_dict
from models import model_dict
from models.basic_template import TrainTask
from utils import extract_features


@model_dict.register('lincls')
class LinearCLS(TrainTask):
    l2_normalize = False

    def set_model(self):
        opt = self.opt
        assert not opt.whole_dataset
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        self.load_pretrained_model(encoder)
        encoder.eval()
        fc = nn.Linear(dim_in, self.num_cluster)
        # init the fc layer
        g = torch.Generator()
        g.manual_seed(0)
        fc.weight.data.normal_(mean=0.0, std=0.01, generator=g)
        fc.bias.data.zero_()
        if opt.lars:
            from utils.optimizers import LARS
            optim = LARS
        else:
            optim = torch.optim.SGD
        optimizer = optim(params=fc.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

        self.logger.modules = [fc, optimizer]
        self.encoder = encoder.cuda()
        self.fc = fc.cuda()
        self.optimizer = optimizer
        self.test_loader = self.build_dataloader(opt.dataset,
                                                 self.test_transform(self.normalize(opt.dataset)),
                                                 train=False,
                                                 sampler=True,
                                                 batch_size=opt.batch_size)[0]

    def load_pretrained_model(self, encoder):
        opt = self.opt

        self.logger.msg_str("=> loading checkpoint '{}'".format(opt.pretrained_path))
        state_dict = torch.load(opt.pretrained_path, map_location="cpu")
        msg = encoder.load_state_dict(state_dict, strict=False)
        self.logger.msg_str(msg)
        self.logger.msg_str("=> loaded pre-trained model '{}'".format(opt.pretrained_path))

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--pretrained_path', type=str)
        parser.add_argument('--lars', help='lars', action='store_true')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        self.encoder.eval()
        images, labels = inputs
        # compute loss
        with torch.no_grad():
            outputs = self.encoder(images)
        outputs = concat_all_gather(outputs)
        labels = concat_all_gather(labels)
        outputs = self.fc(outputs)
        loss = F.cross_entropy(outputs, labels)

        self.optimizer.zero_grad()
        # SGD
        loss.backward()
        self.optimizer.step()

        self.logger.msg([loss], n_iter)

    def psedo_labeling(self, n_iter):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        assert not opt.whole_dataset
        self.encoder.eval()
        preds, labels = extract_features(nn.Sequential(self.encoder, self.fc), self.test_loader)
        acc_top1, acc_top5 = topk_accuracy(preds, labels, (1, 5))
        self.logger.msg_metric([acc_top1, acc_top5], n_iter)

    def train_transform(self, normalize):
        '''
        simclr transform
        :param normalize:
        :return:
        '''
        opt = self.opt
        train_transform = []
        if 'cifar' in opt.dataset:
            train_transform.append(
                transforms.RandomCrop(size=opt.img_size, padding=int(opt.img_size * 0.125),
                                      padding_mode='reflect'))
        else:
            train_transform.append(transforms.RandomResizedCrop(size=opt.img_size))
        train_transform += [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize]
        train_transform = transforms.Compose(train_transform)
        return train_transform