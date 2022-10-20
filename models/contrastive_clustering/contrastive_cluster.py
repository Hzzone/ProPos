import torch
import argparse
import torch.nn as nn

from models.basic_template import TrainTask
from network import backbone_dict
from .cc_wrapper import CCWrapper
from utils.ops import convert_to_ddp

'''
224: knn 86.1 nmi 61.
python -m torch.distributed.launch --nproc_per_node=1 --master_port=17678 main.py --batch_size 256 \
    --epochs 1000 --learning_rate 0.0003 --lr_decay_rate 0 \
    --encoder_name resnet34 --img_size 224 --dataset cifar10 \
    --data_folder /home/zzhuang/DATASET --knn_temp 0.1 --knn_k 200 \
    --save_freq 10 --feat_dim 128 --run_name official --num_workers 32 --weight_decay 0.\
    --model_name contrastive_cluster --instance_temperature 0.5 --cluster_temperature 1.0
    
python -m torch.distributed.launch --nproc_per_node=1 --master_port=17678 main.py --batch_size 256 --epochs 1000 --learning_rate 0.0003 --encoder_name bigresnet18 --img_size 32 --dataset cifar10 --data_folder /home/zzhuang/DATASET --save_freq 1 --feat_dim 128 --run_name cifar10_resnet18 --num_workers 32 --weight_decay 0. --model_name cc --instance_temperature 0.5 --cluster_temperature 1.0 --whole_dataset --reassign 10
'''


class ContrastiveCluster(TrainTask):
    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        contrastive_cluster = CCWrapper(encoder=encoder, dim_in=dim_in, feat_dim=opt.feat_dim,
                                        num_cluster=self.num_classes, batch_size=opt.batch_size,
                                        instance_temperature=opt.instance_temperature,
                                        cluster_temperature=opt.cluster_temperature)
        optimizer = torch.optim.Adam(params=contrastive_cluster.parameters(),
                                     lr=opt.learning_rate, weight_decay=opt.weight_decay)
        self.feature_extractor = nn.Sequential(contrastive_cluster.encoder, contrastive_cluster.instance_projector)

        contrastive_cluster = convert_to_ddp(contrastive_cluster)

        self.logger.modules = [contrastive_cluster, optimizer]
        self.contrastive_cluster = contrastive_cluster
        self.optimizer = optimizer
        self.instance_projector = contrastive_cluster.module.instance_projector
        self.cluster_projector = contrastive_cluster.module.cluster_projector

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument('--instance_temperature', type=float, default=0.5, help='instance_temperature')
        parser.add_argument('--cluster_temperature', type=float, default=1.0, help='cluster_temperature')
        parser.add_argument('--entropy_loss_weight', type=float, default=1.0, help='smooth entropy loss weight')
        parser.add_argument('--num_cluster', type=int, help='num of cluster')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        images, labels = inputs

        im_q, im_k = images

        # compute loss
        loss_instance, loss_cluster, ne_loss = self.contrastive_cluster(im_q, im_k)
        loss = loss_instance + loss_cluster + ne_loss * opt.entropy_loss_weight

        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([loss_instance, loss_cluster, ne_loss, lr], n_iter)

    def adjust_learning_rate(self, n_iter):
        pass
