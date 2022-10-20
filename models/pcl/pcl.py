import torch
import argparse
import torch.nn.functional as F
import time
import numpy as np
import copy

from models.basic_template import TrainTask
from network import backbone_dict
from .pcl_wrapper import PCLWrapper
from utils.ops import convert_to_ddp
import torch.distributed as dist
from models import model_dict
from torch_clustering import evaluate_clustering

@model_dict.register('pcl')
class PCL(TrainTask):
    def set_model(self):
        opt = self.opt
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder_q = encoder_type()
        encoder_k = encoder_type()
        moco = PCLWrapper(encoder_q, encoder_k, in_dim=dim_in, fea_dim=opt.feat_dim,
                          mlp=opt.mlp, symmetric=opt.symmetric, m=opt.moco_momentum,
                          K=opt.queue_size, T=opt.moco_temp)
        optimizer = torch.optim.SGD(params=moco.parameters(),
                                    lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
        moco = convert_to_ddp(moco)
        opt.num_clusters = [int(x) for x in opt.num_clusters.split(',')]
        psedo_labels = {x: None for x in opt.num_clusters}
        cluster_centers = {x: None for x in opt.num_clusters}
        density = {x: None for x in opt.num_clusters}

        self.moco = moco
        self.optimizer = optimizer
        self.feature_extractor = moco.module.encoder_k
        self.psedo_labels, self.cluster_centers, self.density = psedo_labels, cluster_centers, density
        self.logger.modules = [encoder_q, moco, optimizer, psedo_labels, cluster_centers, density]

    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')

        parser.add_argument('--mlp', help='Projection head for moco v2', dest='mlp', action='store_true')
        parser.add_argument('--symmetric', help='Symmetric contrastive loss', dest='symmetric', action='store_true')
        parser.add_argument('--moco_momentum', type=float, default=0.999, help='Moving Average Momentum')
        parser.add_argument('--queue_size', type=int, default=65536, help='Memory queue size')
        parser.add_argument('--moco_temp', type=float, default=0.07, help='temp for contrastive loss, 0.1 for cifar10')
        parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='default 1.0')
        parser.add_argument('--num_clusters', type=str, help='reassign kmeans')
        return parser

    def train(self, inputs, indices, n_iter):
        opt = self.opt

        psedo_labels = {}
        for num_cluster in opt.num_clusters:
            psedo_labels[num_cluster] = self.psedo_labels[num_cluster][indices]

        images, labels = inputs
        self.moco.train()

        im_q, im_k = images

        # compute loss
        contrastive_loss, cls_loss = self.moco(im_q, im_k, psedo_labels, self.cluster_centers, self.density)
        loss = contrastive_loss
        if ((n_iter - 1) / self.iter_per_epoch) >= opt.warmup_epochs:
            loss += cls_loss * opt.cls_loss_weight

        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.msg([contrastive_loss, cls_loss, lr], n_iter)

    @torch.no_grad()
    def psedo_labeling(self, n_iter):
        opt = self.opt

        self.logger.msg_str('Generating the psedo-labels')

        features, labels = self.extract_features(self.feature_extractor, self.memory_loader)
        for num_cluster in opt.num_clusters:
            psedo_labels, cluster_centers = self.clustering(features, num_cluster)
            dist.barrier()
            self.logger.msg_str(torch.unique(psedo_labels.cpu(), return_counts=True))
            self.logger.msg_str(torch.unique(labels.long().cpu(), return_counts=True))

            nmi, ari, f, acc = evaluate_clustering(labels.cpu().numpy(), psedo_labels.cpu().numpy())
            self.logger.msg_metric({f'{num_cluster}/ema_train_nmi': nmi, f'{num_cluster}/ema_train_ari': ari,
                                    f'{num_cluster}/ema_train_acc': acc}, n_iter)
            dist.broadcast(psedo_labels, src=0)
            dist.broadcast(cluster_centers, src=0)

            self.psedo_labels[num_cluster] = psedo_labels
            self.cluster_centers[num_cluster] = cluster_centers

            # concentration estimation (phi)
            density = np.zeros(num_cluster)
            counts = []
            for i in range(num_cluster):
                d = ((features[psedo_labels == i] - cluster_centers[i]) ** 2).sum(dim=1).cpu().numpy()
                counts.append(len(d))
                if len(d) > 1:
                    d = (np.asarray(d) ** 0.5).mean() / np.log(len(d) + 10)
                    density[i] = d
            # if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i in range(num_cluster):
                if counts[i] <= 1:
                    density[i] = dmax

            density = density.clip(np.percentile(density, 10),
                                   np.percentile(density, 90))  # clamp extreme values for stability
            density = opt.moco_temp * density / density.mean()  # scale the mean to temperature

            self.density[num_cluster] = torch.from_numpy(density)
            self.logger.msg_str(f'{num_cluster}, {str(counts)}, {str(density)}')

        if self.num_cluster not in opt.num_clusters:
            psedo_labels, cluster_centers = self.clustering(features, self.num_cluster)
            dist.barrier()
            nmi, ari, f, acc = evaluate_clustering(labels.cpu().numpy(), psedo_labels.cpu().numpy())
            self.logger.msg_metric({f'{self.num_cluster}/ema_train_nmi': nmi, f'{self.num_cluster}/ema_train_ari': ari,
                                    f'{self.num_cluster}/ema_train_acc': acc}, n_iter)
