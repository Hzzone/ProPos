import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import InstanceLoss, ClusterLoss


class CCWrapper(nn.Module):
    def __init__(self, encoder, dim_in, feat_dim, num_cluster,
                 batch_size, instance_temperature, cluster_temperature):
        super(CCWrapper, self).__init__()
        instance_projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, feat_dim),
        )
        cluster_projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, num_cluster),
            nn.Softmax(dim=1)
        )
        self.encoder = encoder
        self.instance_projector = instance_projector
        self.cluster_projector = cluster_projector
        self.criterion_instance = InstanceLoss(batch_size, instance_temperature)
        self.criterion_cluster = ClusterLoss(num_cluster, cluster_temperature)

    def forward(self, x_i, x_j):
        self.encoder.train()
        x = torch.cat([x_i, x_j], dim=0)
        h = self.encoder(x)
        h_i, h_j = torch.chunk(h, 2, dim=0)

        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)
        loss_instance = self.criterion_instance(z_i, z_j)
        loss_cluster, ne_loss = self.criterion_cluster(c_i, c_j)

        return loss_instance, loss_cluster, ne_loss
