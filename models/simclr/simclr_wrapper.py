import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.gather_layer import GatherLayer
from .losses import InstanceLoss


class SimCLRWrapper(nn.Module):
    def __init__(self,
                 encoder,
                 in_dim,
                 temperature,
                 fea_dim=256):
        nn.Module.__init__(self)

        self.fea_dim = fea_dim
        self.encoder = nn.Sequential(encoder,
                                     nn.Linear(in_dim, in_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(in_dim, fea_dim))
        self.loss = InstanceLoss(temperature)

    def forward(self, im_q, im_k):
        z = self.encoder(torch.cat([im_q, im_k], dim=0))
        z_i, z_j = z.chunk(2, dim=0)
        z_i = F.normalize(torch.cat(GatherLayer.apply(z_i), dim=0), dim=1)
        z_j = F.normalize(torch.cat(GatherLayer.apply(z_j), dim=0), dim=1)
        loss = self.loss(z_i, z_j)

        return loss
