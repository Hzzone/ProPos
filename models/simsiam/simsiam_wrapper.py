# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : simsiam_wrapper.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:28 PM 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=3):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        assert num_layers in [2, 3]
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        bias = True
        if num_layers == 3:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.layer2 = nn.Identity()
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=bias),
            # nn.BatchNorm1d(out_dim, affine=False)
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = num_layers

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Predictor(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class SimSiamWrapper(nn.Module):
    def __init__(self, backbone, in_dim, feat_dim, projection_num_layers):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.projector = Projection(in_dim=in_dim, hidden_dim=feat_dim, out_dim=feat_dim,
                                    num_layers=projection_num_layers)
        self.encoder = nn.Sequential(  # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = Predictor(in_dim=feat_dim, hidden_dim=feat_dim // 4, out_dim=feat_dim)

    def forward(self, x1, x2):
        z = self.encoder(torch.cat([x1, x2], dim=0))
        p = self.predictor(z)
        z1, z2 = z.chunk(2, dim=0)
        p1, p2 = p.chunk(2, dim=0)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss, F.normalize(z.detach(), dim=1)
