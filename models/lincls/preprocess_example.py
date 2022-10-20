# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : preprocess_example.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/20 1:03 PM 
'''
import torch

if __name__ == '__main__':
    from models.propos.byol_wrapper import BYOLWrapper
    from network import backbone_dict

    backbone = 'resnet50'
    # backbone = 'bigresnet18'
    encoder_type, dim_in = backbone_dict[backbone]
    encoder = encoder_type()
    byol = BYOLWrapper(encoder,
                       num_cluster=10,
                       in_dim=dim_in,
                       temperature=0.5,
                       hidden_size=4096,
                       fea_dim=256,
                       byol_momentum=0.999,
                       symmetric=True,
                       shuffling_bn=True,
                       latent_std=0.001)

    checkpoint = ''
    msg = byol.load_state_dict(torch.load(checkpoint, map_location='cpu')['byol'], strict=False)
    encoder = byol.encoder_k
    torch.save(encoder.state_dict(), 'encoder_checkpoint.pth')
