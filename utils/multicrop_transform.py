# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : multicrop_transform.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:24 PM 
'''

import random
import cv2
from PIL import ImageFilter
import numpy as np
import torchvision.transforms as transforms


class MultiCropTransform(object):

    def __init__(self,
                 old_transform: transforms.Compose,
                 size_crops,
                 nmb_crops,
                 min_scale_crops,
                 max_scale_crops):
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        trans = []
        for i in range(len(size_crops)):
            # REPLACE
            transform = []
            for t in old_transform.transforms:
                if isinstance(t, transforms.RandomResizedCrop):
                    transform.append(transforms.RandomResizedCrop(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]),
                    ))
                    continue
                transform.append(t)

            trans.extend([transforms.Compose(transform)] * nmb_crops[i])
        self.trans = trans

    def __call__(self, img):
        multi_crops = list(map(lambda trans: trans(img), self.trans))
        return multi_crops