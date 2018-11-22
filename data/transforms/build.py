# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=(0.1307,), std=(0.3081,))
    if is_train:
        transform = T.Compose([
            T.RandomResizedCrop(size=cfg.TRANSFORMS.SIZE, scale=(cfg.TRANSFORMS.MIN_SCALE, cfg.TRANSFORMS.MAX_SCALE)),
            T.RandomHorizontalFlip(p=cfg.TRANSFORMS.PROB),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform
        ])

    return transform
