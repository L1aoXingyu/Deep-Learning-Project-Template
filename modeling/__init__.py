# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .example_model import ResNet18


def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
