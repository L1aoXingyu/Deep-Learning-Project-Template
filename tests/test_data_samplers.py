# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import unittest

sys.path.append('.')
from config.defaults import _C as cfg
from data.transforms import build_transforms
from data.build import build_dataset
from solver.build import make_optimizer
from modeling import build_model


class TestDataSet(unittest.TestCase):
    def test_optimzier(self):
        model = build_model(cfg)
        optimizer = make_optimizer(cfg, model)
        from IPython import embed;
        embed()

    def test_cfg(self):
        cfg.merge_from_file('configs/train_mnist_softmax.yml')
        from IPython import embed;
        embed()

    def test_dataset(self):
        train_transform = build_transforms(cfg)
        val_transform = build_transforms(cfg, False)
        train_set = build_dataset(train_transform)
        val_test = build_dataset(val_transform, False)
        from IPython import embed;
        embed()


if __name__ == '__main__':
    unittest.main()
