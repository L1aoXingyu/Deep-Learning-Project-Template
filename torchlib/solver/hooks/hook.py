# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class Hook(object):

    def before_run(self, solver):
        pass

    def after_run(self, solver):
        pass

    def before_epoch(self, solver):
        pass

    def after_epoch(self, solver):
        pass

    def before_iter(self, solver):
        pass

    def after_iter(self, solver):
        pass

    def before_train_epoch(self, solver):
        self.before_epoch(solver)

    def before_val_epoch(self, solver):
        self.before_epoch(solver)

    def after_train_epoch(self, solver):
        self.after_epoch(solver)

    def after_val_epoch(self, solver):
        self.after_epoch(solver)

    def before_train_iter(self, solver):
        self.before_iter(solver)

    def before_val_iter(self, solver):
        self.before_iter(solver)

    def after_train_iter(self, solver):
        self.after_iter(solver)

    def after_val_iter(self, solver):
        self.after_iter(solver)

    def every_n_epochs(self, solver, n):
        return (solver.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, solver, n):
        return (solver.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, solver, n):
        return (solver.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, solver):
        return solver.inner_iter + 1 == len(solver.data_loader)
