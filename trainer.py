__all__ = ['Trainer', 'ScheduledOptim']
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from config import opt
from tensorboardX import SummaryWriter
from torch.autograd import Variable


class Trainer(object):
    """Base class for all trainer.

    Your model trainer should also subclass this class.
    """

    def __init__(self,
                 train_data=None,
                 test_data=None,
                 model=None,
                 criterion=None,
                 optimizer=None):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.config = self.write_config()
        self.write_result = hasattr(opt, 'result_file')

        # Set tensorboard configuration.
        if hasattr(opt, 'vis_dir'):
            self.writer = SummaryWriter(opt.vis_dir)

        # Set metrics meter log configuration.
        self.metric_log = OrderedDict()
        self.metric_meter = OrderedDict()

        self.n_iter = 0
        self.n_plot = 0

    def train(self):
        """Train a epoch in the whole train set, update in metric dict and tensorboard.
                Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def test(self):
        """Test model in test set, update in metric dict and tensorboard.
                Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def fit(self):
        if self.write_result:
            with open(opt.result_file, 'a') as f:
                f.write(self.config + '\n')
        for e in range(1, opt.max_epoch + 1):
            if hasattr(opt, 'lr_decay_freq') and hasattr(
                    opt, 'lr_decay') and e % opt.lr_decay_freq == 0:
                self.optimizer.lr_multi(opt.lr_decay)

            # Train model on train dataset.
            self.model.train()
            self.reset_meter()
            self.train()

            # Evaluate model on test dataset.
            try:
                self.model.eval()
                self.reset_meter()
                self.test()
            except NotImplementedError:
                print('No test data!')

            # Print metric log on screen and write out metric in result file.
            self.print_config(e)

            # Save model.
            if hasattr(opt, 'save_freq') and hasattr(opt, 'save_file'):
                if e % opt.save_freq == 0:
                    # TODO: add metric to save name.
                    self.save()

    def reset_meter(self):
        for k, v in self.metric_meter.items():
            v.reset()

    def print_config(self, epoch):
        epoch_str = 'Epoch: {}, '.format(epoch)
        for m, v in self.metric_log.items():
            epoch_str += (m + ': ')
            epoch_str += '{:.4f}, '.format(v)
        epoch_str += 'lr: {:.1e}'.format(self.optimizer.learning_rate)
        print(epoch_str)
        print()
        if self.write_result:
            with open(opt.result_file, 'a') as f:
                f.write(epoch_str + '\n')

    @staticmethod
    def write_config():
        config_str = 'Configure: \n'
        if hasattr(opt, 'model'):
            config_str += 'model: ' + str(opt.model) + '\n'
        if hasattr(opt, 'max_epoch'):
            config_str += 'epochs: ' + str(opt.max_epoch) + '\n'
        if hasattr(opt, 'lr'):
            config_str += 'lr: ' + str(opt.lr) + '\n'
        if hasattr(opt, 'lr_decay_freq'):
            config_str += 'lr_decay_freq: ' + str(
                opt.lr_decay_freq) + '\n'
            config_str += 'lr_decay: ' + str(opt.lr_decay) + '\n'
        if hasattr(opt, 'weight_decay'):
            config_str += 'weight_decay: ' + str(opt.weight_decay) + '\n'
        return config_str

    def save(self):
        """Save model, default name is net + time, such as net_0101_23:57:28.pth

        """
        if not os.path.exists(opt.save_file):
            os.mkdir(opt.save_file)
        prefix = os.path.join(opt.save_file, opt.model) + '_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        if hasattr(self.model, 'module'):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def find_lr(self, begin_lr=1e-5, end_lr=1.):
        import matplotlib.pyplot as plt
        self.model.train()
        self.optimizer.set_learning_rate(begin_lr)
        lr_mult = (end_lr / begin_lr) ** (1. / 100)
        lr = []
        losses = []
        best_loss = 1e9
        for data in self.train_data:
            im, label = data
            if torch.cuda.is_available() and opt.use_gpu:
                im = im.cuda()
                label = label.cuda()
            im = Variable(im)
            label = Variable(label)
            # forward
            score = self.model(im)
            loss = self.criterion(score, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            lr.append(self.optimizer.learning_rate)
            losses.append(loss.data[0])
            self.optimizer.multi(lr_mult)
            if loss.data[0] < best_loss:
                best_loss = loss.data[0]
            if loss.data[0] > 3 * best_loss or self.optimizer.learning_rate > 1.:
                break
        plt.figure()
        plt.xticks(
            np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]),
            (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.plot(np.log(lr), losses)
        plt.show()


class ScheduledOptim(object):
    """A wrapper class for learning rate scheduling
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.current_steps = 0

    def step(self):
        "Step by the inner optimizer"
        self.current_steps += 1
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def lr_multi(self, multi):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= multi
        self.lr = self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr
