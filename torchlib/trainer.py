__all__ = ['Trainer', 'ScheduledOptim']
import os
import time
from collections import OrderedDict

import numpy as np
import torch

try:
    from config import opt
except ImportError:
    from .config import opt


class Trainer(object):
    """Base class for all trainer.

    Your model trainer should also subclass this class.
    """

    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.config = self.write_config()

        # Set tensorboard configuration.
        if hasattr(opt, 'vis_dir'):
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(opt.vis_dir)

        # Set metrics meter log configuration.
        self.metric_log = OrderedDict()
        self.metric_meter = OrderedDict()

        self.n_iter = 0
        self.n_plot = 0

        self.best_model = None
        self.best_metric = 1e9

    def train(self, kwargs):
        """Train a epoch in the whole train set, update in metric dict and tensorboard.

        """
        raise NotImplementedError

    def test(self, kwargs):
        """Test model in test set, update in metric dict and tensorboard.
                Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def fit(self, **kwargs):
        if hasattr(opt, 'result_file'):
            with open(opt.result_file, 'a') as f:
                f.write(self.config + '\n')

        if 'epochs' in kwargs:
            max_epochs = kwargs['epochs']
        else:
            max_epochs = opt.max_epoch

        for e in range(1, max_epochs + 1):
            if hasattr(opt, 'lr_decay_freq') and hasattr(
                    opt, 'lr_decay') and e % opt.lr_decay_freq == 0:
                self.optimizer.lr_multi(opt.lr_decay)

            # Train model on train dataset.
            self.train(kwargs)

            # Evaluate model on test dataset.
            try:
                self.test(kwargs)
            except NotImplementedError:
                print('No test data!')

            # Print metric log on screen and write out metric in result file.
            self.print_config(e)

            # Save model every N epochs.
            if hasattr(opt, 'save_freq') and hasattr(opt, 'save_file'):
                if hasattr(opt, 'save_best') and opt.save_best:
                    try:
                        self.get_best_model()
                    except NotImplementedError:
                        print('You need to implement get_best_model!')
                if e % opt.save_freq == 0:
                    # TODO: add metric to save name.
                    self.save()

        if hasattr(opt, 'save_best') and opt.save_best and (self.best_model is not None):
            prefix = os.path.join(opt.save_file, opt.model) + '_'
            name = prefix + 'best_model.pth'
            if not os.path.exists(opt.save_file):
                os.mkdir(opt.save_file)
            torch.save(self.best_model, name)
        else:
            print('do not save best model!')

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
        if hasattr(opt, 'result_file'):
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

    def get_best_model(self):
        raise NotImplementedError

    def find_lr_step(self, data):
        """This is using for find best learning rate, and it's optional.
                If your network is not standard, you need to ovride this subclasses.

        Returns:
            it should return a loss(~torch.autograd.Variable).
        """
        img, label = data
        if opt.use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = torch.autograd.Variable(img)
        label = torch.autograd.Variable(label)
        score = self.model(img)
        loss = self.criterion(score, label)
        return loss

    def find_lr(self, train_data, begin_lr=1e-5, end_lr=1.):
        import matplotlib.pyplot as plt
        self.model.train()
        self.optimizer.set_learning_rate(begin_lr)
        lr_mult = (end_lr / begin_lr) ** (1. / 100)
        lr = list()
        losses = list()
        x_ticks = list()
        best_loss = 1e9
        for data in train_data:

            loss = self.find_lr_step(data)

            lr.append(self.optimizer.learning_rate)
            losses.append(loss.data[0])
            self.optimizer.lr_multi(lr_mult)
            if loss.data[0] < best_loss:
                best_loss = loss.data[0]
            if loss.data[0] > 3 * best_loss or self.optimizer.learning_rate > 1.:
                break
        plt.figure()
        end_lr = self.optimizer.learning_rate
        lr_now = begin_lr
        while True:
            if lr_now <= end_lr:
                x_ticks.append(lr_now)
                lr_now *= 10
            else:
                break
        plt.xticks(
            np.log(x_ticks), x_ticks)
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
