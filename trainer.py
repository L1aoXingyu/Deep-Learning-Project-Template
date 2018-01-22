__all__ = ['Trainer', 'ScheduledOptim']
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from . import meter


# class DefaultConfig(object):
#     model = 'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

#     train_data_root = './data/train/'  # 训练集存放路径
#     test_data_root = './data/test1'  # 测试集存放路径
#     load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

#     batch_size = 128  # batch size
#     use_gpu = True  # use GPU or not
#     num_workers = 4  # how many workers for loading data
#     print_freq = 20  # print info every N batch
#     save_freq = 20  # save model every N epochs
#     save_file = './checkpoints/'
#     debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
#     result_file = 'result.txt'

#     max_epoch = 10
#     lr = 0.1  # initial learning rate
#     lr_decay = 0.95
#     weight_decay = 1e-4


class Trainer(object):
    """ A base class for model training ane evaluating

    """

    def __init__(self,
                 opt=None,
                 train_data=None,
                 test_data=None,
                 model=None,
                 criterion=None,
                 optimizer=None):
        self.opt = opt
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.config = self.write_config()
        self.train_loss = meter.AverageValueMeter()
        self.train_acc = meter.AverageValueMeter()
        self.test_loss = meter.AverageValueMeter()
        self.test_acc = meter.AverageValueMeter()

    def train(self):
        self.train_loss.reset()
        self.train_acc.reset()
        self.model.train()
        for i, data in enumerate(self.train_data):
            im, label = data
            if torch.cuda.is_available() and self.opt.use_gpu:
                im = im.cuda()
                label = label.cuda()
            im = Variable(im)
            label = Variable(label)
            score = self.model(im)
            loss = self.criterion(score, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_loss.add(loss.data[0])
            acc = (score.max(1)[1] == label).type(torch.FloatTensor).mean()
            self.train_acc.add(acc.data[0])
            if (i + 1) % self.opt.print_freq == 0:
                epoch_str = (
                    '{}/{}, Train Loss: {:.4f}, Train Acc: {:.4f} '.format(
                        i + 1, len(self.train_data),
                        self.train_loss.value()[0],
                        self.train_acc.value()[0]))
                print(epoch_str)

        train_str = (' Train Loss: {:.4f}, Train Acc: {:.4f}'.format(
            self.train_loss.value()[0],
            self.train_acc.value()[0]))
        return train_str

    def test(self):
        self.test_loss.reset()
        self.test_acc.reset()
        self.model.eval()
        for i, data in enumerate(self.test_data):
            im, label = data
            if torch.cuda.is_available() and self.opt.use_gpu:
                im = im.cuda()
                label = label.cuda()
            im = Variable(im, volatile=True)
            label = Variable(label, volatile=True)
            score = self.model(im)
            loss = self.criterion(score, label)
            self.test_loss.add(loss.data[0])
            acc = (score.max(1)[1] == label).type(torch.FloatTensor).mean()
            self.test_acc.add(acc.data[0])

        test_str = ('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(
            self.test_loss.value()[0],
            self.test_acc.value()[0]))
        return test_str

    def fit(self):
        with open(self.opt.result_file, 'a') as f:
            f.write(self.config + '\n')
        for e in range(1, self.opt.max_epoch + 1):
            if hasattr(self.opt, 'lr_decay_freq') and hasattr(
                    self.opt, 'lr_decay') and e % self.opt.lr_decay_freq == 0:
                self.optimizer.lr_multi(self.opt.lr_decay)
            prev_time = datetime.now()
            train_str = self.train()
            test_str = self.test()
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = (' Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s))
            epoch_str = ('Epoch: {},'.format(e) + train_str + ', ' + test_str +
                         ', lr: {:.1e}'.format(self.optimizer.learning_rate) +
                         ',' + time_str)
            print(epoch_str)
            print()
            with open(self.opt.result_file, 'a') as f:
                f.write(epoch_str + '\n')
            if e % self.opt.save_freq == 0:
                self.save()

    def write_config(self):
        config_str = (
                'Configure: \n' + 'model: ' + self.opt.model + '\n' + 'epochs: ' +
                str(self.opt.max_epoch) + '\n' + 'lr: ' + str(self.opt.lr) + '\n')
        if hasattr(self.opt, 'lr_decay_freq'):
            config_str += 'lr_decay_freq: ' + str(
                self.opt.lr_decay_freq) + '\n'
            config_str += 'lr_decay: ' + str(self.opt.lr_decay) + '\n'
        if hasattr(self.opt, 'weight_decay'):
            config_str += 'weight_decay: ' + str(self.opt.weight_decay) + '\n'

        return config_str

    def save(self):
        ''' save model, default name is net + time, such as net_0101_23:57:28.pth '''
        if not os.path.exists(self.opt.save_file):
            os.mkdir(self.opt.save_file)
        prefix = os.path.join(self.opt.save_file, self.opt.model) + '_'
        name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        if hasattr(self.model, 'module'):
            torch.save(self.model.module.state_dict(), name)
        else:
            torch.save(self.model.state_dict(), name)

    def find_lr(self, begin_lr=1e-5, end_lr=1.):
        self.model.train()
        self.optimizer.set_learning_rate(begin_lr)
        lr_mult = (end_lr / begin_lr) ** (1. / 100)
        lr = []
        losses = []
        best_loss = 1e9
        for data in self.train_data:
            im, label = data
            if torch.cuda.is_available() and self.opt.use_gpu:
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
    '''A wrapper class for learning rate scheduling'''

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
