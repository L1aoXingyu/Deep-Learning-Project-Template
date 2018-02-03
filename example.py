# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import torch
from config import opt
from mxtorch import meter
from mxtorch.trainer import Trainer
from torch.autograd import Variable
from tqdm import tqdm


def get_train_data():
    pass


def get_test_data():
    pass


def get_model():
    pass


def get_criterion():
    pass


def get_optimizer(model):
    pass


class ModelTrainer(Trainer):
    def __init__(self):
        train_data = get_train_data()
        test_data = get_test_data()
        model = get_model()
        criterion = get_criterion()
        optimizer = get_optimizer(model)

        super().__init__(train_data, test_data, model, criterion, optimizer)

        self.metric_meter['loss'] = meter.AverageValueMeter()
        self.metric_meter['acc'] = meter.AverageValueMeter()

    def train(self):
        for data in tqdm(self.train_data):
            im, label = data
            if torch.cuda.is_available() and opt.use_gpu:
                im = im.cuda(opt.ctx)
                label = label.cuda(self.opt.ctx)
            im = Variable(im)
            label = Variable(label)
            score = self.model(im)
            loss = self.criterion(score, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update meter.
            self.metric_meter['loss'].add(loss.data[0])
            acc = (score.max(1)[1] == label).type(torch.FloatTensor).mean()
            self.metric_meter['acc'].add(acc.data[0])

            # Update to tensorboard.
            if (self.n_iter + 1) % opt.plot_freq == 0:
                self.writer.add_scalars('loss', {'train': self.metric_meter['loss'].value()[0]}, self.n_plot)
                self.writer.add_scalars('acc', {'train': self.metric_log['acc'].value()[0]}, self.n_plot)
                self.n_plot += 1
                self.n_iter += 1

        # Log the train metrics to dict.
        self.metric_log['train loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['train acc'] = self.metric_meter['acc'].value()[0]

    def test(self):
        for data in tqdm(self.test_data):
            im, label = data
            if torch.cuda.is_available() and opt.use_gpu:
                im = im.cuda(opt.ctx)
                label = label.cuda(opt.ctx)
            im = Variable(im, volatile=True)
            label = Variable(label, volatile=True)
            score = self.model(im)
            loss = self.criterion(score, label)

            # Update meter.
            self.metric_meter['loss'].add(loss.data[0])
            acc = (score.max(1)[1] == label).type(torch.FloatTensor).mean()
            self.metric_meter['acc'].add(acc.data[0])

        # Add to tensorboard.
        self.writer.add_scalars('loss', {'eval': self.metric_meter['loss'].value()[0]}, self.n_plot)
        self.writer.add_scalars('acc', {'test': self.metric_meter['acc'].value()[0]}, self.n_plot)
        self.n_plot += 1

        # Log the test metrics to dict.
        self.metric_log['test loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['test acc'] = self.metric_meter['acc'].value()[0]


model_trainer = ModelTrainer()
model_trainer.fit()
