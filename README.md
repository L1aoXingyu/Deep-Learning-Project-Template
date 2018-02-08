# MxTorch
My own deep learning lib based on PyToch, learned from MxNet/Gluon and Chainer.

## Example

Classification Network Example

```python
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
        model = get_model()
        criterion = get_criterion()
        optimizer = get_optimizer(model)

        super().__init__(model, criterion, optimizer)

        self.metric_meter['loss'] = meter.AverageValueMeter()
        self.metric_meter['acc'] = meter.AverageValueMeter()

    def train(self, train_data):
        for data in tqdm(train_data):
            img, label = data
            if torch.cuda.is_available() and opt.use_gpu:
                img = img.cuda(opt.ctx)
                label = label.cuda(opt.ctx)
            img = Variable(img)
            label = Variable(label)
            score = self.model(img)
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

    def test(self, test_data):
        for data in tqdm(test_data):
            img, label = data
            if torch.cuda.is_available() and opt.use_gpu:
                img = img.cuda(opt.ctx)
                label = label.cuda(opt.ctx)
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
            score = self.model(img)
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


def train(**kwargs):
    opt._parse(kwargs)

    # Get train data and test data.
    train_data = get_train_data()
    test_data = get_test_data()
    model_trainer = ModelTrainer()

    model_trainer.fit(train_data, test_data)


if __name__ == '__main__':
    import fire

    fire.Fire()
```

