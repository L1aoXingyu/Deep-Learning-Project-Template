import warnings
from pprint import pprint


class DefaultConfig(object):
    # model = 'AlexNet'
    #
    # # Dataset.
    # train_data_path = './data/train/'
    # test_data_path = './data/test/'
    #
    # # Store result and save models.
    # result_file = 'result.txt'
    # save_file = './checkpoints/'
    # save_freq = 30  # save model every N epochs
    # save_best = False  # If save best test metric model.
    #
    # # Visualization parameters.
    # vis_dir = './vis/'
    # plot_freq = 100  # plot in tensorboard every N iterations
    #
    # # Model hyperparameters.
    # use_gpu = True  # use GPU or not
    # ctx = 0  # running on which cuda device
    # batch_size = 128  # batch size
    # num_workers = 4  # how many workers for loading data
    # max_epoch = 10
    # lr = 0.1  # initial learning rate
    # lr_decay = 0.95
    # lr_decay_freq = 10
    # weight_decay = 1e-4

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('=========user config==========')
        pprint(self._state_dict())
        print('============end===============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
