__all__ = ['ScheduledOptim']


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

    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @property
    def learning_rate(self):
        return self.lr
