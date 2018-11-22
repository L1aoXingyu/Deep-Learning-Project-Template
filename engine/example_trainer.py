# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from tqdm import tqdm


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        checkpointer,
        device,
        checkpoint_period,
        log_period,
        epochs,
):
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'ce_loss': Loss(loss_fn)}, device=device)

    desc = "ITERATION -loss: {:.3f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader), desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_period)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        tqdm.write(
            "Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        tqdm.write(
            "Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss)
        )

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()
