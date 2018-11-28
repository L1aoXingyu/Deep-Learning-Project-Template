# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy


def inference(
        cfg,
        model,
        val_loader
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("template_model.inference")
    logger.info("Start inferencing")
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy()},
                                            device=device)

    # adding handlers using `evaluator.on` decorator API
    @evaluator.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = evaluator.state.metrics
        avg_acc = metrics['accuracy']
        logger.info("Validation Results - Accuracy: {:.3f}".format(avg_acc))

    evaluator.run(val_loader)
