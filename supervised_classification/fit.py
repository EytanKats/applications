from clearml import TaskTypes

from simple_converge.manager.Manager import fit
from simple_converge.mlops.MLOpsTask import MLOpsTask

from settings import settings
from supervised_classification.blocks.models import get_classifier
from supervised_classification.blocks.loss_functions import get_losses_functions
from supervised_classification.blocks.metrics import get_metrics_functions
from supervised_classification.blocks.schedulers import get_scheduler
from supervised_classification.blocks.optimizers import get_optimizer
from supervised_classification.blocks.data_loaders import get_data_loaders


if __name__ == "__main__":

    # Create MLOps task
    settings['mlops']['task_type'] = TaskTypes.training
    mlops_task = MLOpsTask(
        settings=settings['mlops']
    )

    # Get data loaders for training, validation and test
    training_data_loaders, validation_data_loaders, test_data_loaders = get_data_loaders()

    # Run training
    fit(
        settings,
        mlops_task=mlops_task,
        architecture=get_classifier,
        loss_function=get_losses_functions,
        metric=get_metrics_functions,
        scheduler=get_scheduler,
        optimizer=get_optimizer,
        app=None,
        train_dataset=None,
        train_loader=training_data_loaders,
        val_dataset=None,
        val_loader=validation_data_loaders,
        test_dataset=None,
        test_loader=test_data_loaders,
        postprocessor=None
    )

    exit()

