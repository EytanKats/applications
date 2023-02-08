import os
import glob
from clearml import TaskTypes

from simple_converge.manager.Manager import predict
from simple_converge.utils.RunMode import RunMode
from simple_converge.mlops.MLOpsTask import MLOpsTask

from settings import settings
from supervised_classification.blocks.models import get_classifier
from supervised_classification.blocks.data_loaders import get_test_data_loaders


if __name__ == "__main__":

    # Create MLOps task and log settings
    settings['mlops']['task_name'] = f'{settings["mlops"]["task_name"]}_test'
    settings['mlops']['task_type'] = TaskTypes.testing
    mlops_task = MLOpsTask(
        settings=settings['mlops']
    )

    # Get test data files and checkpoints for active folds
    test_settings = {}
    test_dataset_files = []
    test_checkpoints = []
    for fold in settings['manager']['active_folds']:
        test_dataset_file = os.path.join(settings['manager']['output_folder'], str(fold), 'test_data.csv')
        test_dataset_files.append(test_dataset_file)

        checkpoints_template = os.path.join(settings['manager']['output_folder'], str(fold), 'checkpoint/ckpt-*')
        checkpoints_paths = glob.glob(checkpoints_template)
        latest_checkpoint = checkpoints_paths[-1]
        test_checkpoints.append(latest_checkpoint)

    test_settings['dataset_files'] = test_dataset_files
    test_settings['checkpoints'] = test_checkpoints
    settings['test'] = test_settings

    # Get data loaders for test
    test_data_loaders = get_test_data_loaders(settings)

    # Run test
    predict(
        settings,
        mlops_task=mlops_task,
        architecture=get_classifier,
        app=None,
        test_dataset=None,
        test_loader=test_data_loaders,
        postprocessor=None,
        run_mode=RunMode.TEST
    )

    exit()
