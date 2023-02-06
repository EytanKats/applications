import pandas as pd
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torch.utils.data import DataLoader

from supervised_classification.settings import settings
from simple_converge.utils.constants import CIFAR10_MEAN, CIFAR10_STD
from simple_converge.datasets.DataframeImageCategoricalDataset import DataframeImageCategoricalDataset


def training_transform(image, label):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize( CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    image_1 = transform(image)
    return image_1, label


def validation_transform(image, label):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    image = transform(image)
    return image, label


def get_data_loaders():

    # Load data files
    train_data_file = pd.read_csv(settings['dataset']['train_data_file_path'])
    val_data_file = pd.read_csv(settings['dataset']['test_data_file_path'])
    test_data_file = pd.read_csv(settings['dataset']['test_data_file_path'])

    # Use part of the training data if 'settings['dataset']['partial']' is True
    if settings['dataset']['partial']:
        train_data_file, _ = train_test_split(
            train_data_file,
            test_size=settings['dataset']['partial_split'],
            stratify=train_data_file[settings['dataset']['label_name_column']]
        )

    # Get labels
    settings['dataset']['labels'] = train_data_file[settings['dataset']['label_name_column']].unique()

    # Create datasets for each fold
    training_datasets = [DataframeImageCategoricalDataset(settings['dataset'], train_data_file, training_transform)
                         for _ in range(settings['manager']['folds_num'])]
    validation_datasets = [DataframeImageCategoricalDataset(settings['dataset'], val_data_file, validation_transform)
                           for _ in range(settings['manager']['folds_num'])]
    test_datasets = [DataframeImageCategoricalDataset(settings['dataset'], test_data_file, validation_transform)
                     for _ in range(settings['manager']['folds_num'])]

    # Create data loaders for each fold
    training_data_loaders = [
        DataLoader(
            dataset=training_datasets[idx],
            batch_size=settings['dataloader']['batch_size'],
            shuffle=True,
            num_workers=settings['dataloader']['workers_num'],
            pin_memory=True
        )
        for idx in range(settings['manager']['folds_num'])
    ]

    validation_data_loaders = [
        DataLoader(
            dataset=validation_datasets[idx],
            batch_size=settings['dataloader']['batch_size'],
            num_workers=settings['dataloader']['workers_num'],
            pin_memory=True
        )
        for idx in range(settings['manager']['folds_num'])
    ]

    test_data_loaders = [
        DataLoader(
            dataset=test_datasets[idx],
            batch_size=settings['dataloader']['batch_size'],
            num_workers=settings['dataloader']['workers_num'],
            pin_memory=True
        )
        for idx in range(settings['manager']['folds_num'])
    ]

    return training_data_loaders, validation_data_loaders, test_data_loaders


def get_test_data_loaders():

    # Load test dataset files
    test_dfs = [pd.read_csv(dataset_file_path) for dataset_file_path in settings['test']['dataset_files']]

    # Get labels
    settings['dataset']['labels'] = test_dfs[0][settings['dataset']['label_name_column']].unique()
    settings['postprocessor']['labels'] = settings['dataset']['labels']

    # Create dataset for each test data file
    test_datasets = [DataframeImageCategoricalDataset(settings['dataset'], test_df, validation_transform)
                     for test_df in test_dfs]

    # Create data loaders for each dataset
    test_data_loaders = [
        DataLoader(
            dataset=test_dataset,
            batch_size=settings['data_loader']['batch_size'],
            num_workers=settings['data_loader']['workers_num'],
            pin_memory=True
        )
        for test_dataset in test_datasets
    ]

    return test_data_loaders
