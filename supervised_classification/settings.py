mlops_settings = {
    'use_mlops': False,
    'project_name': 'SupervisedClassification',
    'task_name': '',
    'task_type': None,
    'connect_frameworks': {
        'matplotlib': False,
        'tensorflow': False,
        'tensorboard': False,
        'pytorch': False,
        'xgboost': False,
        'scikit': False,
        'fastai': False,
        'lightgbm': False,
        'hydra': False
    }
}

manager_settings = {
    'output_folder': '/Users/eytankats/Simulations/cifar10/supervised',
    'folds_num': 5,
    'active_folds': [0],
    'restore_checkpoint': False,
    'restore_checkpoint_path': ''
}

trainer_settings = {
    'epochs': 400,
    'monitor': 'acc_micro',
    'monitor_regime': 'max',
    'ckpt_freq': 1,
    'ckpt_save_best_only': True,
    'use_early_stopping': False,
    'early_stopping_patience': 30,
    'plateau_patience': 5
}

dataset_settings = {
    'train_data_file_path': '/Users/eytankats/Datasets/cifar10/train/metadata.csv',
    'test_data_file_path': '/Users/eytankats/Datasets/cifar10/test/metadata.csv',
    'image_path_column': 'image_path',
    'label_name_column': 'label',
    'label_column': 'label_idx',
    'labels': [],
    'get_image_as_numpy_array': False,

    'partial': True,
    'partial_split': 0.9,
}

data_loader_settings = {
    'batch_size': 32,
    'workers_num': 8
}

model_settings = {
    'backbone': 'resnet18',  # resnet18, resnet50, wide_resnet
    'pretrained': False,  # resnet18, resnet50
    'class_num': 10  # resnet18, resnet50, wide_resnet
}

optimizer_settings = {
    'optimizer': 'sgd',  # sgd, adam
    'learning_rate': 0.03,  # sgd, adam
    'weight_decay': 5e-4,  # sgd, adam
    'momentum': 0.9,  # sgd
    'nesterov': True,  # sgd

    'bn_and_bias_wo_decay': True  # sgd, adam
}

scheduler_settings = {
    'scheduler': 'cosine_annealing',  # cosine_annealing, None
    'epochs_num': 400,
    'min_lr': 1e-4
}

loss_functions_settings = {
    'loss_names': ['ce'],  # ce, focal

    'alpha': 0.25,  # focal
    'gamma': 2.0  # focal
}

metrics_settings = {
    'metric_names': ['acc_micro']
}

app_settings = {
    'registry_name': 'SingleModelApp',
    'use_reduce_lr_on_plateau': False,
    'reduce_lr_on_plateau_factor': 0.9,
    'reduce_lr_on_plateau_min': 1e-6,
    'use_ema': True,
    'ema_decay': 0.999
}

postprocessor_settings = {
    'registry_name': 'DataframeImageCategoricalPostprocessor',
    'labels': [],
    'activation': 'softmax',
    'per_class_classification_report': True,
    'confusion_matrix': True,
    'recall_vs_discarded_images': False,
    'per_class_f1_vs_discarded_images': False
}

settings = {
    'mlops': mlops_settings,
    'manager': manager_settings,
    'trainer': trainer_settings,
    'dataset': dataset_settings,
    'dataloader': data_loader_settings,
    'model': model_settings,
    'optimizer': optimizer_settings,
    'scheduler': scheduler_settings,
    'loss': loss_functions_settings,
    'metrics': metrics_settings,
    'app': app_settings,
    'postprocessor': postprocessor_settings
}
