import torch
import torchvision

from supervised_classification.blocks.models_blocks.models_wide_resnet import WideResNet


def get_classifier(settings):

    if settings['model']['backbone'] == 'resnet50':
        if settings['model']['pretrained']:
            weights = torchvision.models.ResNet50_Weights
        else:
            weights = None

        model = torchvision.models.resnet50(weights=weights)
        fc_layer = [torch.nn.Linear(in_features=2048, out_features=settings['model']['class_num'])]
        model.fc = torch.nn.Sequential(*fc_layer)

    elif settings['model']['backbone'] == 'resnet18':
        if settings['model']['pretrained']:
            weights = torchvision.models.ResNet18_Weights
        else:
            weights = None

        model = torchvision.models.resnet18(weights=weights)
        fc_layer = [torch.nn.Linear(in_features=512, out_features=settings['model']['class_num'])]
        model.fc = torch.nn.Sequential(*fc_layer)

    elif settings['model']['backbone'] == 'wide_resnet':
        model = WideResNet(
            first_stride=1,
            depth=28,
            num_classes=settings['model']['class_num'],
            widen_factor=2,
            drop_rate=0,
            is_remix=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
