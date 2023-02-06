import torch
from torchmetrics import functional


def acc(
    settings,
    average):

    def acc_fixed(y_pred, y_true):

        y_pred = torch.softmax(y_pred, dim=1)
        metric = functional.accuracy(
            y_pred,
            y_true,
            average=average,
            num_classes=settings['model']['class_num']
        )
        return metric

    return acc_fixed


def f1(
    settings,
    average,
    idx=0):

    def f1_fixed(y_pred, y_true):

        y_pred = torch.softmax(y_pred, dim=1)
        metric = functional.f1_score(
            y_pred,
            y_true,
            average=average,
            num_classes=settings['model']['class_num']
        )

        if average is None:
            metric = metric[idx]  # return f1 score for specific class

        return metric

    return f1_fixed


def get_metrics_functions(
        settings
):

    _metric = acc(settings, average='micro')
    _metric.__name__ = settings['metrics']['metric_names'][0]
    return [_metric]
