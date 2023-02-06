import torch
import kornia


def focal_loss(settings):

    def focal_loss_fixed(y_pred, y_true):

        loss = kornia.losses.focal_loss(
            y_pred,
            y_true,
            alpha=settings['loss']['alpha'],
            gamma=settings['loss']['gamma'],
            reduction='mean'
        )
        return loss

    return focal_loss_fixed


def ce():
    return torch.nn.CrossEntropyLoss(reduction='mean')


def get_losses_functions(settings):

    if settings['loss']['loss_names'][0] == 'ce':
        _loss_fn = ce()

    elif settings['loss']['loss_names'][0] == 'focal':
        _loss_fn = focal_loss(settings)

    _loss_fn.__name__ = settings['loss']['loss_names'][0]
    return [_loss_fn]
