from torch.optim.lr_scheduler import CosineAnnealingLR


def get_scheduler(
        settings,
        optimizer):

    if settings['scheduler']['scheduler'] == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=settings['scheduler']['epochs_num'],
            eta_min=settings['scheduler']['min_lr']
        )
    else:
        scheduler = None

    return scheduler
