from torch.optim import Adam, SGD


def get_optimizer(
        settings,
        model):

    # Disable decay for bias parameters and parameters of batch normalization layers
    if settings['optimizer']['bn_and_bias_wo_decay']:
        params = model.named_parameters()
        decay = []
        no_decay = []
        for name, param in params:
            if 'bn' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        params = [{'params': decay},
                  {'params': no_decay, 'weight_decay': 0.0}]
    else:
        params = model.parameters()

    if settings['optimizer']['optimizer'] == 'adam':
        optimizer = Adam(
            params=params,
            lr=settings['optimizer']['learning_rate'],
            weight_decay=settings['optimizer']['weight_decay']
        )
    elif settings['optimizer']['optimizer'] == 'sgd':
        optimizer = SGD(
            params=params,
            lr=settings['optimizer']['learning_rate'],
            weight_decay=settings['optimizer']['weight_decay'],
            momentum=settings['optimizer']['momentum'],
            nesterov=settings['optimizer']['nesterov']
        )

    return optimizer
