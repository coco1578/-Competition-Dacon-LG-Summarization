def freeze_norms(model):

    for name, params in model.model.named_parameters():
        if 'norm' in name:
            params.requires_grad = False


def freeze_params(model):

    for params in model.parameters():
        params.requires_grad = False


def freeze(model, norm=True):

    freeze_params(model.model.shared)

    # for d in [model.model.encoder, model.model.decoder]:
    for d in [model.model.encoder]:
        freeze_params(d.embed_positions)
        freeze_params(d.embed_tokens)

    if norm:
        freeze_norms(model)