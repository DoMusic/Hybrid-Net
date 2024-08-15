import torch
from torchaudio.pipelines import (HUBERT_BASE, HUBERT_LARGE)


def get_model(model, config, model_path=None, is_train=True, device='cpu'):
    net = model(**config)
    if model_path:
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    if is_train:
        net.train()
    else:
        net.eval()

    return net


def get_hubert_model(name='base', dl_kwargs=None, device='cpu'):
    if name == 'base':
        bundle = HUBERT_BASE
    elif name == 'large':
        bundle = HUBERT_LARGE
    else:
        raise ValueError(f'Invalid model name: {name}')

    model = bundle.get_model(dl_kwargs=dl_kwargs)
    model.to(device)
    model.eval()
    return model
