import torch

from models.unet import UNets
from models.beat_net import BeatNet
from models.chord_net import ChordNet
from models.pitch_net import PitchNet
from models.segment_net import SegmentNet


def get_model_cls(s):
    if s == 'unet':
        return UNets
    elif s == 'beat':
        return BeatNet
    elif s == 'chord':
        return ChordNet
    elif s == 'pitch':
        return PitchNet
    elif s == 'segment':
        return SegmentNet
    else:
        raise ValueError(f'Invalid model name: {s}')


def get_model(model, config, model_path=None, is_train=True, device='cpu'):
    if isinstance(model, str):
        model = get_model_cls(model)

    net = model(**config)
    if model_path:
        net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    if is_train:
        net.train()
    else:
        net.eval()

    return net
