import torch
import torchaudio


def load_waveform(wav_fp, samplerate=16000, n_channel=2, device='cpu'):
    waveform, sr = torchaudio.load(wav_fp)
    assert waveform.ndim == 2
    if waveform.shape[0] != n_channel:
        if n_channel == 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif n_channel <= waveform.shape[0]:
            waveform = waveform[:n_channel]
        else:
            raise ValueError(f'Invalid number of channels: {waveform.shape[0]}')
    if sr != samplerate:
        waveform = torchaudio.transforms.Resample(sr, samplerate)(waveform)
    waveform = waveform.to(device)
    return waveform, samplerate


def gen_waveform(samplerate=16000, n_channel=2, duration=120, device='cpu'):
    waveform = torch.randn(n_channel, samplerate * duration)
    waveform = waveform.to(device)
    return waveform, samplerate


def save_waveform(path, wav, samplerate=16000):
    torchaudio.save(path, wav, samplerate)
