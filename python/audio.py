import torchaudio
import torch


def read_wav(wav_fp, sample_rate=44100, n_channel=2, device='cpu'):
    waveform, sr = torchaudio.load(wav_fp)
    assert waveform.ndim == 2
    assert waveform.shape[0] == n_channel
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    waveform = waveform.to(device)
    return waveform, sample_rate


def gen_wav(sample_rate=44100, n_channel=2, duration=120, device='cpu'):
    waveform = torch.randn(n_channel, sample_rate * duration)
    waveform = waveform.to(device)
    return waveform, sample_rate


def write_wav(path, wav, sample_rate=44100):
    torchaudio.save(path, wav, sample_rate)
