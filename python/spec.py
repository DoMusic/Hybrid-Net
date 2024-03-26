import torch as th


def stft(x, n_fft=4096, hop_length=1024, pad=True):
    z = th.stft(x,
                n_fft,
                hop_length or n_fft // 4,
                window=th.hann_window(n_fft).to(device=x.device),
                win_length=n_fft,
                normalized=True,
                center=pad,
                return_complex=True,
                pad_mode='reflect')
    z = th.transpose(z, 1, 2)
    return z


def istft(stft_feature, n_fft=4096, hop_length=1024, pad=True):
    stft_feature = th.transpose(stft_feature, 1, 2)
    waveform = th.istft(stft_feature,
                        n_fft,
                        hop_length,
                        window=th.hann_window(n_fft).to(device=stft_feature.device),
                        win_length=n_fft,
                        normalized=True,
                        center=pad)
    return waveform


def get_spec(waveform, cfg):
    spec = stft(waveform, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], pad=cfg['pad'])
    return spec  # channel, freq, time


def get_specs(waveforms, cfg):
    # waveforms shape: sources, channel, time
    S, C, T = waveforms.shape
    _waveforms = waveforms.view(S * C, T)
    specs = stft(_waveforms, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], pad=cfg['pad'])
    return specs.view(S, C, specs.shape[-2], specs.shape[-1])  # sources, channel, freq, time


def get_mixed_spec(waveforms, cfg):
    mixed_waveform = th.sum(waveforms, dim=0)
    mixed_spec = stft(mixed_waveform, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], pad=cfg['pad'])
    return mixed_spec  # channel, freq, time
