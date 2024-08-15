import torch
from librosa import feature


def stft(signal, fft_size=512, hop_length=None, pad=True):
    """
    Perform Short-Time Fourier Transform (STFT) on the input signal.

    Parameters:
    - signal (torch.Tensor): The input signal tensor.
    - fft_size (int): The size of the FFT window. Default is 512.
    - hop_length (int): The number of samples between successive frames. Default is fft_size // 4.
    - pad (bool, optional): whether to pad :attr:`input` on both sides
        so that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
        Default: ``True``

    Returns:
    - torch.Tensor: The STFT of the input signal.
    """
    *ot_shape, n_time = signal.shape
    signal = signal.reshape(-1, n_time)
    result = torch.stft(signal,
                        fft_size,
                        hop_length or fft_size // 4,
                        window=torch.hann_window(fft_size).to(signal),
                        win_length=fft_size,
                        normalized=True,
                        center=pad,
                        return_complex=True,
                        pad_mode='reflect')
    _, freqs, frames = result.shape
    return result.view(*ot_shape, freqs, frames)


def istft(stft_matrix, hop_length=None, signal_length=None, pad=True):
    """
    Perform Inverse Short-Time Fourier Transform (ISTFT) on the input STFT matrix.

    Parameters:
    - stft_matrix (torch.Tensor): The input STFT matrix tensor.
    - hop_length (int): The number of samples between successive frames. Default is None.
    - signal_length (int): The length of the original signal. Default is None.
    - pad (bool, optional): whether to pad :attr:`input` on both sides
        so that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
        Default: ``True``

    Returns:
    - torch.Tensor: The reconstructed time-domain signal.
    """
    *ot_shape, n_freqs, n_frames = stft_matrix.shape
    fft_size = 2 * n_freqs - 2
    stft_matrix = stft_matrix.view(-1, n_freqs, n_frames)
    win_length = fft_size

    result = torch.istft(stft_matrix,
                         fft_size,
                         hop_length,
                         window=torch.hann_window(win_length).to(stft_matrix.real),
                         win_length=win_length,
                         normalized=True,
                         length=signal_length,
                         center=pad)
    _, length = result.shape
    return result.view(*ot_shape, length)


def get_spectrogram(waveform, config):
    """
    Get the spectrogram of the input waveform based on the provided configuration.

    Parameters:
    - waveform (torch.Tensor): The input waveform tensor.
    - config (dict): The configuration dictionary containing 'n_fft', 'hop_length', and 'pad' keys.

    Returns:
    - torch.Tensor: The spectrogram of the input waveform.
    """
    spectrogram = stft(waveform, fft_size=config.get('n_fft', 4096),
                       hop_length=config.get('hop_length', 1024), pad=config.get('pad', True))
    spectrogram = spectrogram.transpose(-1, -2)
    return spectrogram  # channel, freq, time


def chroma(waveform, n_chroma=12, sample_rate=44100, hop_length=512, bins_per_octave=24):
    dtype = waveform.dtype
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    y = feature.chroma_cqt(y=waveform, sr=sample_rate, n_chroma=n_chroma, hop_length=hop_length, n_octaves=12,
                           bins_per_octave=bins_per_octave, cqt_mode='hybrid')
    y = torch.from_numpy(y).to(dtype)
    return y
