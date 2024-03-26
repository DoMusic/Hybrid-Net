import torch.nn.functional as F
from common import CHORD_LABELS, SEGMENT_LABELS


def build_masked_stft(masks, stft_feature, n_fft=4096):
    out = []
    for i in range(len(masks)):
        mask = masks[i, :, :, :]
        pad_num = n_fft // 2 + 1 - mask.size(-1)
        mask = F.pad(mask, (0, pad_num, 0, 0, 0, 0))
        inst_stft = mask.type(stft_feature.dtype) * stft_feature
        out.append(inst_stft)
    return out


def get_chord_name(chord_idx_list):
    chords = [CHORD_LABELS[idx] for idx in chord_idx_list]
    return chords


def get_segment_name(segments):
    segments = [SEGMENT_LABELS[idx] for idx in segments]
    return segments


def get_lyrics(waveform, sr, cfg):
    # asr and wav2vec2
    raise NotImplementedError()
