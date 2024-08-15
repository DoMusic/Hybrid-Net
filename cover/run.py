import os.path

import torch as th
import torch.nn.functional as F
import numpy as np

from load import get_model, get_hubert_model
from utils.audio import load_waveform, save_waveform
from utils.spec import get_spectrogram, istft
from utils.utils import generate_uuid
from models import UNets, PitchNet, CombineNet


def build_masked_stft(masks, stft_feature, n_fft=4096):
    out = []
    for i in range(len(masks)):
        mask = masks[i, :, :, :]
        pad_num = n_fft // 2 + 1 - mask.size(-1)
        mask = F.pad(mask, (0, pad_num, 0, 0, 0, 0))
        inst_stft = mask.type(stft_feature.dtype) * stft_feature
        out.append(inst_stft)
    return out


def merge_wav(stem_list, volume_list=None):
    stem_num = len(stem_list)
    length = min([stem.shape[0] for stem in stem_list])

    for i in range(stem_num):
        stem = stem_list[i][:length]  # shape: time, channel
        if stem.ndim == 1:
            stem = np.tile(np.expand_dims(stem, axis=1), (1, 2))
        stem = stem / np.abs(stem).max()
        stem_list[i] = stem
        if volume_list:
            stem_list[i] = stem_list[i] * volume_list[i]

    mix = sum(stem_list) / stem_num
    mix = mix / np.abs(mix).max()
    return mix


class AudioGenerate(object):
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.n_channel = self.config['n_channel']
        self.sources = self.config['sources']
        self.samplerate = self.config['samplerate']
        self.separate_config = self.config['separate']
        self.f0_config = self.config['f0']
        self.hubert_config = self.config['hubert']
        self.generate_config = self.config['generate']

        self.separate_model_cfg = self.separate_config['model']
        self.separate_model_cfg['sources'] = self.sources
        self.separate_model_cfg['n_channel'] = self.n_channel
        self.unet = get_model(UNets, self.separate_model_cfg,
                              model_path=self.separate_config['model_path'],
                              is_train=False, device=device)

        self.f0_model_cfg = self.f0_config['model']
        self.f0_extractor = get_model(PitchNet, self.f0_model_cfg,
                                      model_path=self.f0_config['model_path'],
                                      is_train=False, device=device)
        self.hubert_model = get_hubert_model(self.hubert_config['name'],
                                             dl_kwargs=self.hubert_config['download'],
                                             device=device)

        self.generate_model_cfg = self.generate_config['model']
        self.ai_cover_model = get_model(CombineNet, self.generate_model_cfg,
                                        model_path=self.generate_config['model_path'],
                                        device=device)

    def separate(self, waveform, samplerate):
        assert samplerate == self.samplerate
        wav_len = waveform.shape[-1]

        spec_config = self.separate_config['spec']
        n_fft = spec_config['n_fft']
        hop_length = spec_config['hop_length']
        n_time = spec_config['n_time']

        split_len = (n_time - 5) * hop_length + n_fft

        output_waveforms = [[] for _ in range(self.sources)]
        for i in range(0, wav_len, split_len):
            with th.no_grad():
                x = waveform[:, i:i + split_len]
                pad_num = 0
                if x.shape[-1] < split_len:
                    pad_num = split_len - (wav_len - i)
                    x = F.pad(x, (0, pad_num))

                # separator
                z = get_spectrogram(x, spec_config)
                mag_z = th.abs(z).unsqueeze(0)
                masks = self.unet(mag_z)
                masks = masks.squeeze(0)
                _masked_stfts = build_masked_stft(masks, z, n_fft=n_fft)
                # build waveform
                for j, _masked_stft in enumerate(_masked_stfts):
                    _masked_stft = _masked_stft.transpose(-1, -2)
                    _waveform = istft(_masked_stft, hop_length=hop_length)
                    if pad_num > 0:
                        _waveform = _waveform[:, :-pad_num]
                    output_waveforms[j].append(_waveform)

        inst_waveforms = []
        for waveform_list in output_waveforms:
            inst_waveforms.append(th.cat(waveform_list, dim=-1))
        return th.stack(inst_waveforms, dim=0)

    def get_feat(self, audio):
        feats, _ = self.hubert_model(audio)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        return feats

    def generate(self, audio_fp, save_path):
        waveform, samplerate = load_waveform(audio_fp, samplerate=self.samplerate,
                                             n_channel=self.n_channel, device=self.device)

        waveforms = self.separate(waveform, samplerate)

        vocal_waveform = waveforms[0]
        other_waveform = waveforms[1]

        with th.no_grad():
            feat = self.get_feat(vocal_waveform)
            f0 = self.f0_extractor(vocal_waveform)
            out_waveform, _, _ = self.ai_cover_model(vocal_waveform, feat, f0)

        final_waveform = merge_wav([vocal_waveform, other_waveform],
                                   volume_list=self.generate_config['volume_list'])

        fp = os.path.join(save_path, f'{generate_uuid()}.wav')
        save_waveform(fp, final_waveform, self.samplerate)

