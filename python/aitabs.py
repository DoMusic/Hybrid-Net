import torchaudio as ta
import torch as th
import torch.nn.functional as F
from librosa.feature import tempo

from audio import read_wav, write_wav, gen_wav
from utils import build_masked_stft, get_chord_name, get_segment_name, get_lyrics
from spec import istft, get_spec, get_specs, get_mixed_spec
from modulation import search_key
from models import get_model


class AITabTranscription(object):
    def __init__(self, config):
        self.config = config
        self.n_channel = self.config['n_channel']
        self.sources = self.config['sources']
        self.sample_rate = self.config['sample_rate']
        self.sep_config = self.config['separate']
        self.lyrics_cfg = self.config['lyrics']
        self.beat_cfg = self.config['beat']
        self.chord_cfg = self.config['chord']
        self.segment_cfg = self.config['segment']
        self.pitch_cfg = self.config['pitch']
        self.spec_cfg = self.config['spec']
        self.tempo_cfg = self.config['tempo']

    def separate(self, waveform, sample_rate, device='cpu'):
        assert sample_rate == self.sample_rate
        wav_len = waveform.shape[-1]

        model_config = self.sep_config['model']
        spec_config = self.sep_config['spec']
        n_fft = self.sep_config['spec']['n_fft']
        hop_length = self.sep_config['spec']['hop_length']
        n_time = self.sep_config['spec']['n_time']

        _model_cfg = {
            'sources': self.sources,
            'n_channel': self.n_channel,
        }
        _model_cfg.update(model_config)
        unet = get_model(self.sep_config['model_name'], _model_cfg, model_path=self.sep_config['model_path'],
                         is_train=False, device=device)

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
                z = get_spec(x, spec_config)
                mag_z = th.abs(z).unsqueeze(0)
                masks = unet(mag_z)
                masks = masks.squeeze(0)
                _masked_stfts = build_masked_stft(masks, z, n_fft=n_fft)
                # build waveform
                for j, _masked_stft in enumerate(_masked_stfts):
                    _waveform = istft(_masked_stft, n_fft=n_fft, hop_length=hop_length, pad=True)
                    if pad_num > 0:
                        _waveform = _waveform[:, :-pad_num]
                    output_waveforms[j].append(_waveform)

        inst_waveforms = []
        for waveform_list in output_waveforms:
            inst_waveforms.append(th.cat(waveform_list, dim=-1))
        return th.stack(inst_waveforms, dim=0)

    def transcribe(self, wav_fp, device='cpu'):

        waveform, sample_rate = read_wav(wav_fp, sample_rate=self.sample_rate, n_channel=self.n_channel, device=device)
        # print(waveform.shape, sample_rate)

        inst_waveforms = self.separate(waveform, sample_rate)
        # print(inst_waveforms.shape)

        # laod model
        beat_net = get_model(self.beat_cfg['model_name'], self.beat_cfg['model'],
                             model_path=self.beat_cfg['model_path'], is_train=False, device=device)
        chord_net = get_model(self.chord_cfg['model_name'], self.chord_cfg['model'],
                              model_path=self.chord_cfg['model_path'], is_train=False, device=device)
        segment_net = get_model(self.segment_cfg['model_name'], self.segment_cfg['model'],
                                model_path=self.segment_cfg['model_path'], is_train=False, device=device)
        pitch_net = get_model(self.pitch_cfg['model_name'], self.pitch_cfg['model'],
                              model_path=self.pitch_cfg['model_path'], is_train=False, device=device)

        vocal_waveform = inst_waveforms[0].numpy()
        orig_spec = get_spec(waveform, self.spec_cfg)
        inst_specs = get_specs(inst_waveforms, self.spec_cfg)  # vocal, bass, drum, other
        vocal_spec = get_spec(inst_waveforms[0], self.spec_cfg)  # vocal
        other_spec = get_mixed_spec(inst_waveforms[1:], self.spec_cfg)  # bass + drum + other

        # pred lyrics
        lyrics, lyrics_matrix = get_lyrics(vocal_waveform, sample_rate, self.lyrics_cfg)

        with th.no_grad():
            # pred beat
            beat_features = inst_specs[:, :, :, :self.spec_cfg['n_fft'] // 2].unsqueeze(0)  # B, S, C, T, F
            beat_features_mag = th.abs(beat_features)

            beat_pred, beat_logist = beat_net(beat_features_mag)
            print('beat info', beat_pred.shape, beat_logist.shape)

            # pred chord
            chord_features = other_spec[:, :, :self.spec_cfg['n_fft'] // 2].unsqueeze(0)
            chord_features_mag = th.abs(chord_features)

            chord_pred, chord_logist = chord_net(chord_features_mag, beat_logist)
            print('chord info', chord_pred.shape, chord_logist.shape)

            # pred segment
            segment_features = orig_spec[:, :, :self.spec_cfg['n_fft'] // 2].unsqueeze(0)
            segment_features_mag = th.abs(segment_features)
            segment_pred = segment_net(segment_features_mag, chord_logist)
            print('segment info', segment_pred.shape)

            # pred pitch
            pitch_features = vocal_spec[:, :, :self.spec_cfg['n_fft'] // 2].unsqueeze(0)
            pitch_features_mag = th.abs(pitch_features)
            pitch_pred, pitch_logist = pitch_net(pitch_features_mag, lyrics_matrix)
            print('pitch info', pitch_pred.shape, pitch_logist.shape)

        beats = beat_pred.squeeze(0).numpy()
        bpm = tempo(onset_envelope=beats, hop_length=self.tempo_cfg['hop_length']).tolist()
        chord_pred = chord_pred.squeeze(0)
        chords = get_chord_name(chord_pred)
        song_key = search_key(chords)
        segment_pred = segment_pred.squeeze(0)
        segment = get_segment_name(segment_pred)
        beats = beats.tolist()
        pitch_list = pitch_pred.squeeze(0).tolist()

        ret = {
            'bpm': bpm,
            'key': song_key,
            'chords': chords,
            'beat': beats,
            'segment': segment,
            'pitch': pitch_list,
            'lyrics': lyrics,
        }
        return ret, inst_waveforms

