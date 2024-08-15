import torch
import torchaudio
from torch import nn
import torch.nn.functional as F

from utils.spec import stft


class ResConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 momentum=0.01,
                 bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu2 = nn.ReLU()

        if in_channels != out_channels or stride != (1, 1):
            self.residual_connection = nn.Conv2d(in_channels,
                                                 out_channels,
                                                 kernel_size=(1, 1),
                                                 stride=stride,
                                                 padding=(0, 0),
                                                 bias=bias)
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        identity = self.residual_connection(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu2(x)
        return x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.res_conv_layers = nn.ModuleList()
        for i in range(self.n_blocks):
            if i == 0:
                self.res_conv_layers.append(ResConvBlock(in_channels, out_channels, momentum=momentum))
            else:
                self.res_conv_layers.append(ResConvBlock(out_channels, out_channels, momentum=momentum))

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.res_conv_layers[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_layers,
                 pool_size,
                 n_blocks=1,
                 momentum=0.01):
        super().__init__()
        self.n_layers = n_layers

        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, n_blocks=n_blocks, momentum=momentum))
            self.pool_layers.append(nn.AvgPool2d(kernel_size=pool_size))
            in_channels = out_channels
            out_channels *= 2
        self.out_channels = out_channels

    def forward(self, x):
        x = self.bn(x)
        h = []
        for i in range(self.n_layers):
            t = self.layers[i](x)
            x = self.pool_layers[i](t)
            h.append(t)
        return x, h


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, n_blocks, momentum=0.01):
        super().__init__()

        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(ResEncoderBlock(in_channels, out_channels, n_blocks=n_blocks, momentum=momentum))
            else:
                self.layers.append(ResEncoderBlock(out_channels, out_channels, n_blocks=n_blocks, momentum=momentum))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=(0, 1) if stride == (1, 2) else (1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU()
        )
        self.res_conv_layers = nn.ModuleList()
        for i in range(self.n_blocks):
            if i == 0:
                self.res_conv_layers.append(ResConvBlock(out_channels * 2, out_channels, momentum=momentum))
            else:
                self.res_conv_layers.append(ResConvBlock(out_channels, out_channels, momentum=momentum))

    def forward(self, x, h):
        x = self.conv(x)
        x = torch.cat((x, h), dim=1)
        for i in range(self.n_blocks):
            x = self.res_conv_layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 n_layers,
                 stride,
                 n_blocks,
                 momentum=0.01):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                ResDecoderBlock(in_channels, in_channels // 2, stride, n_blocks=n_blocks, momentum=momentum))
            in_channels //= 2

    def forward(self, x, h):
        for i in range(self.n_layers):
            x = self.layers[i](x, h[-i - 1])
        return x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.25):
        super().__init__()
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout)

    def forward(self, x):
        x, _ = self.gru(x)
        return x


class MelSpectrogram(nn.Module):
    def __init__(self,
                 nfft,
                 n_mels,
                 hop_length = None,
                 mel_f_min=30.0,
                 mel_f_max=8000.0,
                 samplerate=44100):
        super().__init__()
        self.samplerate = samplerate
        self.nfft = nfft
        self.n_stft = nfft // 2 + 1
        self.hop_length = nfft // 4 if hop_length is None else hop_length
        self.n_mels = n_mels
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=samplerate, n_stft=self.n_stft, f_min=mel_f_min, f_max=mel_f_max)

    def _stft(self, x):
        hl = self.hop_length
        nfft = self.nfft
        return stft(x, fft_size=nfft, hop_length=hl)

    def _mel(self, stft):
        magnitude = stft.abs().pow(2)
        mel = self.mel_scale(magnitude)
        return mel

    def forward(self, x):
        stft = self._stft(x)
        mel = self._mel(stft)
        return mel


class PitchNet(nn.Module):

    def __init__(self,
                 nfft,
                 mel_f_min=30.0,
                 mel_f_max=8000.0,
                 hop_length=None,
                 samplerate=16000,
                 en_layers=5,
                 re_layers=4,
                 de_layers=5,
                 n_blocks=5,
                 in_channels=1,
                 en_out_channels=16,
                 pool_size=(2, 2),
                 n_gru=1,
                 n_mels=128,
                 n_classes=360):
        super().__init__()

        self.mel_spectrogram = MelSpectrogram(n_mels=n_mels,
                                              samplerate=samplerate,
                                              hop_length=hop_length,
                                              nfft=nfft,
                                              mel_f_min=mel_f_min,
                                              mel_f_max=mel_f_max)
        self.encoder = Encoder(in_channels, en_out_channels, en_layers, n_blocks=n_blocks, pool_size=pool_size)
        self.enhancer = Enhancer(self.encoder.out_channels // 2, self.encoder.out_channels, re_layers,
                                 n_blocks=n_blocks)
        self.decoder = Decoder(self.encoder.out_channels, de_layers, stride=pool_size, n_blocks=n_blocks)

        self.conv = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru > 0:
            self.fc = nn.Sequential(
                BiGRU(3 * n_mels, 256, num_layers=n_gru),
                nn.Linear(256 * 2, n_classes),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * n_mels, n_classes),
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.mel_spectrogram(x)
        n_frames = x.size(-1)
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            x = F.pad(x, (0, n_pad), mode="constant")
        x = x.transpose(-1, -2).unsqueeze(1)
        x, h = self.encoder(x)
        x = self.enhancer(x)
        x = self.decoder(x, h)
        x = self.conv(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        x = x[:, :n_frames]
        return x

