import torch
import torch.nn.functional as F

from torch import nn


class UPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        return F.pad(x, self.padding_setting, "constant", 0)


class UTransposedPad(nn.Module):
    def __init__(self, padding_setting=(1, 2, 1, 2)):
        super().__init__()
        self.padding_setting = padding_setting

    def forward(self, x):
        l, r, t, b = self.padding_setting
        return x[:, :, l:-r, t:-b]


def get_activation(activation):
    if activation == "ReLU":
        activation_fn = nn.ReLU()
    elif activation == "ELU":
        activation_fn = nn.ELU()
    else:
        activation_fn = nn.LeakyReLU(0.2)
    return activation_fn


class UNet(nn.Module):
    def __init__(self,
                 n_channel=2,
                 conv_n_filters=(16, 32, 64, 128, 256, 512),
                 down_activation="ELU",
                 up_activation="ELU",
                 down_dropouts=None,
                 up_dropouts=None):
        super().__init__()

        conv_num = len(conv_n_filters)

        down_activation_fn = get_activation(down_activation)
        up_activation_fn = get_activation(up_activation)

        down_dropouts = [0] * conv_num if down_dropouts is None else down_dropouts
        up_dropouts = [0] * conv_num if up_dropouts is None else up_dropouts

        self.down_layers = nn.ModuleList()
        for i in range(conv_num):
            in_ch = n_channel if i == 0 else conv_n_filters[i - 1]
            out_ch = conv_n_filters[i]
            dropout = down_dropouts[i]

            _down_layers = [
                UPad(),
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(out_ch, track_running_stats=True, eps=1e-3, momentum=0.01),
                down_activation_fn
            ]
            if dropout > 0:
                _down_layers.append(nn.Dropout(dropout))
            self.down_layers.append(nn.Sequential(*_down_layers))

        self.up_layers = nn.ModuleList()
        for i in range(conv_num - 1, -1, -1):
            in_ch = conv_n_filters[conv_num - 1] if i == conv_num - 1 else conv_n_filters[i + 1]
            out_ch = 1 if i == 0 else conv_n_filters[i - 1]
            dropout = up_dropouts[i]

            _up_layer = [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=5, stride=2, padding=0),
                UTransposedPad(),
                up_activation_fn,
                nn.BatchNorm2d(out_ch, track_running_stats=True, eps=1e-3, momentum=0.01)
            ]
            if dropout > 0:
                _up_layer.append(nn.Dropout(dropout))
            self.up_layers.append(nn.Sequential(*_up_layer))

        self.last_layer = nn.Conv2d(1, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        d_convs = []
        for layer in self.down_layers:
            x = layer(x)
            d_convs.append(x)

        n = len(self.up_layers)
        for i, layer in enumerate(self.up_layers):
            if i == 0:
                x = layer(x)
            else:
                x1 = d_convs[n - i - 1]
                x = torch.cat([x, x1], axis=1)
                x = layer(x)
        x = self.last_layer(x)
        return x


class UNets(nn.Module):
    def __init__(self,
                 sources,
                 n_channel=2,
                 conv_n_filters=(16, 32, 64, 128, 256, 512),
                 down_activation="ELU",
                 up_activation="ELU",
                 down_dropouts=None,
                 up_dropouts=None,
                 *args, **kwargs):
        super().__init__()

        self.unet_layers = nn.ModuleList()
        for i in range(sources):
            layer = UNet(n_channel=n_channel,
                         conv_n_filters=conv_n_filters,
                         down_activation=down_activation,
                         up_activation=up_activation,
                         down_dropouts=down_dropouts,
                         up_dropouts=up_dropouts)
            self.unet_layers.append(layer)

    def forward(self, x):
        # x shape: (batch, channel, time, freq)
        _layers = []
        for layer in self.unet_layers:
            y = layer(x)
            _layers.append(y)

        y = torch.stack(_layers, axis=1)
        y = F.softmax(y, dim=1)  # shape: (batch, sources, channel, time, freq)
        return y


if __name__ == '__main__':
    model_params = {
        'sources': 4,
        'n_channel': 2,
        'conv_n_filters': [8, 16, 32, 64, 128, 256, 512, 1024],
        'down_activation': "ELU",
        'up_activation': "ELU",
        # 'down_dropouts': [0, 0, 0, 0, 0, 0, 0, 0],
        # 'up_dropouts': [0, 0, 0, 0, 0, 0, 0, 0],
    }
    net = UNets(**model_params)
    print(net)
    print(net(torch.rand(1, 2, 256, 2049)).shape)
