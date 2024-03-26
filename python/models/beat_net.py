import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class BeatNet(nn.Module):
    def __init__(self,
                 source=3,
                 n_classes=3,
                 weights=(0.4, 0.3, 0.3),
                 n_freq=2048,
                 n_group=32,
                 f_layers=2,
                 f_nhead=4,
                 t_layers=2,
                 t_nhead=4,
                 d_layers=2,
                 d_nhead=8,
                 dropout=0.5,
                 *args, **kwargs
                 ):
        super().__init__()
        self.weights = weights
        self.transformer_layers = nn.ModuleList()
        for _ in range(source):
            _layer = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                     f_layers=f_layers, f_nhead=f_nhead, f_dropout=dropout,
                                     t_layers=t_layers, t_nhead=t_nhead, t_dropout=dropout,
                                     d_layers=d_layers, d_nhead=d_nhead, d_dropout=dropout)
            self.transformer_layers.append(_layer)
        self.dropout = nn.Dropout(dropout)
        self.beat_fc = nn.Linear(n_freq, n_classes)

        self.reset_parameters(0.05)

    def reset_parameters(self, confidence):
        self.beat_fc.bias.data.fill_(-torch.log(torch.tensor(1 / confidence - 1)))

    def forward(self, inp):
        # shape: (batch, source, channel, time, freq)

        y_list = []
        logits_weight_list = []
        for i, layer in enumerate(self.transformer_layers):
            x = inp[:, i, :, :, :]
            x, _f = layer(x)
            w = self.weights[i]
            x = x * w
            y_list.append(x)
            logits_weight_list.append(_f * w)
        y = torch.sum(torch.stack(y_list, dim=0), dim=0)
        logits_weight = torch.sum(torch.stack(logits_weight_list, dim=0), dim=0)

        y = self.dropout(y)
        beats = self.beat_fc(y)
        beats = torch.argmax(beats, dim=-1)
        return beats, logits_weight


if __name__ == '__main__':
    model = BeatNet()
    print(model)
    x = torch.randn(6, 3, 2, 256, 2048)
    b, weight = model(x)
    print(x.shape, b.shape, weight.shape)
