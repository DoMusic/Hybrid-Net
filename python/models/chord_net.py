import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class ChordNet(nn.Module):
    def __init__(self,
                 n_freq=2048,
                 n_classes=122,
                 n_group=32,
                 f_layers=5,
                 f_nhead=8,
                 t_layers=5,
                 t_nhead=8,
                 d_layers=5,
                 d_nhead=8,
                 dropout=0.5,
                 *args, **kwargs):
        super().__init__()

        self.transformers = BaseTransformer(n_freq=n_freq, n_group=n_group,
                                            f_layers=f_layers, f_nhead=f_nhead, f_dropout=dropout,
                                            t_layers=t_layers, t_nhead=t_nhead, t_dropout=dropout,
                                            d_layers=d_layers, d_nhead=d_nhead, d_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_classes)

    def forward(self, x, weight=None):
        # x shape: (batch, channel, time, freq)
        output, weight_logits = self.transformers(x, weight)
        output = self.dropout(output)
        output = self.fc(output)
        output = output.argmax(dim=-1)
        return output, weight_logits


if __name__ == '__main__':
    model = ChordNet()
    print(model)
    x = torch.randn(6, 2, 256, 2048)
    y, weight = model(x)
    print(y.shape, weight.shape)
