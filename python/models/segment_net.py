import torch
import torch.nn as nn
from models.transformers import BaseTransformer


class SegmentEmbeddings(nn.Module):
    def __init__(self, n_channel=2, n_hidden=128, d_model=2048, dropout=0.5):
        super().__init__()
        self.n_channel = n_channel
        self.n_hidden = n_hidden

        self.act_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channel, n_hidden // 2, kernel_size=(1, 1))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_hidden // 2, n_hidden, kernel_size=(1, 1))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(n_hidden, n_hidden // 2, kernel_size=(1, 1))
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(n_hidden // 2, n_channel, kernel_size=(1, 1))
        self.drop4 = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model // 4)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: batch, n_channel, n_time, n_freq
        x = self.conv1(x)
        x = self.pool0(x)
        x = self.act_fn(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act_fn(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.act_fn(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.act_fn(x)
        x = self.drop4(x)

        x = self.norm(x)
        x = self.drop(x)

        return x


class SegmentNet(nn.Module):
    def __init__(self,
                 n_freq=2048,
                 n_channel=2,
                 n_classes=10,
                 emb_hidden=128,
                 n_group=32,
                 f_layers=2,
                 f_nhead=8,
                 t_layers=2,
                 t_nhead=8,
                 d_layers=2,
                 d_nhead=8,
                 dropout=0.5,
                 *args, **kwargs):
        super().__init__()

        divisor = 4
        d_model = n_freq // divisor
        self.embeddings = SegmentEmbeddings(n_channel=n_channel, n_hidden=emb_hidden, d_model=n_freq, dropout=dropout)
        self.wfc = nn.Linear(n_freq, n_freq // divisor)

        self.transformer = BaseTransformer(n_channel=n_channel, n_freq=d_model, n_group=n_group,
                                           f_layers=f_layers, f_nhead=f_nhead, f_dropout=dropout,
                                           t_layers=t_layers, t_nhead=t_nhead, t_dropout=dropout,
                                           d_layers=d_layers, d_nhead=d_nhead, d_dropout=dropout,
                                           )

        self.norm = nn.LayerNorm(d_model)

        self.segment_classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, weight=None):
        # x: batch, n_channel, n_time, n_freq
        if weight is None:
            B, C, T, F = x.shape
            weight = torch.ones(B, T, F, device=x.device)
        x = self.embeddings(x)
        weight = self.wfc(weight)

        x, _ = self.transformer(x, weight=weight)
        x = self.norm(x)
        x = self.segment_classifier(x)
        x = torch.argmax(x, dim=-1)
        return x


if __name__ == '__main__':
    net = SegmentNet(n_freq=2048)
    print(net)
    x = torch.randn(2, 2, 1024, 2048)
    y = net(x)
    print(y.shape)
