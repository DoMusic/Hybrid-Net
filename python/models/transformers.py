import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(batch_size, n_time, n_feature, zero_pad=False, scale=False, dtype=torch.float32):
    pos_indices = torch.tile(torch.unsqueeze(torch.arange(n_time), 0), [batch_size, 1])

    pos = torch.arange(n_time, dtype=dtype).reshape(-1, 1)
    pos_enc = pos / torch.pow(10000, 2 * torch.arange(0, n_feature, dtype=dtype) / n_feature)
    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2])

    if zero_pad:
        pos_enc = torch.cat([torch.zeros(size=[1, n_feature]), pos_enc[1:, :]], 0)

    outputs = F.embedding(pos_indices, pos_enc)

    if scale:
        outputs = outputs * (n_feature ** 0.5)

    return outputs


class FeedForward(nn.Module):
    def __init__(self, n_feature=2048, n_hidden=512, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(n_feature, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_feature)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_layer = nn.LayerNorm(n_feature)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        y = self.norm_layer(y)
        return y


class EncoderFre(nn.Module):
    def __init__(self, n_freq, n_group, nhead=8, n_layers=5, dropout=0.5, pr=0.01):
        super().__init__()
        assert n_freq % n_group == 0
        self.d_model = d_model = n_freq // n_group
        self.n_freq = n_freq
        self.n_group = n_group
        self.pr = pr
        self.n_layers = n_layers

        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()

        for _ in range(self.n_layers):
            _attn_layer = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            _ff_layer = FeedForward(d_model, dropout=dropout)

            self.attn_layer.append(_attn_layer)
            self.ff_layer.append(_ff_layer)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)
        self.norm_layer = nn.LayerNorm(n_freq)

    def forward(self, x):
        # x: batch, n_time, n_freq
        B, T, Fr = x.shape
        x_reshape = x.reshape(B * T, self.n_group, self.d_model)
        x_reshape += positional_encoding(
            batch_size=x_reshape.shape[0], n_time=x_reshape.shape[1], n_feature=x_reshape.shape[2]
        ) * self.pr

        for _attn_layer, _ff_layer in zip(self.attn_layer, self.ff_layer):
            x_reshape, _ = _attn_layer(x_reshape, x_reshape, x_reshape, need_weights=False)
            x_reshape = _ff_layer(x_reshape)

        y = x_reshape.reshape(B, T, self.n_freq)

        y = self.dropout(y)
        y = self.fc(y)
        y = self.norm_layer(y)
        return y


class EncoderTime(nn.Module):
    def __init__(self, n_freq, nhead=8, n_layers=5, dropout=0.5, pr=0.02):
        super().__init__()
        self.n_freq = n_freq
        self.n_layers = n_layers
        self.pr = pr

        self.attn_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(self.n_layers):
            _attn_layer = nn.MultiheadAttention(n_freq, nhead, batch_first=True)
            _ff_layer = FeedForward(n_freq, dropout=dropout)

            self.attn_layer.append(_attn_layer)
            self.ff_layer.append(_ff_layer)

        self.norm_layer = nn.LayerNorm(n_freq)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_freq, n_freq)

    def forward(self, x):
        # x: batch, n_time, n_freq
        B, T, Fr = x.shape
        x += positional_encoding(B, T, Fr) * self.pr

        for _attn_layer, _ff_layer in zip(self.attn_layer, self.ff_layer):
            x, _ = _attn_layer(x, x, x, need_weights=False)
            x = _ff_layer(x)

        x = self.dropout(x)
        x = self.fc(x)
        x = self.norm_layer(x)
        return x


class Decoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, n_layers=5, dropout=0.5, r1=1.0, r2=1.0, wr=1.0, pr=0.01):
        super().__init__()

        self.r1 = r1
        self.r2 = r2
        self.wr = wr
        self.n_layers = n_layers
        self.pr = pr

        self.attn1_layer = nn.ModuleList()
        self.attn2_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(n_layers):
            _layer1 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            _layer2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            _layer3 = FeedForward(d_model, dropout=dropout)
            self.attn1_layer.append(_layer1)
            self.attn2_layer.append(_layer2)
            self.ff_layer.append(_layer3)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x1, x2, weight=None):
        y = x1 * self.r1 + x2 * self.r2
        if weight is not None:
            y += weight * self.wr

        y += positional_encoding(y.shape[0], y.shape[1], y.shape[2]) * self.pr + self.pr

        for i in range(self.n_layers):
            _attn1_layer = self.attn1_layer[i]
            _attn2_layer = self.attn2_layer[i]
            _ff_layer = self.ff_layer[i]
            y, _ = _attn1_layer(y, y, y, need_weights=False)
            y, _ = _attn2_layer(y, x2, x2, need_weights=False)
            y = _ff_layer(y)
        output = self.dropout(y)
        output = self.fc(output)
        return output, y


class BaseTransformer(nn.Module):
    def __init__(self,
                 n_channel=2,
                 n_freq=2048,
                 # EncoderFre
                 n_group=32,
                 f_layers=2,
                 f_nhead=4,
                 f_dropout=0.5,
                 f_pr=0.01,
                 # EncoderTime
                 t_layers=2,
                 t_nhead=4,
                 t_dropout=0.5,
                 t_pr=0.01,
                 # Decoder
                 d_layers=2,
                 d_nhead=4,
                 d_dropout=0.5,
                 d_pr=0.02,
                 r1=1.0,
                 r2=1.0,
                 wr=0.2,
                 ):
        super().__init__()
        self.n_channel = n_channel
        self.encoder_fre_layers = nn.ModuleList()
        self.encoder_time_layers = nn.ModuleList()
        for _ in range(n_channel):
            _encoder_fre_layer = EncoderFre(n_freq=n_freq, n_group=n_group, nhead=f_nhead, n_layers=f_layers,
                                            dropout=f_dropout, pr=f_pr)
            _encoder_time_layer = EncoderTime(n_freq=n_freq, nhead=t_nhead, n_layers=t_layers, dropout=t_dropout,
                                              pr=t_pr)
            self.encoder_fre_layers.append(_encoder_fre_layer)
            self.encoder_time_layers.append(_encoder_time_layer)

        self.decoder = Decoder(d_model=n_freq, nhead=d_nhead, n_layers=d_layers, dropout=d_dropout,
                               r1=r1, r2=r2, wr=wr, pr=d_pr)

    def forward(self, x, weight=None):
        # x: batch, channel, n_time, n_freq
        ff_list = []
        tf_list = []
        for i in range(self.n_channel):
            x1 = self.encoder_fre_layers[i](x[:, i, :, :])
            x2 = self.encoder_time_layers[i](x[:, i, :, :])
            ff_list.append(x1)
            tf_list.append(x2)

        y1 = torch.sum(torch.stack(ff_list, dim=0), dim=0)
        y2 = torch.sum(torch.stack(tf_list, dim=0), dim=0)
        y, w = self.decoder(y1, y2, weight=weight)
        return y, w


if __name__ == '__main__':
    net = BaseTransformer(n_freq=2048)
    # net = EncoderTime(2048)
    # net = EncoderFre(2048, 32)

    # net = EncoderFre(2048, 8)
    x = torch.randn(1, 2, 1024, 2048)

    y, logits = net(x)
    y1, logits = net(x)

    # print(np.allclose(x, y))
    print(y.shape, logits.shape)
    print(y[0, 0, 0], y[0, 1, 0])
    print(y1[0, 0, 0], y1[0, 1, 0])
