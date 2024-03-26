import torch
import torch.nn as nn
from models.transformers import EncoderFre, EncoderTime, FeedForward, positional_encoding


class PitchEmbedding(nn.Module):
    def __init__(self, n_channel, d_model, n_hidden=32, dropout=0.5):
        super().__init__()
        self.n_channel = n_channel
        self.n_hidden = n_hidden

        self.act_fn = nn.ReLU()

        self.conv1 = nn.Conv2d(n_channel, n_hidden // 4, kernel_size=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(n_hidden // 4, n_hidden // 2, kernel_size=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(n_hidden // 2, n_hidden, kernel_size=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop4 = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model // 16)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: batch, n_channel, n_time, n_freq
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act_fn(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act_fn(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.act_fn(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.act_fn(x)
        x = self.drop4(x)

        x = self.norm(x)
        x = self.drop(x)

        return x


class PitchEncoder(nn.Module):
    def __init__(self,
                 n_freq=2048,
                 n_group=32,
                 weights=(0.6, 0.4),
                 f_layers=2,
                 f_nhead=8,
                 f_pr=0.01,
                 t_layers=2,
                 t_nhead=8,
                 t_pr=0.01,
                 dropout=0.5):
        super().__init__()
        self.weights = weights

        self.encoder_fre = EncoderFre(n_freq=n_freq, n_group=n_group, nhead=f_nhead, n_layers=f_layers, dropout=dropout,
                                      pr=f_pr)
        self.encoder_time = EncoderTime(n_freq=n_freq, nhead=t_nhead, n_layers=t_layers, dropout=dropout, pr=t_pr)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_freq)

    def forward(self, x):
        # x: batch, channel, n_time, n_freq
        y1 = self.encoder_fre(x)
        y2 = self.encoder_time(x)
        y = y1 * self.weights[0] + y2 * self.weights[1]
        y = self.dropout(y)
        y = self.norm(y)
        return y


class PitchNet(nn.Module):
    def __init__(self,
                 n_freq=2048,
                 n_channel=2,
                 n_classes=850,  # 85 * 10
                 emb_hidden=16,
                 wr=0.3,
                 pr=0.02,
                 n_group=32,
                 f_layers=2,
                 f_nhead=8,
                 t_layers=2,
                 t_nhead=8,
                 enc_weights=(0.6, 0.4),
                 d_layers=2,
                 d_nhead=8,
                 dropout=0.5,
                 *args, **kwargs):
        super().__init__()
        self.wr = wr
        self.pr = pr
        self.embedding = PitchEmbedding(n_channel, n_freq, emb_hidden, dropout)
        d_model = n_freq // 16 * emb_hidden
        self.encoder = PitchEncoder(n_freq=d_model, n_group=n_group,
                                    weights=enc_weights,
                                    f_layers=f_layers, f_nhead=f_nhead,
                                    t_layers=t_layers, t_nhead=t_nhead,
                                    dropout=dropout)

        self.attn1_layer = nn.ModuleList()
        self.attn2_layer = nn.ModuleList()
        self.ff_layer = nn.ModuleList()
        for _ in range(d_layers):
            _layer1 = nn.MultiheadAttention(d_model, d_nhead, batch_first=True)
            _layer2 = nn.MultiheadAttention(d_model, d_nhead, batch_first=True)
            _layer3 = FeedForward(d_model, dropout=dropout)
            self.attn1_layer.append(_layer1)
            self.attn2_layer.append(_layer2)
            self.ff_layer.append(_layer3)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, weight=None):
        # x: batch, channel, n_time, n_freq
        y = self.embedding(x)

        B, C, T, F = y.shape
        y = y.permute(0, 2, 1, 3)  # B, C, T, F => B, T, C, F
        y = y.reshape(B, T, C * F)  # B, T, C * F
        y = self.encoder(y)

        if weight is None:
            weight = torch.zeros_like(y)
        y_w = y + weight * self.wr
        B, T, F = y_w.shape
        y_w += positional_encoding(B, T, F) * self.pr

        for _attn1_layer, _attn2_layer, _ff_layer in zip(self.attn1_layer, self.attn2_layer, self.ff_layer):
            y, _ = _attn1_layer(y, y, y, need_weights=False)
            y, _ = _attn1_layer(y, y_w, y_w, need_weights=False)
            y = _ff_layer(y)

        output = self.dropout(y)
        output = self.classifier(output)
        output = torch.argmax(output, dim=-1)
        return output, y


if __name__ == '__main__':
    model = PitchNet()
    print(model)
    x = torch.randn(6, 2, 256, 2048)
    y1, y2 = model(x, None)
    print(y1.shape, y2.shape)
