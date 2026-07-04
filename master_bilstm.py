import math

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from base_model import SequenceModel
from master import Gate, PositionalEncoding, SAttention, TemporalAttention


class BiLSTMTemporal(nn.Module):
    def __init__(self, d_model, dropout, ffn_expand=4, num_layers=1):
        super().__init__()
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        hidden_size = math.ceil(d_model / 2)
        self.temporal = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = Linear(hidden_size * 2, d_model)
        self.dropout = Dropout(p=dropout)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        if ffn_expand <= 1:
            self.ffn = nn.Sequential(
                Linear(d_model, d_model),
                nn.ReLU(),
                Dropout(p=dropout),
                Linear(d_model, d_model),
                Dropout(p=dropout),
            )
        else:
            ffn_hidden = d_model * ffn_expand
            self.ffn = nn.Sequential(
                Linear(d_model, ffn_hidden),
                nn.GELU(),
                Dropout(p=dropout),
                Linear(ffn_hidden, d_model),
                Dropout(p=dropout),
            )

    def forward(self, x):
        xt = self.norm1(x)
        lstm_out, _ = self.temporal(xt)
        lstm_out = self.dropout(self.proj(lstm_out))
        xt = x + lstm_out
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class MASTERBiLSTM(nn.Module):
    def __init__(
        self,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        T_dropout_rate,
        S_dropout_rate,
        gate_input_start_index,
        gate_input_end_index,
        beta,
        ffn_expand=4,
        bilstm_layers=1,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            BiLSTMTemporal(
                d_model=d_model,
                dropout=T_dropout_rate,
                ffn_expand=ffn_expand,
                num_layers=bilstm_layers,
            ),
            SAttention(
                d_model=d_model,
                nhead=s_nhead,
                dropout=S_dropout_rate,
                ffn_expand=ffn_expand,
            ),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        return self.layers(src).squeeze(-1)


class MASTERBiLSTMModel(SequenceModel):
    def __init__(
        self,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        gate_input_start_index,
        gate_input_end_index,
        T_dropout_rate,
        S_dropout_rate,
        beta,
        ffn_expand=4,
        bilstm_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat
        self.ffn_expand = ffn_expand
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.bilstm_layers = bilstm_layers
        self.init_model()

    def init_model(self):
        self.model = MASTERBiLSTM(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            ffn_expand=self.ffn_expand,
            bilstm_layers=self.bilstm_layers,
        )
        super().init_model()
