import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm

from base_model import SequenceModel
from master import PositionalEncoding, SAttention, TAttention, TemporalAttention


class TemporalCrossAttentionGate(nn.Module):
    """
    Replaces the single-timestep Gate with a cross-attention mechanism that attends
    over all T market-feature timesteps to produce the feature importance weights.

    A learned query summarises what temporal market context is most relevant, then
    projects that context to a softmax-normalised importance vector over d_feat.
    """

    def __init__(self, d_market, d_feat, beta=1.0, n_heads=2, dropout=0.0):
        super().__init__()
        self.d_feat = d_feat
        self.beta = beta
        d_attn = max(n_heads, (d_market // n_heads) * n_heads)
        self.kv_proj = nn.Linear(d_market, d_attn) if d_attn != d_market else nn.Identity()
        self.query = nn.Parameter(torch.zeros(1, 1, d_attn))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = LayerNorm(d_attn, eps=1e-5)
        self.proj = nn.Linear(d_attn, d_feat)

    def forward(self, gate_seq):
        N = gate_seq.shape[0]
        kv = self.kv_proj(gate_seq)
        q = self.query.expand(N, -1, -1)
        ctx, _ = self.attn(q, kv, kv)
        ctx = self.norm(ctx.squeeze(1))
        out = self.proj(ctx)
        out = torch.softmax(out / self.beta, dim=-1)
        return self.d_feat * out


class MASTERCrossAttnGate(nn.Module):
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
        gate_n_heads=2,
        gate_dropout=0.0,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = TemporalCrossAttentionGate(
            d_market=self.d_gate_input,
            d_feat=d_feat,
            beta=beta,
            n_heads=gate_n_heads,
            dropout=gate_dropout,
        )
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate, ffn_expand=ffn_expand),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate, ffn_expand=ffn_expand),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_seq = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        gate_weight = self.feature_gate(gate_seq)
        src = src * gate_weight.unsqueeze(1)
        return self.layers(src).squeeze(-1)


class MASTERCrossAttnGateModel(SequenceModel):
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
        gate_n_heads=2,
        gate_dropout=0.0,
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
        self.gate_n_heads = gate_n_heads
        self.gate_dropout = gate_dropout
        self.init_model()

    def init_model(self):
        self.model = MASTERCrossAttnGate(
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
            gate_n_heads=self.gate_n_heads,
            gate_dropout=self.gate_dropout,
        )
        super().init_model()
