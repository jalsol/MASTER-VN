import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, ffn_expand=4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)

        if ffn_expand <= 1:
            self.ffn = nn.Sequential(
                Linear(d_model, d_model),
                nn.ReLU(),
                Dropout(p=dropout),
                Linear(d_model, d_model),
                Dropout(p=dropout)
            )
            print('original, no ffn expand')
        else:
            ffn_hidden = d_model * ffn_expand
            self.ffn = nn.Sequential(
                Linear(d_model, ffn_hidden),
                nn.GELU(),
                Dropout(p=dropout),
                Linear(ffn_hidden, d_model),
                Dropout(p=dropout)
            )
            print('ffn expand', ffn_expand)

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, ffn_expand=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        if ffn_expand <= 1:
            self.ffn = nn.Sequential(
                Linear(d_model, d_model),
                nn.ReLU(),
                Dropout(p=dropout),
                Linear(d_model, d_model),
                Dropout(p=dropout)
            )
            print('original, no ffn expand')
        else:
            ffn_hidden = d_model * ffn_expand
            self.ffn = nn.Sequential(
                Linear(d_model, ffn_hidden),
                nn.GELU(),
                Dropout(p=dropout),
                Linear(ffn_hidden, d_model),
                Dropout(p=dropout)
            )
            print('ffn expand', ffn_expand)

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i==self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output,  beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output =d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output/self.t, dim=-1)
        return self.d_output*output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
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
    ):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index) # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate, ffn_expand=ffn_expand),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate, ffn_expand=ffn_expand),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index] # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
       
        output = self.layers(src).squeeze(-1)

        return output


class MASTERSwap(nn.Module):
    """
    Swapped attention order (spatial first, then temporal). Works better for CSI universes.
    """

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
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            SAttention(
                d_model=d_model,
                nhead=s_nhead,
                dropout=S_dropout_rate,
                ffn_expand=ffn_expand,
            ),
            TAttention(
                d_model=d_model,
                nhead=t_nhead,
                dropout=T_dropout_rate,
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


class MASTERInterOnly(nn.Module):
    """
    Only inter-stock aggregation; skips temporal attention block.
    """

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
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
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


class MASTERMultiHead(nn.Module):
    """
    Variant that keeps separate temporal/spatial attention paths and fuses them
    with cross-attention so temporal context can reweight spatial cues.
    """

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
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.feature_proj = nn.Linear(d_feat, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.temporal_encoder = TAttention(
            d_model=d_model,
            nhead=t_nhead,
            dropout=T_dropout_rate,
            ffn_expand=ffn_expand,
        )
        self.spatial_encoder = SAttention(
            d_model=d_model,
            nhead=s_nhead,
            dropout=S_dropout_rate,
            ffn_expand=ffn_expand,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=s_nhead,
            dropout=S_dropout_rate,
            batch_first=False,
        )
        self.cross_gate = nn.Parameter(torch.zeros(1))
        self.cross_norm = LayerNorm(d_model, eps=1e-5)
        if ffn_expand <= 1:
            self.cross_ffn = nn.Sequential(
                Linear(d_model, d_model),
                nn.ReLU(),
                Dropout(p=S_dropout_rate),
                Linear(d_model, d_model),
                Dropout(p=S_dropout_rate),
            )
        else:
            cross_hidden = d_model * ffn_expand
            self.cross_ffn = nn.Sequential(
                Linear(d_model, cross_hidden),
                nn.GELU(),
                Dropout(p=S_dropout_rate),
                Linear(cross_hidden, d_model),
                Dropout(p=S_dropout_rate),
            )

        self.temporal_pool = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        src = self.feature_proj(src)
        src = self.pos_encoding(src)

        temporal_ctx = self.temporal_encoder(src)
        spatial_ctx = self.spatial_encoder(src)

        q = temporal_ctx.transpose(0, 1)
        k = spatial_ctx.transpose(0, 1)
        v = spatial_ctx.transpose(0, 1)
        fused_ctx, _ = self.cross_attn(q, k, v)
        fused_ctx = fused_ctx.transpose(0, 1)

        gate = torch.sigmoid(self.cross_gate)
        fusion = temporal_ctx + gate * fused_ctx
        fusion = self.cross_norm(fusion)
        fusion = fusion + self.cross_ffn(fusion)

        pooled = self.temporal_pool(fusion)
        output = self.decoder(pooled).squeeze(-1)
        return output


class MASTEREnsemble(nn.Module):
    """
    Ensemble over several MASTER variants with a learned fusion gate.
    """

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
        expert_types=None,
    ):
        super().__init__()
        if expert_types is None:
            expert_types = ["base", "swap", "inter"]

        self.expert_types = expert_types
        self.num_experts = len(expert_types)
        self.gate_input_start_index = gate_input_start_index

        self.experts = nn.ModuleList(
            [self._build_expert(t, d_feat, d_model, t_nhead, s_nhead,
                                T_dropout_rate, S_dropout_rate,
                                gate_input_start_index, gate_input_end_index,
                                beta, ffn_expand) for t in expert_types]
        )

        fusion_hidden = max(32, d_model // 2)
        self.fusion = nn.Sequential(
            nn.Linear(gate_input_start_index, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, self.num_experts),
        )

    def _build_expert(
        self,
        expert_type,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        T_dropout_rate,
        S_dropout_rate,
        gate_input_start_index,
        gate_input_end_index,
        beta,
        ffn_expand,
    ):
        if expert_type == "base":
            return MASTER(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=T_dropout_rate,
                S_dropout_rate=S_dropout_rate,
                gate_input_start_index=gate_input_start_index,
                gate_input_end_index=gate_input_end_index,
                beta=beta,
                ffn_expand=ffn_expand,
            )
        if expert_type == "swap":
            return MASTERSwap(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=T_dropout_rate,
                S_dropout_rate=S_dropout_rate,
                gate_input_start_index=gate_input_start_index,
                gate_input_end_index=gate_input_end_index,
                beta=beta,
                ffn_expand=ffn_expand,
            )
        if expert_type == "inter":
            return MASTERInterOnly(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=T_dropout_rate,
                S_dropout_rate=S_dropout_rate,
                gate_input_start_index=gate_input_start_index,
                gate_input_end_index=gate_input_end_index,
                beta=beta,
                ffn_expand=ffn_expand,
            )
        if expert_type == "multihead":
            return MASTERMultiHead(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=T_dropout_rate,
                S_dropout_rate=S_dropout_rate,
                gate_input_start_index=gate_input_start_index,
                gate_input_end_index=gate_input_end_index,
                beta=beta,
                ffn_expand=ffn_expand,
            )
        if expert_type == "base_wide":
            expand = max(4, ffn_expand)
            return MASTER(
                d_feat=d_feat,
                d_model=d_model,
                t_nhead=t_nhead,
                s_nhead=s_nhead,
                T_dropout_rate=T_dropout_rate,
                S_dropout_rate=S_dropout_rate,
                gate_input_start_index=gate_input_start_index,
                gate_input_end_index=gate_input_end_index,
                beta=beta,
                ffn_expand=expand,
            )
        raise ValueError(f"Unsupported expert type: {expert_type}")

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [N, num_experts]

        fusion_input = x[:, -1, :self.gate_input_start_index]
        weights = torch.softmax(self.fusion(fusion_input), dim=-1)
        output = torch.sum(expert_outputs * weights, dim=-1)
        return output


class MASTERModel(SequenceModel):
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
        use_multihead=False,
        architecture="base",
        ensemble_expert_types=None,
        **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
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
        self.use_multihead = use_multihead  # backwards compatibility
        self.architecture = architecture
        self.ensemble_expert_types = ensemble_expert_types

        self.init_model()

    def init_model(self):
        arch = self.architecture.lower() if isinstance(self.architecture, str) else "base"
        if arch == "base":
            ModelClass = MASTERMultiHead if self.use_multihead else MASTER
        elif arch == "multihead":
            ModelClass = MASTERMultiHead
        elif arch == "swap":
            ModelClass = MASTERSwap
        elif arch == "inter":
            ModelClass = MASTERInterOnly
        elif arch == "ensemble":
            def ModelClass(**kwargs):
                return MASTEREnsemble(
                    expert_types=self.ensemble_expert_types,
                    **kwargs,
                )
        else:
            raise ValueError(f"Unknown MASTER architecture '{self.architecture}'")

        self.model = ModelClass(
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
        )
        super(MASTERModel, self).init_model()
