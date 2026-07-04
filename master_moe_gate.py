import torch
from torch import nn

from base_model import SequenceModel
from master import Gate, PositionalEncoding, SAttention, TAttention, TemporalAttention


class RegimeMoEFeatureGate(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        beta=1.0,
        n_experts=3,
        top_k=1,
        router_noise_std=0.02,
        load_balance_coef=0.001,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = max(1, min(top_k, n_experts))
        self.router_noise_std = router_noise_std
        self.load_balance_coef = load_balance_coef
        self.experts = nn.ModuleList([Gate(d_input, d_output, beta=beta) for _ in range(n_experts)])
        self.register_buffer("expert_masks", self._build_expert_masks(d_input=d_input, n_experts=n_experts))
        hidden = max(16, d_input)
        self.router = nn.Sequential(
            nn.Linear(d_input * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )
        self._aux_loss = None

    def _build_expert_masks(self, d_input, n_experts):
        masks = torch.zeros(n_experts, d_input)
        if n_experts != 3 or d_input < 21:
            return torch.ones(n_experts, d_input)
        group_size = d_input // 3
        for g in range(3):
            base = g * group_size
            idx_spec = [0, 1, 2, 5, 6]
            idx_fund = [13, 14, 17, 18]
            idx_liq = [3, 4, 7, 8, 9, 10, 11, 12, 15, 16, 19, 20]
            for i in idx_spec:
                if base + i < d_input:
                    masks[0, base + i] = 1.0
            for i in idx_fund:
                if base + i < d_input:
                    masks[1, base + i] = 1.0
            for i in idx_liq:
                if base + i < d_input:
                    masks[2, base + i] = 1.0
        for i in range(n_experts):
            if torch.sum(masks[i]) <= 0:
                masks[i] = 1.0
        return masks

    def forward(self, gate_sequence):
        gate_last = gate_sequence[:, -1, :]
        gate_mean = torch.mean(gate_sequence, dim=1)
        gate_std = torch.std(gate_sequence, dim=1, unbiased=False)
        router_input = torch.cat([gate_last, gate_mean, gate_std], dim=-1)
        router_logits = self.router(router_input)
        if self.training and self.router_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_noise_std
        routing_probs = torch.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(routing_probs, k=self.top_k, dim=-1)
        routing_weights = torch.zeros_like(routing_probs).scatter(-1, topk_idx, topk_vals)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_outputs = torch.stack(
            [expert(gate_last * self.expert_masks[i].unsqueeze(0)) for i, expert in enumerate(self.experts)],
            dim=-1,
        )
        output = torch.sum(expert_outputs * routing_weights.unsqueeze(1), dim=-1)
        importance = routing_probs.mean(dim=0)
        usage = routing_weights.mean(dim=0)
        uniform = torch.full_like(importance, 1.0 / self.n_experts)
        balance_loss = torch.mean((importance - uniform) ** 2) + torch.mean((usage - uniform) ** 2)
        self._aux_loss = balance_loss * self.load_balance_coef
        return output

    def get_aux_loss(self, device):
        if self._aux_loss is None:
            return torch.tensor(0.0, device=device)
        return self._aux_loss


class MASTERMoEGate(nn.Module):
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
        moe_top_k=1,
        moe_router_noise_std=0.02,
        moe_load_balance_coef=0.001,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = RegimeMoEFeatureGate(
            self.d_gate_input,
            d_feat,
            beta=beta,
            n_experts=3,
            top_k=moe_top_k,
            router_noise_std=moe_router_noise_std,
            load_balance_coef=moe_load_balance_coef,
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
        gate_input = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        output = self.layers(src).squeeze(-1)
        return output

    def get_aux_loss(self):
        device = next(self.parameters()).device
        return self.feature_gate.get_aux_loss(device=device)


class MASTERMoEGateModel(SequenceModel):
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
        moe_top_k=1,
        moe_router_noise_std=0.02,
        moe_load_balance_coef=0.001,
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
        self.moe_top_k = moe_top_k
        self.moe_router_noise_std = moe_router_noise_std
        self.moe_load_balance_coef = moe_load_balance_coef
        self.init_model()

    def init_model(self):
        self.model = MASTERMoEGate(
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
            moe_top_k=self.moe_top_k,
            moe_router_noise_std=self.moe_router_noise_std,
            moe_load_balance_coef=self.moe_load_balance_coef,
        )
        super().init_model()
