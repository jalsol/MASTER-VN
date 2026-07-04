import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import nn

from base_model import SequenceModel
from master import PositionalEncoding, SAttention, TAttention, TemporalAttention
from master_moe_gate import RegimeMoEFeatureGate

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class MoELGBMFeatureGate(nn.Module):
    """
    MoE feature gate with a learned LGBM-prior blend.

    RegimeMoEFeatureGate provides dynamic expert routing (strong IC signal).
    The LGBM-derived prior_weight regularises the output toward stable, data-driven
    feature importance (stronger IR / lower variance).

    The mix coefficient is a learnable scalar so the model can decide how much
    to rely on the prior vs the dynamic MoE routing.
    """

    def __init__(
        self,
        d_input,
        d_output,
        beta=1.0,
        n_experts=3,
        top_k=1,
        router_noise_std=0.02,
        load_balance_coef=0.001,
        prior_mix=0.2,
    ):
        super().__init__()
        self.d_output = d_output
        self.moe = RegimeMoEFeatureGate(
            d_input=d_input,
            d_output=d_output,
            beta=beta,
            n_experts=n_experts,
            top_k=top_k,
            router_noise_std=router_noise_std,
            load_balance_coef=load_balance_coef,
        )
        self.register_buffer("prior_weight", torch.ones(d_output, dtype=torch.float32))
        prior_mix = float(np.clip(prior_mix, 1e-6, 1.0 - 1e-6))
        self.prior_mix_logit = nn.Parameter(
            torch.tensor(float(np.log(prior_mix / (1.0 - prior_mix))), dtype=torch.float32)
        )

    def set_prior_weight(self, prior_weight: np.ndarray):
        prior = np.asarray(prior_weight, dtype=np.float32).reshape(-1)
        prior = np.clip(prior, a_min=0.0, a_max=None)
        total = float(prior.sum())
        if not np.isfinite(total) or total <= 0.0:
            prior = np.ones(self.d_output, dtype=np.float32) / float(self.d_output)
        else:
            prior = prior / total
        prior = prior * float(self.d_output)
        self.prior_weight.copy_(
            torch.from_numpy(prior).to(device=self.prior_weight.device, dtype=self.prior_weight.dtype)
        )

    def forward(self, gate_sequence):
        moe_out = self.moe(gate_sequence)
        mix = torch.sigmoid(self.prior_mix_logit)
        prior = self.prior_weight.unsqueeze(0)
        return (1.0 - mix) * moe_out + mix * prior

    def get_aux_loss(self, device):
        return self.moe.get_aux_loss(device=device)


class MASTERMoELGBMGate(nn.Module):
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
        prior_mix=0.2,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = MoELGBMFeatureGate(
            d_input=self.d_gate_input,
            d_output=d_feat,
            beta=beta,
            n_experts=3,
            top_k=moe_top_k,
            router_noise_std=moe_router_noise_std,
            load_balance_coef=moe_load_balance_coef,
            prior_mix=prior_mix,
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
        src = src * self.feature_gate(gate_input).unsqueeze(1)
        return self.layers(src).squeeze(-1)

    def get_aux_loss(self):
        device = next(self.parameters()).device
        return self.feature_gate.get_aux_loss(device=device)

    def set_prior_weight(self, prior_weight: np.ndarray):
        self.feature_gate.set_prior_weight(prior_weight)


class MASTERMoELGBMGateModel(SequenceModel):
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
        prior_mix=0.2,
        lgbm_num_leaves=31,
        lgbm_learning_rate=0.05,
        lgbm_n_estimators=300,
        lgbm_feature_fraction=0.9,
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
        self.prior_mix = prior_mix
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_learning_rate = lgbm_learning_rate
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_feature_fraction = lgbm_feature_fraction
        self.lgbm_model = None
        self.init_model()

    def init_model(self):
        self.model = MASTERMoELGBMGate(
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
            prior_mix=self.prior_mix,
        )
        super().init_model()

    def _extract_train_arrays(self, dl_train):
        if hasattr(dl_train, "_data"):
            data = np.asarray(dl_train._data, dtype=np.float32)
        else:
            data = np.asarray([dl_train[i] for i in range(len(dl_train))], dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("Expected training data with shape (N, T, F)")
        gate_x = data[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        labels = data[:, -1, -1]
        return gate_x, labels

    def _fit_lgbm_prior(self, dl_train):
        if lgb is None:
            raise ImportError("lightgbm is required. Install with `uv pip install lightgbm`.")
        gate_x, labels = self._extract_train_arrays(dl_train)
        import pandas as pd
        index = dl_train.get_index()
        gate_x = np.nan_to_num(gate_x, nan=0.0, posinf=0.0, neginf=0.0)
        label_series = pd.Series(labels, index=index)
        day_vol = np.abs(label_series).groupby(level="datetime").mean()
        target = day_vol.reindex(index.get_level_values("datetime")).to_numpy(dtype=np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = np.isfinite(target)
        if valid_mask.sum() < 10:
            return
        self.lgbm_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.lgbm_n_estimators,
            learning_rate=self.lgbm_learning_rate,
            num_leaves=self.lgbm_num_leaves,
            feature_fraction=self.lgbm_feature_fraction,
            random_state=self.seed,
        )
        self.lgbm_model.fit(gate_x[valid_mask], target[valid_mask])
        importance = np.asarray(
            self.lgbm_model.booster_.feature_importance(importance_type="gain"), dtype=np.float32
        )
        imp_total = float(importance.sum())
        if not np.isfinite(imp_total) or imp_total <= 0.0:
            importance = np.ones_like(importance) / float(max(1, importance.shape[0]))
        else:
            importance = importance / imp_total
        d_gate = self.gate_input_end_index - self.gate_input_start_index
        d_feat = self.d_feat
        prior = np.ones(d_feat, dtype=np.float32) / float(d_feat)
        if len(importance) == d_gate and d_gate > 0:
            x_src = np.linspace(0, 1, d_gate)
            x_dst = np.linspace(0, 1, d_feat)
            f = interp1d(x_src, importance, kind="cubic", fill_value="extrapolate")
            prior = f(x_dst).astype(np.float32)
            prior = np.clip(prior, 0.0, None)
            total = float(prior.sum())
            if np.isfinite(total) and total > 0.0:
                prior = prior / total
        self.model.set_prior_weight(prior)

    def fit(self, dl_train, dl_valid=None):
        self._fit_lgbm_prior(dl_train)
        super().fit(dl_train, dl_valid)
