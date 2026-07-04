import numpy as np
import torch
from torch import nn

from base_model import SequenceModel
from master import Gate, PositionalEncoding, SAttention, TAttention, TemporalAttention

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LGBMMarketPriorGate(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        beta=1.0,
        prior_mix=0.35,
        shuffle_mix=0.0,
        dynamic_prior_mix=0.6,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.feature_gate = Gate(d_input, d_output, beta=beta)
        self.register_buffer("prior_weight", torch.ones(d_output, dtype=torch.float32))
        self.register_buffer("corr_map", torch.zeros(d_output, d_input, dtype=torch.float32))
        self.register_buffer("market_importance", torch.ones(d_input, dtype=torch.float32))
        self.register_buffer("gate_mean", torch.zeros(d_input, dtype=torch.float32))
        self.register_buffer("gate_std", torch.ones(d_input, dtype=torch.float32))
        prior_mix = float(np.clip(prior_mix, 1e-6, 1.0 - 1e-6))
        shuffle_mix = float(np.clip(shuffle_mix, 1e-6, 1.0 - 1e-6))
        dynamic_prior_mix = float(np.clip(dynamic_prior_mix, 1e-6, 1.0 - 1e-6))
        self.prior_mix_logit = nn.Parameter(torch.tensor(float(np.log(prior_mix / (1.0 - prior_mix))), dtype=torch.float32))
        self.shuffle_mix_logit = nn.Parameter(torch.tensor(float(np.log(shuffle_mix / (1.0 - shuffle_mix))), dtype=torch.float32))
        self.dynamic_prior_mix_logit = nn.Parameter(torch.tensor(float(np.log(dynamic_prior_mix / (1.0 - dynamic_prior_mix))), dtype=torch.float32))

    def set_prior_weight(self, prior_weight: np.ndarray):
        prior = np.asarray(prior_weight, dtype=np.float32).reshape(-1)
        if prior.shape[0] != self.d_output:
            raise ValueError(f"Expected prior length {self.d_output}, got {prior.shape[0]}")
        prior = np.clip(prior, a_min=0.0, a_max=None)
        total = float(prior.sum())
        if not np.isfinite(total) or total <= 0.0:
            prior = np.ones(self.d_output, dtype=np.float32) / float(self.d_output)
        else:
            prior = prior / total
        prior = prior * float(self.d_output)
        self.prior_weight.copy_(torch.from_numpy(prior).to(device=self.prior_weight.device, dtype=self.prior_weight.dtype))

    def set_market_map(
        self,
        corr_map: np.ndarray,
        market_importance: np.ndarray,
        gate_mean: np.ndarray,
        gate_std: np.ndarray,
    ):
        corr = np.asarray(corr_map, dtype=np.float32)
        if corr.shape != (self.d_output, self.d_input):
            raise ValueError(f"Expected corr_map shape {(self.d_output, self.d_input)}, got {corr.shape}")
        importance = np.asarray(market_importance, dtype=np.float32).reshape(-1)
        if importance.shape[0] != self.d_input:
            raise ValueError(f"Expected market_importance length {self.d_input}, got {importance.shape[0]}")
        imp_total = float(importance.sum())
        if not np.isfinite(imp_total) or imp_total <= 0.0:
            importance = np.ones(self.d_input, dtype=np.float32) / float(self.d_input)
        else:
            importance = importance / imp_total
        mean = np.asarray(gate_mean, dtype=np.float32).reshape(-1)
        std = np.asarray(gate_std, dtype=np.float32).reshape(-1)
        if mean.shape[0] != self.d_input or std.shape[0] != self.d_input:
            raise ValueError("gate_mean and gate_std must match gate input dimension")
        std = np.clip(std, a_min=1e-6, a_max=None)
        self.corr_map.copy_(torch.from_numpy(corr).to(device=self.corr_map.device, dtype=self.corr_map.dtype))
        self.market_importance.copy_(torch.from_numpy(importance).to(device=self.market_importance.device, dtype=self.market_importance.dtype))
        self.gate_mean.copy_(torch.from_numpy(mean).to(device=self.gate_mean.device, dtype=self.gate_mean.dtype))
        self.gate_std.copy_(torch.from_numpy(std).to(device=self.gate_std.device, dtype=self.gate_std.dtype))

    def _dynamic_prior(self, gate_input: torch.Tensor) -> torch.Tensor:
        z = (gate_input - self.gate_mean.unsqueeze(0)) / self.gate_std.unsqueeze(0)
        market_state = torch.abs(z) * self.market_importance.unsqueeze(0)
        raw_prior = torch.matmul(market_state, self.corr_map.transpose(0, 1)).clamp_min(0.0)
        norm = raw_prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return raw_prior / norm * float(self.d_output)

    def forward(self, src, gate_input):
        dynamic_weight = self.feature_gate(gate_input)
        dyn_tree_prior = self._dynamic_prior(gate_input)
        static_tree_prior = self.prior_weight.unsqueeze(0)
        tree_mix = torch.sigmoid(self.dynamic_prior_mix_logit)
        tree_prior = (1.0 - tree_mix) * static_tree_prior + tree_mix * dyn_tree_prior
        mix = torch.sigmoid(self.prior_mix_logit)
        combined_weight = (1.0 - mix) * dynamic_weight + mix * tree_prior
        gated_src = src * combined_weight.unsqueeze(1)
        if self.training:
            shuffle_mix = torch.sigmoid(self.shuffle_mix_logit)
            if float(shuffle_mix.detach().cpu().item()) > 1e-6 and gated_src.shape[0] > 1:
                shuffled = gated_src[torch.randperm(gated_src.shape[0], device=gated_src.device)]
                gated_src = (1.0 - shuffle_mix) * gated_src + shuffle_mix * shuffled
        return gated_src


class MASTERLGBMGate(nn.Module):
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
        prior_mix=0.35,
        shuffle_mix=0.0,
        dynamic_prior_mix=0.6,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = LGBMMarketPriorGate(
            d_input=self.d_gate_input,
            d_output=d_feat,
            beta=beta,
            prior_mix=prior_mix,
            shuffle_mix=shuffle_mix,
            dynamic_prior_mix=dynamic_prior_mix,
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
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = self.feature_gate(src, gate_input)
        return self.layers(src).squeeze(-1)

    def set_prior_weight(self, prior_weight: np.ndarray):
        self.feature_gate.set_prior_weight(prior_weight)

    def set_market_map(
        self,
        corr_map: np.ndarray,
        market_importance: np.ndarray,
        gate_mean: np.ndarray,
        gate_std: np.ndarray,
    ):
        self.feature_gate.set_market_map(
            corr_map=corr_map,
            market_importance=market_importance,
            gate_mean=gate_mean,
            gate_std=gate_std,
        )


class MASTERLGBMGateModel(SequenceModel):
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
        prior_mix=0.35,
        shuffle_mix=0.0,
        dynamic_prior_mix=0.6,
        lgbm_num_leaves=31,
        lgbm_learning_rate=0.05,
        lgbm_n_estimators=300,
        lgbm_feature_fraction=0.9,
        lgbm_target="vol",
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
        self.prior_mix = prior_mix
        self.shuffle_mix = shuffle_mix
        self.dynamic_prior_mix = dynamic_prior_mix
        self.lgbm_num_leaves = lgbm_num_leaves
        self.lgbm_learning_rate = lgbm_learning_rate
        self.lgbm_n_estimators = lgbm_n_estimators
        self.lgbm_feature_fraction = lgbm_feature_fraction
        self.lgbm_target = str(lgbm_target).lower()
        self.lgbm_model = None
        self.init_model()

    def init_model(self):
        self.model = MASTERLGBMGate(
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
            prior_mix=self.prior_mix,
            shuffle_mix=self.shuffle_mix,
            dynamic_prior_mix=self.dynamic_prior_mix,
        )
        super().init_model()

    def _extract_train_arrays(self, dl_train):
        if hasattr(dl_train, "_data"):
            data = np.asarray(dl_train._data, dtype=np.float32)
        else:
            data = np.asarray([dl_train[i] for i in range(len(dl_train))], dtype=np.float32)
        if data.ndim != 3:
            raise ValueError("Expected training data with shape (N, T, F)")
        index = dl_train.get_index()
        gate_x = data[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        stock_x = data[:, -1, :self.gate_input_start_index]
        labels = data[:, -1, -1]
        return gate_x, stock_x, labels, index

    def _build_market_target(self, labels, index):
        label_series = np.asarray(labels, dtype=np.float32)
        df = {"label": label_series}
        import pandas as pd
        day_label = pd.Series(df["label"], index=index).groupby(level="datetime").mean()
        if self.lgbm_target == "direction":
            target_by_day = np.sign(day_label)
        else:
            day_abs = pd.Series(np.abs(df["label"]), index=index).groupby(level="datetime").mean()
            target_by_day = day_abs
        target = target_by_day.reindex(index.get_level_values("datetime")).to_numpy(dtype=np.float32)
        return np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_stock_prior_and_corr(self, gate_x, stock_x, market_importance):
        gate_x = np.nan_to_num(gate_x, nan=0.0, posinf=0.0, neginf=0.0)
        stock_x = np.nan_to_num(stock_x, nan=0.0, posinf=0.0, neginf=0.0)
        n_stock = stock_x.shape[1]
        n_gate = gate_x.shape[1]
        corr = np.zeros((n_stock, n_gate), dtype=np.float32)
        stock_std = stock_x.std(axis=0)
        gate_std = gate_x.std(axis=0)
        for j in range(n_stock):
            if stock_std[j] < 1e-8:
                continue
            sx = stock_x[:, j]
            sx_center = sx - sx.mean()
            for k in range(n_gate):
                if gate_std[k] < 1e-8:
                    continue
                gx = gate_x[:, k]
                gx_center = gx - gx.mean()
                denom = float(np.sqrt((sx_center * sx_center).mean() * (gx_center * gx_center).mean()))
                if denom < 1e-12:
                    continue
                corr[j, k] = float(np.abs((sx_center * gx_center).mean() / denom))
        prior = corr @ market_importance
        total = float(prior.sum())
        if not np.isfinite(total) or total <= 0.0:
            prior = np.ones(n_stock, dtype=np.float32) / float(max(1, n_stock))
        else:
            prior = prior / total
        return prior, corr

    def _fit_lgbm_prior(self, dl_train):
        if lgb is None:
            raise ImportError("lightgbm is required for MASTERLGBMGateModel. Install with `uv pip install lightgbm`.")
        gate_x, stock_x, labels, index = self._extract_train_arrays(dl_train)
        target = self._build_market_target(labels, index)
        gate_x = np.nan_to_num(gate_x, nan=0.0, posinf=0.0, neginf=0.0)
        valid_mask = np.isfinite(target)
        if valid_mask.sum() < 10:
            prior = np.ones(self.d_feat, dtype=np.float32) / float(self.d_feat)
            self.model.set_prior_weight(prior)
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
        importance = np.asarray(self.lgbm_model.booster_.feature_importance(importance_type="gain"), dtype=np.float32)
        imp_total = float(importance.sum())
        if not np.isfinite(imp_total) or imp_total <= 0.0:
            importance = np.ones_like(importance) / float(max(1, importance.shape[0]))
        else:
            importance = importance / imp_total
        stock_prior, corr_map = self._compute_stock_prior_and_corr(
            gate_x=gate_x,
            stock_x=stock_x,
            market_importance=importance,
        )
        gate_mean = np.mean(gate_x, axis=0).astype(np.float32)
        gate_std = np.std(gate_x, axis=0).astype(np.float32)
        self.model.set_prior_weight(stock_prior)
        self.model.set_market_map(
            corr_map=corr_map,
            market_importance=importance,
            gate_mean=gate_mean,
            gate_std=gate_std,
        )

    def fit(self, dl_train, dl_valid=None):
        self._fit_lgbm_prior(dl_train)
        super().fit(dl_train, dl_valid)
