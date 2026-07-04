import numpy as np
import pandas as pd
import torch
from torch import nn

from base_model import SequenceModel
from master import Gate, PositionalEncoding, SAttention, TAttention, TemporalAttention

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class MASTERLeafInputEncoder(nn.Module):
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
        static_feature_start,
        static_feature_end,
        leaves_per_tree,
        max_trees,
        ffn_expand=4,
        embedding_stage="pre_spatial",
        leaf_integration_mode="add",
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.static_feature_start = int(static_feature_start)
        self.static_feature_end = int(static_feature_end)
        self.leaves_per_tree = int(leaves_per_tree)
        self.max_trees = int(max_trees)
        self.leaf_integration_mode = str(leaf_integration_mode).lower()
        self.embedding_stage = str(embedding_stage).lower()

        self.feature_gate = Gate(gate_input_end_index - gate_input_start_index, d_feat, beta=beta)
        self.feature_proj = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.temporal_block = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate, ffn_expand=ffn_expand)
        self.spatial_block = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate, ffn_expand=ffn_expand)
        self.pooling = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

        total_leaf_nodes = self.max_trees * self.leaves_per_tree
        if self.leaf_integration_mode == "concat":
            self.leaf_embedding = nn.Embedding(total_leaf_nodes, d_model)
            self.leaf_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.leaf_embedding = nn.Embedding(total_leaf_nodes, d_model)
            self.leaf_proj = None

        self.tree_offsets = torch.arange(self.max_trees, dtype=torch.long) * self.leaves_per_tree
        self.active_trees = self.max_trees
        self.leaf_model = None

    def set_leaf_model(self, leaf_model, active_trees):
        self.leaf_model = leaf_model
        self.active_trees = int(active_trees)
        offsets = torch.arange(self.active_trees, dtype=torch.long) * self.leaves_per_tree
        self.tree_offsets = offsets

    def _leaf_indices(self, x):
        if self.leaf_model is None:
            raise ValueError("Leaf LightGBM model is not fitted.")
        static_part = x[:, -1, self.static_feature_start:self.static_feature_end].detach().cpu().numpy().astype(np.float32, copy=False)
        leaf_idx = self.leaf_model.booster_.predict(static_part, pred_leaf=True)
        leaf_idx = np.asarray(leaf_idx, dtype=np.int64)
        if leaf_idx.ndim == 1:
            leaf_idx = leaf_idx[:, None]
        leaf_idx = leaf_idx[:, :self.active_trees]
        leaf_idx = np.clip(leaf_idx, 0, self.leaves_per_tree - 1)
        offsets = self.tree_offsets.detach().cpu().numpy()[None, :]
        global_idx = leaf_idx + offsets
        return torch.from_numpy(global_idx).to(x.device, non_blocking=True)

    def encode(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        h = self.feature_proj(src)
        leaf_indices = self._leaf_indices(x)
        leaf_emb = self.leaf_embedding(leaf_indices).sum(dim=1)
        leaf_expand = leaf_emb.unsqueeze(1).expand(-1, h.shape[1], -1)

        if self.embedding_stage == "pre_temporal":
            if self.leaf_integration_mode == "concat":
                h = self.leaf_proj(torch.cat([h, leaf_expand], dim=-1))
            else:
                h = h + leaf_expand
        h = self.pos_enc(h)
        h = self.temporal_block(h)
        if self.embedding_stage in {"pre_spatial", "post_temporal"}:
            if self.leaf_integration_mode == "concat":
                h = self.leaf_proj(torch.cat([h, leaf_expand], dim=-1))
            else:
                h = h + leaf_expand
        h = self.spatial_block(h)
        if self.embedding_stage in {"post_spatial", "post_inter"}:
            pooled = self.pooling(h)
            if self.leaf_integration_mode == "concat":
                pooled = self.leaf_proj(torch.cat([pooled, leaf_emb], dim=-1))
            else:
                pooled = pooled + leaf_emb
            return pooled
        return self.pooling(h)

    def forward(self, x):
        emb = self.encode(x)
        return self.decoder(emb).squeeze(-1)


class MASTERLGBMLeafInputModel(SequenceModel):
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
        lgbm_num_leaves=63,
        lgbm_learning_rate=0.03,
        lgbm_n_estimators=600,
        lgbm_feature_fraction=0.85,
        lgbm_subsample=0.9,
        lgbm_subsample_freq=1,
        lgbm_lambda_l1=0.0,
        lgbm_lambda_l2=0.0,
        lgbm_min_child_samples=30,
        lgbm_early_stopping_rounds=100,
        lgbm_task="regression",
        lgbm_rank_bins=20,
        lgbm_label_mode="cs_zscore",
        static_feature_start=0,
        static_feature_end=None,
        embedding_stage="pre_spatial",
        leaf_integration_mode="add",
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
        self.static_feature_start = int(static_feature_start)
        if static_feature_end is None:
            static_feature_end = gate_input_start_index
        self.static_feature_end = int(static_feature_end)
        self.embedding_stage = str(embedding_stage).lower()
        self.leaf_integration_mode = str(leaf_integration_mode).lower()

        self.lgbm_num_leaves = int(lgbm_num_leaves)
        self.lgbm_learning_rate = float(lgbm_learning_rate)
        self.lgbm_n_estimators = int(lgbm_n_estimators)
        self.lgbm_feature_fraction = float(lgbm_feature_fraction)
        self.lgbm_subsample = float(lgbm_subsample)
        self.lgbm_subsample_freq = int(lgbm_subsample_freq)
        self.lgbm_lambda_l1 = float(lgbm_lambda_l1)
        self.lgbm_lambda_l2 = float(lgbm_lambda_l2)
        self.lgbm_min_child_samples = int(lgbm_min_child_samples)
        self.lgbm_early_stopping_rounds = int(lgbm_early_stopping_rounds)
        self.lgbm_task = str(lgbm_task).lower()
        self.lgbm_rank_bins = int(lgbm_rank_bins)
        self.lgbm_label_mode = str(lgbm_label_mode).lower()

        self.leaf_model = None
        self.init_model()

    def init_model(self):
        self.model = MASTERLeafInputEncoder(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            static_feature_start=self.static_feature_start,
            static_feature_end=self.static_feature_end,
            leaves_per_tree=self.lgbm_num_leaves,
            max_trees=self.lgbm_n_estimators,
            ffn_expand=self.ffn_expand,
            embedding_stage=self.embedding_stage,
            leaf_integration_mode=self.leaf_integration_mode,
        )
        super().init_model()

    def _extract_leaf_training_data(self, dl_data):
        loader = self._init_data_loader(dl_data, shuffle=False, drop_last=False)
        xs = []
        ys = []
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                data = torch.squeeze(data, dim=0)
                feature = data[:, :, 0:-1]
                label = data[:, -1, -1]
                static_part = feature[:, -1, self.static_feature_start:self.static_feature_end].detach().cpu().numpy()
                xs.append(static_part.astype(np.float32, copy=False))
                ys.append(label.detach().cpu().numpy().astype(np.float32, copy=False))
        if len(xs) == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32), dl_data.get_index()[:0]
        x = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
        y = np.concatenate(ys, axis=0).astype(np.float32, copy=False)
        return x, y, dl_data.get_index()

    def _transform_labels(self, y, index):
        if self.lgbm_label_mode != "cs_zscore":
            return y
        y_series = pd.Series(y, index=index)
        transformed = []
        for _, day in y_series.groupby(level="datetime", sort=True):
            arr = day.to_numpy(dtype=np.float32)
            mu = float(np.nanmean(arr))
            sigma = float(np.nanstd(arr))
            if (not np.isfinite(sigma)) or sigma < 1e-8:
                transformed.append(np.zeros_like(arr, dtype=np.float32))
            else:
                transformed.append((arr - mu) / sigma)
        if len(transformed) == 0:
            return y
        return np.concatenate(transformed, axis=0).astype(np.float32, copy=False)

    def _filter_finite(self, x, y, index):
        mask = np.isfinite(y)
        if x.shape[0] != y.shape[0] or y.shape[0] != len(index):
            raise ValueError("Mismatched x/y/index lengths.")
        if int(mask.sum()) == y.shape[0]:
            return x, y, index
        return x[mask], y[mask], index[mask]

    def _build_groups_from_index(self, index):
        if len(index) == 0:
            return []
        return pd.Series(1, index=index).groupby(level="datetime", sort=True).sum().astype(int).tolist()

    def _build_rank_labels(self, y, index):
        bins = max(2, int(self.lgbm_rank_bins))
        y_series = pd.Series(np.asarray(y, dtype=np.float32), index=index)
        out = np.zeros(len(y_series), dtype=np.int32)
        cursor = 0
        for _, day in y_series.groupby(level="datetime", sort=True):
            arr = day.to_numpy(dtype=np.float32)
            n = arr.shape[0]
            if n <= 1:
                out[cursor:cursor + n] = 0
                cursor += n
                continue
            ranks = pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float32)
            labels = np.minimum((ranks * bins).astype(np.int32), bins - 1)
            out[cursor:cursor + n] = labels
            cursor += n
        return out

    def _fit_leaf_model(self, dl_train, dl_valid=None):
        if lgb is None:
            raise ImportError("lightgbm is required for MASTERLGBMLeafInputModel. Install with `uv pip install lightgbm`.")
        x_train, y_train, index_train = self._extract_leaf_training_data(dl_train)
        x_train, y_train, index_train = self._filter_finite(x_train, y_train, index_train)
        if int(y_train.shape[0]) < 20:
            raise ValueError("Not enough valid training labels for leaf LightGBM encoder.")
        if dl_valid is not None:
            x_valid, y_valid, index_valid = self._extract_leaf_training_data(dl_valid)
            x_valid, y_valid, index_valid = self._filter_finite(x_valid, y_valid, index_valid)
        else:
            x_valid = np.empty((0, x_train.shape[1]), dtype=np.float32)
            y_valid = np.empty((0,), dtype=np.float32)
            index_valid = index_train[:0]

        if self.lgbm_task == "ranker":
            y_train_fit = self._build_rank_labels(y_train, index_train)
            group_train = self._build_groups_from_index(index_train)
            leaf_model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=self.lgbm_n_estimators,
                learning_rate=self.lgbm_learning_rate,
                num_leaves=self.lgbm_num_leaves,
                colsample_bytree=self.lgbm_feature_fraction,
                subsample=self.lgbm_subsample,
                subsample_freq=self.lgbm_subsample_freq,
                reg_alpha=self.lgbm_lambda_l1,
                reg_lambda=self.lgbm_lambda_l2,
                min_child_samples=self.lgbm_min_child_samples,
                random_state=self.seed,
                verbosity=-1,
            )
            if dl_valid is None or y_valid.shape[0] < 20:
                leaf_model.fit(x_train, y_train_fit, group=group_train)
            else:
                y_valid_fit = self._build_rank_labels(y_valid, index_valid)
                group_valid = self._build_groups_from_index(index_valid)
                callbacks = []
                if self.lgbm_early_stopping_rounds > 0:
                    callbacks.append(lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=False))
                leaf_model.fit(
                    x_train,
                    y_train_fit,
                    group=group_train,
                    eval_set=[(x_valid, y_valid_fit)],
                    eval_group=[group_valid],
                    eval_metric="ndcg",
                    callbacks=callbacks,
                )
            self.leaf_model = leaf_model
            return

        y_train_fit = self._transform_labels(y_train, index_train)
        leaf_model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.lgbm_n_estimators,
            learning_rate=self.lgbm_learning_rate,
            num_leaves=self.lgbm_num_leaves,
            colsample_bytree=self.lgbm_feature_fraction,
            subsample=self.lgbm_subsample,
            subsample_freq=self.lgbm_subsample_freq,
            reg_alpha=self.lgbm_lambda_l1,
            reg_lambda=self.lgbm_lambda_l2,
            min_child_samples=self.lgbm_min_child_samples,
            random_state=self.seed,
            verbosity=-1,
        )
        if dl_valid is None or y_valid.shape[0] < 20:
            leaf_model.fit(x_train, y_train_fit)
        else:
            y_valid_fit = self._transform_labels(y_valid, index_valid)
            callbacks = []
            if self.lgbm_early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.lgbm_early_stopping_rounds, verbose=False))
            leaf_model.fit(
                x_train,
                y_train_fit,
                eval_set=[(x_valid, y_valid_fit)],
                eval_metric="l2",
                callbacks=callbacks,
            )
        self.leaf_model = leaf_model

    def fit(self, dl_train, dl_valid=None):
        self._fit_leaf_model(dl_train, dl_valid)
        probe_x, _, _ = self._extract_leaf_training_data(dl_train)
        if probe_x.shape[0] == 0:
            raise ValueError("Cannot infer active tree count from empty training data.")
        probe_leaf = self.leaf_model.booster_.predict(probe_x[:1], pred_leaf=True)
        probe_leaf = np.asarray(probe_leaf)
        active_trees = int(probe_leaf.shape[1]) if probe_leaf.ndim > 1 else 1
        self.model.set_leaf_model(self.leaf_model, active_trees=active_trees)
        self.model.to(self.device)
        super().fit(dl_train, dl_valid)
