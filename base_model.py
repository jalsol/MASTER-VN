import numpy as np
import pandas as pd
import copy
import sys
import os

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
from tqdm import tqdm

from reversal_focal_loss import build_reversal_calendar, reversal_focal_loss

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def calc_topk_dropout_ar_ir(
    predictions: pd.Series,
    labels: pd.Series,
    benchmark_returns: pd.Series = None,
    topk_ratio=0.2,
    n_drop_ratio=0.1,
    topk=None,
    n_drop=None,
    open_cost=0.0025,
    close_cost=0.0025,
    initial_capital=1e8,
    min_cost=5.0,
    rebalance_step=5,
    return_horizon_days=1,
):
    frame = pd.DataFrame({"pred": predictions, "label": labels}).dropna()
    if frame.empty:
        return np.nan, np.nan
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        return np.nan, np.nan
    grouped_days = list(frame.groupby(level="datetime", sort=True))
    if len(grouped_days) == 0:
        return np.nan, np.nan
    market_env = os.environ.get("BACKTEST_MARKET", "auto").strip().lower()
    med_universe = float(np.median([len(day) for _, day in grouped_days]))
    is_vn30 = market_env in {"vn30", "vn", "vietnam"} or (market_env == "auto" and med_universe <= 40.0)
    rebalance_step = max(1, int(rebalance_step))
    signal_shift = max(1, int(os.environ.get("BACKTEST_SIGNAL_SHIFT_DAYS", "1")))
    impact_cost = float(os.environ.get("BACKTEST_IMPACT_COST", "0.0"))
    no_bench_default = "0.65" if is_vn30 else "0.5"
    no_bench_return_scale = float(os.environ.get("BACKTEST_NO_BENCH_RETURN_SCALE", no_bench_default))
    annual_days_default = "252.0" if is_vn30 else "238.0"
    annual_trading_days = float(os.environ.get("BACKTEST_ANNUAL_TRADING_DAYS", annual_days_default))
    period_excess_returns = []
    prev_weights = {}
    nav_port = float(initial_capital)

    label_scale_env = os.environ.get("BACKTEST_LABEL_SCALE")
    if label_scale_env is not None:
        label_scale = float(label_scale_env)
    else:
        daily_mean = frame.groupby(level="datetime")["label"].mean()
        daily_std = frame.groupby(level="datetime")["label"].std()
        mean_abs = float(np.nanmedian(np.abs(daily_mean.values))) if len(daily_mean) > 0 else np.nan
        std_med = float(np.nanmedian(daily_std.values)) if len(daily_std) > 0 else np.nan
        if np.isfinite(mean_abs) and np.isfinite(std_med) and mean_abs < 0.1 and 0.8 <= std_med <= 1.2:
            label_vol_default = "0.015" if is_vn30 else "0.02"
            label_scale = float(os.environ.get("BACKTEST_LABEL_DAILY_VOL", label_vol_default))
        else:
            label_scale = 1.0

    for i in range(0, max(0, len(grouped_days) - signal_shift), rebalance_step):
        pred_dt, day_pred = grouped_days[i]
        ret_dt, day_ret = grouped_days[i + signal_shift]
        if day_pred.empty or day_ret.empty:
            continue
        day_pred = day_pred.sort_values("pred", ascending=False)
        instruments = day_pred.index.get_level_values("instrument")
        n = len(day_pred)
        topk_count = int(topk) if topk is not None else max(1, int(n * topk_ratio))
        topk_count = max(1, min(n, topk_count))
        n_drop_count = int(n_drop) if n_drop is not None else max(1, int(topk_count * n_drop_ratio))
        n_drop_count = max(0, min(topk_count, n_drop_count))

        if prev_weights:
            daily_pred_by_instrument = day_pred.groupby(level="instrument")["pred"].first()
            existing = [ins for ins in prev_weights.keys() if ins in daily_pred_by_instrument.index]
            existing_sorted = sorted(existing, key=lambda ins: float(daily_pred_by_instrument.loc[ins]))
            drop_set = set(existing_sorted[: min(n_drop_count, len(existing_sorted))])
            target = [ins for ins in existing if ins not in drop_set]
            for ins in instruments:
                if len(target) >= topk_count:
                    break
                if ins not in target:
                    target.append(ins)
        else:
            target = list(instruments[:topk_count])

        if len(target) == 0:
            continue

        target_weight = 1.0 / len(target)
        weights = {ins: target_weight for ins in target}
        union = set(prev_weights.keys()) | set(weights.keys())
        deltas = {ins: weights.get(ins, 0.0) - prev_weights.get(ins, 0.0) for ins in union}
        turnover_rate = 0.5 * sum(abs(delta) for delta in deltas.values())
        ret_by_instrument = day_ret.groupby(level="instrument")["label"].first()
        ret_vec = np.array([float(ret_by_instrument.get(ins, 0.0)) for ins in target], dtype=float)
        gross_return = float(np.mean(ret_vec)) * label_scale
        bench_return = float(day_ret["label"].mean()) * label_scale
        has_real_benchmark = False
        if benchmark_returns is not None and ret_dt in benchmark_returns.index:
            bench_value = float(benchmark_returns.loc[ret_dt])
            if not np.isnan(bench_value):
                bench_return = bench_value
                has_real_benchmark = True
        if not has_real_benchmark:
            gross_return *= no_bench_return_scale
            bench_return *= no_bench_return_scale

        total_fee = 0.0
        for delta in deltas.values():
            if delta > 0:
                notional = float(delta * nav_port)
                fee = notional * open_cost
                if min_cost > 0:
                    fee = max(fee, float(min_cost))
                total_fee += fee
            elif delta < 0:
                notional = float((-delta) * nav_port)
                fee = notional * close_cost
                if min_cost > 0:
                    fee = max(fee, float(min_cost))
                total_fee += fee
        cost_rate = total_fee / max(nav_port, 1e-12) + turnover_rate * impact_cost
        net_return = gross_return - cost_rate
        nav_port = nav_port * (1.0 + net_return)
        if nav_port <= 0:
            continue
        period_excess = gross_return - bench_return - cost_rate
        if np.isnan(period_excess):
            continue
        period_excess_returns.append(period_excess)
        prev_weights = weights

    if len(period_excess_returns) == 0:
        return np.nan, np.nan
    excess = np.asarray(period_excess_returns, dtype=float)
    excess = excess[np.isfinite(excess)]
    if excess.size == 0:
        return np.nan, np.nan
    annual_factor = annual_trading_days / max(1, int(rebalance_step))
    mean_excess = float(np.mean(excess))
    ar = mean_excess * annual_factor
    std = float(np.std(excess, ddof=1)) if excess.size > 1 else np.nan
    if (not np.isfinite(std)) or std < 1e-12:
        return ar, np.nan
    ir = float((mean_excess / std) * np.sqrt(annual_factor))
    return ar, ir

def zscore(x, eps=1e-6):
    if torch.is_tensor(x):
        if x.numel() == 0:
            return x
        mean = torch.mean(x)
        std = torch.std(x, unbiased=False)
        if torch.isnan(std) or std < eps:
            std = torch.tensor(eps, device=x.device, dtype=x.dtype)
        return (x - mean) / std
    else:
        std = x.std()
        if pd.isna(std) or std < eps:
            std = eps
        return (x - x.mean()).div(std)

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    if percent_2_5 == 0 or percent_2_5 * 2 >= N:
        mask = torch.ones_like(x, device=x.device, dtype=torch.bool)
        return mask, x
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        index = self.data_source.get_index()
        grouped = pd.Series(index=index).groupby("datetime")
        self.daily_count = grouped.size().values
        self.batch_dates = grouped.size().index.values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        self._current_date = None

    def __iter__(self):
        if self.shuffle:
            order = np.arange(len(self.daily_count))
            np.random.shuffle(order)
            for i in order:
                self._current_date = self.batch_dates[i]
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for i, (idx, count) in enumerate(zip(self.daily_index, self.daily_count)):
                self._current_date = self.batch_dates[i]
                yield np.arange(idx, idx + count)

    def get_current_date(self):
        return self._current_date

    def __len__(self):
        return len(self.daily_count)


class SequenceModel():
    def __init__(
        self,
        n_epochs,
        lr,
        GPU=None,
        seed=None,
        train_stop_loss_thred=None,
        save_path='model/',
        save_prefix='',
        use_amp=True,
        use_reversal_focal_loss=False,
        reversal_multiplier=5.0,
        focal_gamma=2.0,
        reversal_window_days=3,
        reversal_trend_window=20,
        reversal_prominence_pct=0.5,
        reversal_min_distance=5,
        reversal_market_symbol='VN30',
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_reversal_focal_loss = use_reversal_focal_loss
        self.reversal_multiplier = reversal_multiplier
        self.focal_gamma = focal_gamma
        self.reversal_window_days = reversal_window_days
        self.reversal_trend_window = reversal_trend_window
        self.reversal_prominence_pct = reversal_prominence_pct
        self.reversal_min_distance = reversal_min_distance
        self.reversal_market_symbol = reversal_market_symbol
        self._reversal_dates = None
        if GPU is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{GPU}")
            device_msg = f"cuda:{GPU}"
        else:
            self.device = torch.device("cpu")
            device_msg = "cpu"
        print(f"[MASTER] Using device: {device_msg}, AMP: {self.use_amp}", flush=True)
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        
        # Enable cudnn benchmark for faster training
        torch.backends.cudnn.benchmark = True
        
        self.fitted = -1

        self.model = None
        self.train_optimizer = None
        self.scaler = None  # For AMP

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)
        
        # Initialize GradScaler for AMP
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

    def _is_reversal_batch(self, batch_date):
        if not self.use_reversal_focal_loss or batch_date is None or not self._reversal_dates:
            return False
        return pd.Timestamp(batch_date) in self._reversal_dates

    def loss_fn(self, pred, label, batch_date=None):
        if self.use_reversal_focal_loss:
            return reversal_focal_loss(
                pred,
                label,
                in_reversal_window=self._is_reversal_batch(batch_date),
                reversal_multiplier=self.reversal_multiplier,
                focal_gamma=self.focal_gamma,
            )
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    def _get_batch_date(self, data_loader):
        sampler = getattr(data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "get_current_date"):
            return sampler.get_current_date()
        return None

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in tqdm(data_loader, desc="  batch", leave=False, file=sys.stdout):
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            
            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            mask, label = drop_extreme(label)
            feature = feature[mask, :, :]
            label = zscore(label) # CSZscoreNorm
            #########################

            self.train_optimizer.zero_grad()
            
            batch_date = self._get_batch_date(data_loader)
            if self.use_amp:
                # Mixed precision training
                with torch.amp.autocast('cuda'):
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label, batch_date=batch_date)
                    if hasattr(self.model, "get_aux_loss"):
                        loss = loss + self.model.get_aux_loss()
                losses.append(loss.item())
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.train_optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.scaler.step(self.train_optimizer)
                self.scaler.update()
            else:
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label, batch_date=batch_date)
                if hasattr(self.model, "get_aux_loss"):
                    loss = loss + self.model.get_aux_loss()
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Note the difference: 
            # 1) The qlib.DropnaLabel drop **samples** according to label.
            # 2) Here we use all samples to compute the inter-stock correlation, but only drop the na labels to compute metrics (loss, etc.).
            # 3) If you already used qlib.DropnaLabel to process the validation data, this will do nothing.
            mask, label = drop_na(label)
            label = zscore(label)
                        
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pred = self.model(feature.float())
            else:
                pred = self.model(feature.float())
            loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))
    
    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None):
        if self.use_reversal_focal_loss:
            self._reversal_dates, reversal_source = build_reversal_calendar(
                dl_train,
                symbol=self.reversal_market_symbol,
                trend_window=self.reversal_trend_window,
                prominence_pct=self.reversal_prominence_pct,
                window_days=self.reversal_window_days,
                min_distance=self.reversal_min_distance,
            )
            print(
                f"[MASTER] Reversal-weighted focal loss enabled: "
                f"{len(self._reversal_dates)} reversal days from {reversal_source}, "
                f"multiplier={self.reversal_multiplier}, gamma={self.focal_gamma}",
                flush=True,
            )
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        best_valid_icir = float("-inf")
        epoch_bar = tqdm(range(self.n_epochs), desc="epoch", unit="ep", file=sys.stdout)
        for step in epoch_bar:
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                ic, icir, ric, ricir = metrics['IC'], metrics['ICIR'], metrics['RIC'], metrics['RICIR']
                ar, ir = metrics['AR'], metrics['IR']
                epoch_bar.set_postfix(loss=f"{train_loss:.4f}", ic=f"{ic:.4f}", icir=f"{icir:.3f}", ric=f"{ric:.4f}", best_icir=f"{best_valid_icir:.3f}")
                print(f"  Epoch {step:3d} | loss={train_loss:.6f} | ic={ic:.4f} icir={icir:.3f} ric={ric:.4f} ricir={ricir:.3f} ar={ar:.4f} ir={ir:.3f}", flush=True)
                if icir > best_valid_icir:
                    best_valid_icir = icir
                    best_param = copy.deepcopy(self.model.state_dict())
            else:
                epoch_bar.set_postfix(loss=f"{train_loss:.4f}")
                print(f"  Epoch {step:3d} | loss={train_loss:.6f}", flush=True)
            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                if best_param is None:
                    best_param = copy.deepcopy(self.model.state_dict())
                break
        if best_param is not None:
            self.model.load_state_dict(best_param)
        

    def predict(self, dl_test):
        if self.fitted<0:
            raise ValueError("model is not fitted yet!")
        else:
            print('Epoch:', self.fitted, flush=True)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        labels = []
        benchmark_by_date = {}
        full_index = dl_test.get_index()
        index_offset = 0
        gate_start = getattr(self.model, "gate_input_start_index", None)
        gate_end = getattr(self.model, "gate_input_end_index", None)
        use_index_feature_benchmark = os.environ.get("BACKTEST_USE_INDEX_FEATURE_BENCHMARK", "0") == "1"
        bench_group_env = os.environ.get("BACKTEST_BENCHMARK_GROUP")
        bench_group = int(bench_group_env) if bench_group_env is not None else -1
        bench_feature_idx = None
        if use_index_feature_benchmark and gate_start is not None and gate_end is not None:
            d_gate = int(gate_end - gate_start)
            if bench_group >= 0 and d_gate % 3 == 0:
                block = d_gate // 3
                bench_feature_idx = gate_start + min(bench_group, 2) * block
            else:
                bench_feature_idx = gate_start

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            batch_size = int(data.shape[0])
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            if bench_feature_idx is not None and batch_size > 0:
                batch_index = full_index[index_offset:index_offset + batch_size]
                day_dt = batch_index.get_level_values("datetime")[0]
                benchmark_by_date[day_dt] = float(feature[0, -1, bench_feature_idx].detach().cpu().item())
            index_offset += batch_size
            
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())
            labels.append(label.detach().cpu().numpy().ravel())

        index = dl_test.get_index()
        predictions = pd.Series(np.concatenate(preds), index=index)
        label_series = pd.Series(np.concatenate(labels), index=index)
        benchmark_series = pd.Series(benchmark_by_date).sort_index() if len(benchmark_by_date) > 0 else None

        ic = []
        ric = []
        eval_df = pd.DataFrame({"pred": predictions, "label": label_series}).dropna()
        for _, day in eval_df.groupby(level="datetime", sort=True):
            if len(day) < 2:
                continue
            daily_ic = day["pred"].corr(day["label"])
            daily_ric = day["pred"].corr(day["label"], method="spearman")
            if not np.isnan(daily_ic):
                ic.append(daily_ic)
            if not np.isnan(daily_ric):
                ric.append(daily_ric)
        
        backtest_horizon = int(os.environ.get("BACKTEST_HORIZON_DAYS", os.environ.get("LABEL_HORIZON_DAYS", "5")))
        return_horizon_days = int(os.environ.get("BACKTEST_RETURN_HORIZON_DAYS", "1"))
        topk_env = os.environ.get("BACKTEST_TOPK")
        n_drop_env = os.environ.get("BACKTEST_NDROP")
        initial_capital_env = os.environ.get("BACKTEST_INITIAL_CAPITAL")
        min_cost_env = os.environ.get("BACKTEST_MIN_COST")
        ar, ir = calc_topk_dropout_ar_ir(
            predictions,
            label_series,
            benchmark_returns=benchmark_series,
            topk=int(topk_env) if topk_env is not None else None,
            n_drop=int(n_drop_env) if n_drop_env is not None else None,
            initial_capital=float(initial_capital_env) if initial_capital_env is not None else 1e8,
            min_cost=float(min_cost_env) if min_cost_env is not None else 5.0,
            rebalance_step=backtest_horizon,
            return_horizon_days=return_horizon_days,
        )

        metrics = {
            'IC': np.mean(ic) if len(ic) > 0 else np.nan,
            'ICIR': (np.mean(ic)/np.std(ic)) if len(ic) > 1 and np.std(ic) > 1e-12 else np.nan,
            'RIC': np.mean(ric) if len(ric) > 0 else np.nan,
            'RICIR': (np.mean(ric)/np.std(ric)) if len(ric) > 1 and np.std(ric) > 1e-12 else np.nan,
            'AR': ar,
            'IR': ir
        }

        return predictions, metrics
