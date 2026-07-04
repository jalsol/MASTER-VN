#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install numpy and pandas before running this script.") from exc

try:
    from vnstock import Quote  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install vnstock v4+ via `uv pip install vnstock`.") from exc

try:
    from qlib.contrib.data.loader import Alpha158DL  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install pyqlib via `uv pip install pyqlib`.") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.master_dataset import DatasetMetadata, MasterTensorDataset  # noqa: E402

ALPHA_FEATURE_NAMES = Alpha158DL.get_feature_config()[1]
ALPHA_WINDOWS = [5, 10, 20, 30, 60]
MARKET_WINDOWS = [5, 10, 20, 30, 60]
MARKET_INDEXES = ["VNINDEX", "VN100", "VN30"]

MARKET_FEATURE_NAMES: List[str] = []
for idx in MARKET_INDEXES:
    MARKET_FEATURE_NAMES.append(f"{idx}_RET")
    for window in MARKET_WINDOWS:
        MARKET_FEATURE_NAMES.extend(
            [
                f"{idx}_RET_MEAN_{window}",
                f"{idx}_RET_STD_{window}",
                f"{idx}_VOL_MEAN_{window}",
                f"{idx}_VOL_STD_{window}",
            ]
        )

LABEL_COL = "future_return"
EPS = 1e-12


def parse_segments(raw_segments: Sequence[str]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    segments: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for raw in raw_segments:
        try:
            name, start, end = raw.split(":")
        except ValueError as exc:
            raise ValueError(f"Invalid segment spec '{raw}'. Use name:start:end") from exc
        segments[name] = (pd.Timestamp(start), pd.Timestamp(end))
    return segments


def apply_purge_gap(
    segments: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]], purge_gap_days: int
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if purge_gap_days <= 0:
        return dict(segments)
    ordered = sorted(segments.items(), key=lambda kv: kv[1][0])
    delta = pd.Timedelta(days=int(purge_gap_days))
    adjusted: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for i, (name, (start, end)) in enumerate(ordered):
        s = start + delta if i > 0 else start
        e = end - delta if i < len(ordered) - 1 else end
        if s > e:
            raise ValueError(f"Segment {name} becomes empty after purge gap={purge_gap_days}")
        adjusted[name] = (s, e)
    return adjusted


def read_symbol_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"date", "company", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["company"] = df["company"].astype(str)
    return df


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def compute_linreg_stats(series: pd.Series, windows: Sequence[int]) -> Dict[int, Tuple[pd.Series, pd.Series, pd.Series]]:
    values = series.to_numpy(dtype=float)
    idx = series.index
    stats: Dict[int, Tuple[pd.Series, pd.Series, pd.Series]] = {}
    for window in windows:
        slopes = np.full_like(values, np.nan, dtype=float)
        rsqrs = np.full_like(values, np.nan, dtype=float)
        resis = np.full_like(values, np.nan, dtype=float)
        for i in range(len(values)):
            start = max(0, i - window + 1)
            seg = values[start : i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size < 2:
                continue
            x = np.arange(len(seg), dtype=float)
            x_mean = x.mean()
            y_mean = seg.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                continue
            slope = ((x - x_mean) * (seg - y_mean)).sum() / denom
            intercept = y_mean - slope * x_mean
            y_pred = slope * x + intercept
            ss_tot = ((seg - y_mean) ** 2).sum()
            ss_res = ((seg - y_pred) ** 2).sum()
            rsqr = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
            slopes[i] = slope
            rsqrs[i] = rsqr
            resis[i] = seg[-1] - y_pred[-1]
        stats[window] = (
            pd.Series(slopes, index=idx),
            pd.Series(rsqrs, index=idx),
            pd.Series(resis, index=idx),
        )
    return stats


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_fn(values: np.ndarray) -> float:
        if np.isnan(values[-1]):
            return np.nan
        arr = values[np.isfinite(values)]
        if arr.size == 0:
            return np.nan
        last = arr[-1]
        return (np.sum(arr <= last) - 0.5) / arr.size

    return series.rolling(window, min_periods=1).apply(rank_fn, raw=True)


def rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
    def idx_fn(values: np.ndarray) -> float:
        if np.all(np.isnan(values)):
            return np.nan
        return np.nanargmax(values) + 1

    return series.rolling(window, min_periods=1).apply(idx_fn, raw=True)


def rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
    def idx_fn(values: np.ndarray) -> float:
        if np.all(np.isnan(values)):
            return np.nan
        return np.nanargmin(values) + 1

    return series.rolling(window, min_periods=1).apply(idx_fn, raw=True)


def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    vwap = (open_ + high + low + close) / 4.0

    feature_store: Dict[str, pd.Series] = {}

    feature_store["KMID"] = safe_div(close - open_, open_)
    feature_store["KLEN"] = safe_div(high - low, open_)
    feature_store["KMID2"] = (close - open_) / (high - low + EPS)
    upper_body = high - np.maximum(open_, close)
    lower_body = np.minimum(open_, close) - low
    feature_store["KUP"] = safe_div(upper_body, open_)
    feature_store["KUP2"] = upper_body / (high - low + EPS)
    feature_store["KLOW"] = safe_div(lower_body, open_)
    feature_store["KLOW2"] = lower_body / (high - low + EPS)
    swing = 2 * close - high - low
    feature_store["KSFT"] = safe_div(swing, open_)
    feature_store["KSFT2"] = swing / (high - low + EPS)

    feature_store["OPEN0"] = safe_div(open_, close)
    feature_store["HIGH0"] = safe_div(high, close)
    feature_store["LOW0"] = safe_div(low, close)
    feature_store["VWAP0"] = safe_div(vwap, close)

    log_volume = np.log(volume.replace(0, np.nan) + 1)
    price_ratio = safe_div(close, close.shift(1))
    volume_ratio = safe_div(volume, volume.shift(1))
    volume_ratio = np.log(volume_ratio + 1)

    linreg_stats = compute_linreg_stats(close, ALPHA_WINDOWS)

    for window in ALPHA_WINDOWS:
        shifted_close = close.shift(window)
        feature_store[f"ROC{window}"] = safe_div(shifted_close, close)
        rolling_close = close.rolling(window, min_periods=1)
        feature_store[f"MA{window}"] = safe_div(rolling_close.mean(), close)
        feature_store[f"STD{window}"] = safe_div(rolling_close.std(), close)
        slope, rsqr, resi = linreg_stats[window]
        feature_store[f"BETA{window}"] = safe_div(slope, close)
        feature_store[f"RSQR{window}"] = rsqr
        feature_store[f"RESI{window}"] = safe_div(resi, close)
        feature_store[f"MAX{window}"] = safe_div(high.rolling(window, min_periods=1).max(), close)
        feature_store[f"MIN{window}"] = safe_div(low.rolling(window, min_periods=1).min(), close)
        feature_store[f"QTLU{window}"] = safe_div(close.rolling(window, min_periods=1).quantile(0.8), close)
        feature_store[f"QTLD{window}"] = safe_div(close.rolling(window, min_periods=1).quantile(0.2), close)
        feature_store[f"RANK{window}"] = rolling_rank(close, window)
        roll_max = high.rolling(window, min_periods=1).max()
        roll_min = low.rolling(window, min_periods=1).min()
        feature_store[f"RSV{window}"] = (close - roll_min) / (roll_max - roll_min + EPS)
        feature_store[f"IMAX{window}"] = rolling_idxmax(high, window) / window
        feature_store[f"IMIN{window}"] = rolling_idxmin(low, window) / window
        feature_store[f"IMXD{window}"] = (rolling_idxmax(high, window) - rolling_idxmin(low, window)) / window
        feature_store[f"CORR{window}"] = close.rolling(window, min_periods=2).corr(log_volume)
        feature_store[f"CORD{window}"] = price_ratio.rolling(window, min_periods=2).corr(volume_ratio)

        up_mask = close > close.shift(1)
        down_mask = close < close.shift(1)
        feature_store[f"CNTP{window}"] = up_mask.rolling(window, min_periods=1).mean()
        feature_store[f"CNTN{window}"] = down_mask.rolling(window, min_periods=1).mean()
        feature_store[f"CNTD{window}"] = feature_store[f"CNTP{window}"] - feature_store[f"CNTN{window}"]

        diff = close - close.shift(1)
        abs_diff = diff.abs()
        pos_diff = diff.clip(lower=0)
        neg_diff = (-diff).clip(lower=0)
        sum_abs = abs_diff.rolling(window, min_periods=1).sum() + EPS
        feature_store[f"SUMP{window}"] = pos_diff.rolling(window, min_periods=1).sum() / sum_abs
        feature_store[f"SUMN{window}"] = neg_diff.rolling(window, min_periods=1).sum() / sum_abs
        feature_store[f"SUMD{window}"] = (pos_diff.rolling(window, min_periods=1).sum() - neg_diff.rolling(window, min_periods=1).sum()) / sum_abs

        roll_volume = volume.rolling(window, min_periods=1)
        feature_store[f"VMA{window}"] = safe_div(roll_volume.mean(), volume + EPS)
        feature_store[f"VSTD{window}"] = safe_div(roll_volume.std(), volume + EPS)

        vol_vol = (price_ratio.sub(1).abs() * volume)
        mean_vol_vol = vol_vol.rolling(window, min_periods=1).mean() + EPS
        feature_store[f"WVMA{window}"] = vol_vol.rolling(window, min_periods=1).std() / mean_vol_vol

        vol_diff = volume - volume.shift(1)
        vol_abs = vol_diff.abs()
        vol_pos = vol_diff.clip(lower=0)
        vol_neg = (-vol_diff).clip(lower=0)
        vol_sum_abs = vol_abs.rolling(window, min_periods=1).sum() + EPS
        feature_store[f"VSUMP{window}"] = vol_pos.rolling(window, min_periods=1).sum() / vol_sum_abs
        feature_store[f"VSUMN{window}"] = vol_neg.rolling(window, min_periods=1).sum() / vol_sum_abs
        feature_store[f"VSUMD{window}"] = (vol_pos.rolling(window, min_periods=1).sum() - vol_neg.rolling(window, min_periods=1).sum()) / vol_sum_abs

    return pd.DataFrame(feature_store, index=df.index, dtype=np.float32)[ALPHA_FEATURE_NAMES]


def compute_label(close: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-horizon)
    next_day = close.shift(-1)
    return safe_div(future, next_day) - 1


def build_market_features(calendar: pd.DatetimeIndex, source: str) -> pd.DataFrame:
    features = pd.DataFrame(index=calendar)
    for symbol in MARKET_INDEXES:
        src_candidates = [source, "KBS", "VCI"]
        src_candidates = list(dict.fromkeys([s.upper() for s in src_candidates]))
        hist = None
        for src in src_candidates:
            try:
                quote = Quote(symbol=symbol, source=src, show_log=False)
                candidate = quote.history(start=str(calendar[0].date()), end=str(calendar[-1].date()), interval="1D")
                if candidate is not None and not candidate.empty:
                    hist = candidate
                    break
            except Exception:
                continue
        if hist is None or hist.empty:
            features[f"{symbol}_RET"] = 0.0
            for window in MARKET_WINDOWS:
                features[f"{symbol}_RET_MEAN_{window}"] = 0.0
                features[f"{symbol}_RET_STD_{window}"] = 0.0
                features[f"{symbol}_VOL_MEAN_{window}"] = 0.0
                features[f"{symbol}_VOL_STD_{window}"] = 0.0
            continue
        if "time" in hist.columns:
            hist = hist.rename(columns={"time": "date"})
        elif "date" not in hist.columns:
            hist = hist.reset_index().rename(columns={"index": "date"})
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.set_index("date").sort_index().reindex(calendar)
        hist["close"] = hist["close"].ffill()
        hist["volume"] = hist["volume"].ffill().fillna(0)
        ret = safe_div(hist["close"], hist["close"].shift(1)) - 1
        vol = hist["volume"]
        features[f"{symbol}_RET"] = ret
        for window in MARKET_WINDOWS:
            roll_ret = ret.rolling(window, min_periods=1)
            roll_vol = vol.rolling(window, min_periods=1)
            features[f"{symbol}_RET_MEAN_{window}"] = roll_ret.mean()
            features[f"{symbol}_RET_STD_{window}"] = roll_ret.std()
            features[f"{symbol}_VOL_MEAN_{window}"] = roll_vol.mean()
            features[f"{symbol}_VOL_STD_{window}"] = roll_vol.std()
    return features.fillna(0.0)


def build_symbol_frames(
    csv_frames: List[pd.DataFrame],
    calendar: pd.DatetimeIndex,
    market_features: pd.DataFrame,
    label_horizon: int,
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for df in csv_frames:
        symbol = df["company"].iloc[0]
        sdf = df.set_index("date").reindex(calendar).sort_index().ffill()
        alpha = build_alpha_features(sdf)
        label = compute_label(sdf["close"], label_horizon)
        combined = pd.concat([alpha, market_features, label.rename(LABEL_COL)], axis=1)
        frames[symbol] = combined
    return frames


def compute_robust_stats(
    frames: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    train_segment: Tuple[pd.Timestamp, pd.Timestamp],
) -> Tuple[pd.Series, pd.Series]:
    start, end = train_segment
    stacks = []
    for df in frames.values():
        stacks.append(df.loc[(df.index >= start) & (df.index <= end), feature_cols])
    train_data = pd.concat(stacks)
    median = train_data.median()
    mad = (train_data - median).abs().median()
    scale = 1.4826 * mad.replace(0, np.nan)
    scale = scale.fillna(1.0)
    return median, scale


def normalize_frames(
    frames: Dict[str, pd.DataFrame],
    cols: Sequence[str],
    median: pd.Series,
    scale: pd.Series,
) -> None:
    for symbol, df in frames.items():
        normalized = ((df[list(cols)] - median) / scale).clip(-3, 3).fillna(0.0).astype(np.float32)
        frames[symbol].loc[:, cols] = normalized
        frames[symbol][LABEL_COL] = df[LABEL_COL]


def build_segment_dataset(
    frames: Dict[str, pd.DataFrame],
    segment: Tuple[pd.Timestamp, pd.Timestamp],
    lookback: int,
    label_horizon: int,
    feature_cols: Sequence[str],
    market_cols: Sequence[str],
) -> MasterTensorDataset:
    start, end = segment
    samples: List[np.ndarray] = []
    index: List[Tuple[pd.Timestamp, str]] = []

    for symbol, sdf in frames.items():
        subset = sdf.loc[(sdf.index >= start) & (sdf.index <= end)]
        if subset.empty:
            continue
        for current_date in subset.index:
            loc = sdf.index.get_loc(current_date)
            if loc < lookback - 1:
                continue
            if (loc + label_horizon) >= len(sdf):
                continue
            if sdf.index[loc + label_horizon] > end:
                continue
            window = sdf.iloc[loc - lookback + 1 : loc + 1]
            label_value = window[LABEL_COL].iloc[-1]
            if pd.isna(label_value):
                continue
            array = window[list(feature_cols) + list(market_cols) + [LABEL_COL]].to_numpy(dtype=np.float32)
            samples.append(array)
            index.append((current_date, symbol))

    if not samples:
        raise ValueError(f"No samples generated for segment {start.date()}–{end.date()}")

    stacked = np.stack(samples, axis=0)
    multi_index = pd.MultiIndex.from_tuples(index, names=["datetime", "instrument"])
    metadata = DatasetMetadata(
        lookback=lookback,
        base_feature_names=feature_cols,
        market_feature_names=market_cols,
        label_name=LABEL_COL,
    )
    return MasterTensorDataset(stacked, multi_index, metadata)


def dump_dataset(dataset: MasterTensorDataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(dataset, fp)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert VN30 CSV files into MASTER-ready PKL datasets (Alpha158).")
    parser.add_argument("--csv-dir", type=Path, required=True, help="Directory containing per-symbol CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "vn30",
        help="Where to store the generated PKL files (default: data/vn30).",
    )
    parser.add_argument("--universe", type=str, default="vn30", help="Universe name used in output file names.")
    parser.add_argument(
        "--segments",
        nargs="+",
        default=[
            "train:2020-01-01:2023-12-31",
            "valid:2024-01-01:2024-06-30",
            "test:2024-07-01:2025-12-31",
        ],
        help="Date splits formatted as name:start:end (default covers VN30 crawl range).",
    )
    parser.add_argument("--lookback", type=int, default=8, help="Number of historical days per sample.")
    parser.add_argument("--label-horizon", type=int, default=5, help="Forward days used to compute the label.")
    parser.add_argument("--purge-gap-days", type=int, default=5, help="Gap in days between adjacent splits.")
    parser.add_argument("--market-source", type=str, default="KBS", help="vnstock data source for market indices.")
    parser.add_argument(
        "--expected-symbols",
        type=int,
        default=None,
        help="Fail if train split does not contain exactly this many instruments.",
    )
    parser.add_argument(
        "--min-symbols",
        type=int,
        default=None,
        help="Fail if train split contains fewer than this many instruments.",
    )

    args = parser.parse_args(argv)

    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {args.csv_dir}")

    symbol_frames_raw = [read_symbol_csv(path) for path in csv_files]
    calendar_union = set()
    for df in symbol_frames_raw:
        calendar_union.update(df["date"].tolist())
    calendar = pd.DatetimeIndex(sorted(calendar_union))
    if len(calendar) == 0:
        raise ValueError("No trading dates found from input CSV files.")
    market_features = build_market_features(calendar, args.market_source)
    symbol_frames = build_symbol_frames(symbol_frames_raw, calendar, market_features, args.label_horizon)

    segments_raw = parse_segments(args.segments)
    segments = apply_purge_gap(segments_raw, args.purge_gap_days)
    train_segment = segments.get("train")
    if train_segment is None:
        raise ValueError("Train segment must be provided to compute normalization statistics.")

    feature_cols = tuple(ALPHA_FEATURE_NAMES)
    market_cols = tuple(MARKET_FEATURE_NAMES)
    median, scale = compute_robust_stats(symbol_frames, feature_cols + market_cols, train_segment)
    normalize_frames(symbol_frames, feature_cols + market_cols, median, scale)

    for split, window in segments.items():
        dataset = build_segment_dataset(
            symbol_frames,
            window,
            args.lookback,
            args.label_horizon,
            feature_cols,
            market_cols,
        )
        out_path = args.output_dir / args.universe / f"{args.universe}_dl_{split}.pkl"
        dump_dataset(dataset, out_path)
        summary = dataset.summary()
        if split == "train":
            n_symbols = len(dataset.get_index().get_level_values("instrument").unique())
            if args.expected_symbols is not None and n_symbols != args.expected_symbols:
                raise ValueError(
                    f"{out_path} has {n_symbols} symbols, expected exactly {args.expected_symbols}"
                )
            if args.min_symbols is not None and n_symbols < args.min_symbols:
                raise ValueError(
                    f"{out_path} has {n_symbols} symbols, expected at least {args.min_symbols}"
                )
        print(
            f"[{split}] samples={summary['num_samples']}, "
            f"lookback={summary['lookback']}, features={summary['feature_columns']}"
        )
        print(
            f"    gate inputs span columns {summary['gate_feature_start']}–{summary['gate_feature_end'] - 1}"
        )


if __name__ == "__main__":
    main()

