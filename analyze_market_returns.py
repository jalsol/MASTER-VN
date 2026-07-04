#!/usr/bin/env python3
"""Cross-validate §4.1 label distribution claims against VN100 index market data.

Computes daily-return statistics, Sharpe ratios, max drawdown, tail asymmetry,
and yearly return decomposition for each training window. These market-level
metrics complement the stock-level label distribution stats in analysis_report.md.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Load VN100 equal-weighted index proxy ──
csv_dir = Path("raw_csv/vn100_10y")
symbols = sorted([f.stem for f in csv_dir.glob("*.csv") if f.stem.isupper()])
all_closes = []
for sym in symbols:
    df = pd.read_csv(csv_dir / f"{sym}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    all_closes.append(df["close"].rename(sym))

price_df = pd.concat(all_closes, axis=1)
price_df = price_df[price_df.count(axis=1) >= len(symbols) * 0.5]
vn100 = price_df.mean(axis=1)
vn100_ret = vn100.pct_change().dropna()

# ── Training periods ──
periods = {
    "3Y": ("2022-01-01", "2023-12-31"),
    "5Y": ("2020-01-01", "2023-12-31"),
    "10Y": ("2016-01-01", "2022-12-31"),
}

print("=" * 90)
print("MARKET-LEVEL TECHNICAL CROSS-VALIDATION OF §4.1 LABEL DISTRIBUTION")
print("=" * 90)

for name, (start, end) in periods.items():
    sub = vn100_ret.loc[start:end]
    n_days = len(sub)

    # Annualized stats
    mu = sub.mean() * 252
    sigma = sub.std() * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0

    # Cumulative return
    cum = (1 + sub).prod() - 1

    # Skewness and kurtosis of daily returns
    skew_daily = ((sub - sub.mean()) ** 3).mean() / (sub.std() ** 3 + 1e-12)
    kurt_daily = ((sub - sub.mean()) ** 4).mean() / (sub.std() ** 4 + 1e-12)

    # Daily return percentiles
    p01, p05, p10, p50, p90, p95, p99 = sub.quantile(
        [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    ).values

    # Tail asymmetry: right-tail magnitude / left-tail magnitude
    right_tail = abs(p99 - p50)
    left_tail = abs(p50 - p01)
    tail_asymmetry = right_tail / (left_tail + 1e-12)

    # Extreme event counts
    sigma_daily = sub.std()
    plus2sig = (sub > 2 * sigma_daily).sum()
    minus2sig = (sub < -2 * sigma_daily).sum()
    plus3sig = (sub > 3 * sigma_daily).sum()
    minus3sig = (sub < -3 * sigma_daily).sum()

    # Up/down day statistics
    up_days = (sub > 0).sum()
    down_days = (sub < 0).sum()
    avg_up = sub[sub > 0].mean() if up_days > 0 else 0
    avg_down = sub[sub < 0].mean() if down_days > 0 else 0
    win_loss_ratio = abs(avg_up / (avg_down + 1e-12))

    # Max drawdown
    cum_series = (1 + sub).cumprod()
    running_max = cum_series.expanding().max()
    dd = (cum_series - running_max) / running_max
    max_dd = dd.min()

    # Yearly returns
    yearly = sub.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    n_bull_years = (yearly > 0.10).sum()
    n_bear_years = (yearly < -0.10).sum()

    # Market-level 5-day forward return (for comparison with label mean)
    ret_5d = vn100.pct_change(5).shift(-5).dropna()
    sub_5d = ret_5d.loc[start:end]
    market_5d_mean = sub_5d.mean() if len(sub_5d) > 10 else np.nan

    print(f"\n── {name} ({start} → {end}, {n_days} trading days) ──")
    print(f"  Cumulative return:      {cum*100:+.1f}%")
    print(f"  Annualized μ / σ:       {mu*100:.1f}% / {sigma*100:.1f}%")
    print(f"  Sharpe ratio:           {sharpe:+.2f}")
    print(f"  Max drawdown:           {max_dd*100:.1f}%")
    print(f"  Daily return skew:      {skew_daily:+.3f}")
    print(f"  Daily return kurtosis:  {kurt_daily:.2f}")
    print(f"  Tail asymmetry (R/L):   {tail_asymmetry:.2f}  (>1 = right-tail heavier)")
    print(f"  ±2σ event counts:       +{plus2sig} up / -{minus2sig} down")
    print(f"  ±3σ event counts:       +{plus3sig} up / -{minus3sig} down")
    print(f"  Up/down days:           {up_days} ({up_days/n_days*100:.0f}%) / {down_days} ({down_days/n_days*100:.0f}%)")
    print(f"  Avg up / down day:      {avg_up*100:.2f}% / {avg_down*100:.2f}%")
    print(f"  Win/loss ratio:         {win_loss_ratio:.2f}")
    print(f"  Bull/bear years:        {n_bull_years} / {n_bear_years} (of {len(yearly)})")
    print(f"  Market 5d-fwd mean:     {market_5d_mean*100:.3f}%")
    if len(yearly) > 0:
        print(f"  Yearly returns:         ", end="")
        for y, r in yearly.items():
            print(f"{y.year}: {r*100:+.1f}%  ", end="")
        print()

print()
print("=" * 90)
print("NOTE: Daily market return skew ≠ label (stock-level 5d-fwd) skew.")
print("The label distribution is computed cross-sectionally across stocks at a")
print("5-day horizon, which produces different skew/kurtosis than the index-level")
print("1-day return distribution. Both are reported for completeness.")
print("=" * 90)
