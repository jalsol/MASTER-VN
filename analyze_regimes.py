#!/usr/bin/env python3
"""Quantify regime diversity across 3Y/5Y/10Y VN100 datasets using technical analysis."""

import pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
from scipy import stats as scipy_stats

# ── Load VN100 price data from crawled CSVs ──
csv_dir = Path("raw_csv/vn100_10y")
symbols = sorted([f.stem for f in csv_dir.glob("*.csv") if f.stem.isupper()])

# Build a proxy for VN100 index: equal-weighted average of all stock close prices
all_closes = []
for sym in symbols:
    df = pd.read_csv(csv_dir / f"{sym}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    all_closes.append(df["close"].rename(sym))

price_df = pd.concat(all_closes, axis=1)
# Keep dates where at least 50% of stocks have data
price_df = price_df[price_df.count(axis=1) >= len(symbols) * 0.5]
vn100_proxy = price_df.mean(axis=1)  # equal-weighted index proxy
returns = vn100_proxy.pct_change().dropna()

print("=" * 100)
print("REGIME DIVERSITY QUANTIFICATION")
print("=" * 100)

# ── Define train periods ──
periods = {
    "3Y": ("2022-01-01", "2023-12-31"),
    "5Y": ("2020-01-01", "2023-12-31"),
    "10Y": ("2016-01-01", "2022-12-31"),
}

# ── 1. TREND REGIME: MA crossover state ──
print("\n" + "─" * 100)
print("1. TREND REGIMES — Price vs MA(50) and MA(200)")
print("─" * 100)

for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    cum = (1 + sub).cumprod()
    ma50 = cum.rolling(50, min_periods=1).mean()
    ma200 = cum.rolling(200, min_periods=10).mean()

    # Regime: 1=golden cross (MA50 > MA200, uptrend), 0=death cross (downtrend)
    uptrend = (ma50 > ma200).astype(int)
    changes = (uptrend.diff().abs() == 1).sum()
    regime_runs = []
    current_run = 1
    for i in range(1, len(uptrend)):
        if uptrend.iloc[i] == uptrend.iloc[i-1]:
            current_run += 1
        else:
            regime_runs.append(current_run)
            current_run = 1
    regime_runs.append(current_run)
    avg_duration = np.mean(regime_runs) if regime_runs else 0
    med_duration = np.median(regime_runs) if regime_runs else 0
    pct_uptrend = uptrend.mean() * 100

    # Entropy of trend regime distribution
    counts = Counter(uptrend.values)
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    print(f"  {name}: {len(sub):,} days, {changes} MA crossover events, "
          f"avg run={avg_duration:.0f}d, median run={med_duration:.0f}d, "
          f"uptrend={pct_uptrend:.0f}%, H(regime)={entropy:.3f}")

# ── 2. VOLATILITY REGIME: GARCH-like clustering ──
print("\n" + "─" * 100)
print("2. VOLATILITY REGIMES — Rolling 20d volatility quartile state")
print("─" * 100)

for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    rv20 = sub.rolling(20, min_periods=5).std() * np.sqrt(252)  # annualized

    # Quartile-based regime labels
    q25, q50, q75 = rv20.quantile([0.25, 0.50, 0.75]).values
    regimes = np.where(rv20 < q25, 0,  # low vol
                np.where(rv20 < q50, 1,  # med-low
                np.where(rv20 < q75, 2,  # med-high
                          3)))           # high vol

    changes = (np.diff(regimes) != 0).sum()
    regime_runs = []
    current_run = 1
    for i in range(1, len(regimes)):
        if regimes[i] == regimes[i-1]:
            current_run += 1
        else:
            regime_runs.append(current_run)
            current_run = 1
    regime_runs.append(current_run)
    avg_duration = np.mean(regime_runs)
    med_duration = np.median(regime_runs)

    counts = Counter(regimes)
    total = sum(counts.values())
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())

    # Vol-of-vol: how much does volatility itself fluctuate?
    vol_of_vol = rv20.std() / rv20.mean()

    print(f"  {name}: vol_range=[{rv20.min():.1%},{rv20.max():.1%}] vol_of_vol={vol_of_vol:.2f}, "
          f"transitions={changes}, avg_dur={avg_duration:.0f}d, med_dur={med_duration:.0f}d, "
          f"H(vol_regime)={entropy:.3f}")
    print(f"         regime distribution: low={counts[0]/total*100:.0f}% "
          f"med_lo={counts[1]/total*100:.0f}% med_hi={counts[2]/total*100:.0f}% high={counts[3]/total*100:.0f}%")

# ── 3. DRAWDOWN REGIME ──
print("\n" + "─" * 100)
print("3. DRAWDOWN REGIMES — Fraction of time in drawdown > X%")
print("─" * 100)

for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    cum = (1 + sub).cumprod()
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max

    # Regime levels
    regimes = np.where(drawdown > -0.05, 0,      # normal (-5% to 0%)
                np.where(drawdown > -0.15, 1,     # correction (-15% to -5%)
                np.where(drawdown > -0.30, 2,     # bear (-30% to -15%)
                          3)))                     # crash (< -30%)
    counts = Counter(regimes)
    total = sum(counts.values())
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())

    max_dd = drawdown.min()
    dd_changes = (np.diff(regimes) != 0).sum()

    print(f"  {name}: max_dd={max_dd:.1%}, dd_transitions={dd_changes}, "
          f"H(dd_regime)={entropy:.3f}")
    print(f"         normal(0-5%)={counts.get(0,0)/total*100:.0f}% "
          f"correction(5-15%)={counts.get(1,0)/total*100:.0f}% "
          f"bear(15-30%)={counts.get(2,0)/total*100:.0f}% "
          f"crash(>30%)={counts.get(3,0)/total*100:.0f}%")

# ── 4. COMBINED REGIME STATE ──
print("\n" + "─" * 100)
print("4. COMBINED REGIME STATE (Trend × Volatility interactions)")
print("─" * 100)

for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    cum = (1 + sub).cumprod()
    ma50 = cum.rolling(50, min_periods=1).mean()
    ma200 = cum.rolling(200, min_periods=10).mean()
    uptrend = (ma50 > ma200).astype(int)

    rv20 = sub.rolling(20, min_periods=5).std() * np.sqrt(252)
    q50 = rv20.quantile(0.50)
    high_vol = (rv20 > q50).astype(int)

    # 4 combined states: uptrend+lowvol, uptrend+highvol, downtrend+lowvol, downtrend+highvol
    combined = uptrend * 2 + high_vol  # 0=down/lo, 1=down/hi, 2=up/lo, 3=up/hi
    counts = Counter(combined)
    total = sum(counts.values())
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())

    labels = {0: "DOWN+LOWVOL", 1: "DOWN+HIGHVOL", 2: "UP+LOWVOL", 3: "UP+HIGHVOL"}
    transitions = (np.diff(combined.values) != 0).sum()

    print(f"  {name}: combined_states={len(counts)}, transitions={transitions}, "
          f"H(combined)={entropy:.3f} (max entropy=2.000)")
    for k in sorted(counts.keys()):
        print(f"         {labels[k]:>14s}: {counts[k]/total*100:5.1f}%")

# ── 5. SUMMARY: Composite Regime Diversity Score ──
print("\n" + "=" * 100)
print("5. COMPOSITE REGIME DIVERSITY SCORE")
print("=" * 100)

print(f"\n{'Metric':<40s} {'3Y':>12s} {'5Y':>12s} {'10Y':>12s}")
print("-" * 76)

scores = {}
for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    cum = (1 + sub).cumprod()

    # Trend
    ma50 = cum.rolling(50, min_periods=1).mean()
    ma200 = cum.rolling(200, min_periods=10).mean()
    trend_regimes = (ma50 > ma200).astype(int)
    trend_transitions = (trend_regimes.diff().abs() == 1).sum()

    # Vol
    rv20 = sub.rolling(20, min_periods=5).std() * np.sqrt(252)
    vol_quartiles = pd.qcut(rv20, 4, labels=False, duplicates='drop')
    vol_transitions = (vol_quartiles.diff().abs() > 0).sum()
    vol_of_vol = rv20.std() / rv20.mean() if rv20.mean() > 0 else 0

    # Drawdown
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    dd_bins = np.where(dd > -0.05, 0, np.where(dd > -0.15, 1, np.where(dd > -0.30, 2, 3)))
    dd_transitions = (np.diff(dd_bins) != 0).sum()
    dd_entropy = -sum((c/len(dd_bins))*np.log2(c/len(dd_bins))
                      for c in Counter(dd_bins).values())

    # Combined entropy
    combined = trend_regimes * 2 + (rv20 > rv20.quantile(0.5)).astype(int)
    combined_entropy = -sum((c/len(combined))*np.log2(c/len(combined))
                            for c in Counter(combined.values).values())

    # Volatility range
    vol_range = rv20.max() - rv20.min()

    # Return dispersion (yearly returns — measure of regime extremity)
    yearly_rets = sub.resample("YE").apply(lambda x: (1+x).prod() - 1)
    ret_dispersion = yearly_rets.std() * np.sqrt(len(yearly_rets)) if len(yearly_rets) > 1 else 0

    scores[name] = {
        "trend_transitions": trend_transitions,
        "vol_transitions": vol_transitions,
        "vol_of_vol": vol_of_vol,
        "dd_transitions": dd_transitions,
        "dd_entropy": dd_entropy,
        "combined_entropy": combined_entropy,
        "vol_range": vol_range,
        "ret_dispersion": ret_dispersion,
    }

# Print metrics
metrics = [
    ("Trend transitions (MA crossovers)", "trend_transitions", "d"),
    ("Volatility regime transitions", "vol_transitions", "d"),
    ("Volatility of volatility", "vol_of_vol", ".3f"),
    ("Drawdown regime transitions", "dd_transitions", "d"),
    ("Drawdown regime entropy", "dd_entropy", ".3f"),
    ("Combined regime entropy (H)", "combined_entropy", ".3f"),
    ("Volatility range (ann.)", "vol_range", ".1%"),
    ("Yearly return dispersion (σ)", "ret_dispersion", ".1%"),
]

for label, key, fmt in metrics:
    v3 = scores.get("3Y", {}).get(key, 0)
    v5 = scores.get("5Y", {}).get(key, 0)
    v10 = scores.get("10Y", {}).get(key, 0)
    if fmt == "d":
        print(f"{label:<40s} {v3:>12.0f} {v5:>12.0f} {v10:>12.0f}")
    elif fmt == ".3f":
        print(f"{label:<40s} {v3:>12.3f} {v5:>12.3f} {v10:>12.3f}")
    elif fmt == ".2%":
        print(f"{label:<40s} {v3:>11.1%} {v5:>11.1%} {v10:>11.1%}")
    else:
        print(f"{label:<40s} {v3:>12.2f} {v5:>12.2f} {v10:>12.2f}")

# ── 6. REGIME CHANGE DETECTION BY TECHNICAL INDICATORS ──
print("\n" + "=" * 100)
print("6. TECHNICAL REGIME DETECTION — Individual Indicators")
print("=" * 100)

for name, (start, end) in periods.items():
    sub = returns.loc[start:end]
    if len(sub) < 100:
        continue
    cum = (1 + sub).cumprod()
    price = cum.values
    n = len(price)

    # ADX-like trend strength
    plus_dm = np.maximum(np.diff(price, prepend=price[0]), 0)
    minus_dm = np.maximum(-np.diff(price, prepend=price[0]), 0)
    tr = np.maximum(np.maximum(
        np.abs(price - np.roll(price, 1)),
        np.abs(price - np.roll(price, 1))),
        np.abs(price - np.roll(price, -1)))
    tr[0] = tr[1] if len(tr) > 1 else 1
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    pdi14 = pd.Series(plus_dm).rolling(14, min_periods=1).mean().values / (atr14 + 1e-12)
    mdi14 = pd.Series(minus_dm).rolling(14, min_periods=1).mean().values / (atr14 + 1e-12)
    dx = np.abs(pdi14 - mdi14) / (pdi14 + mdi14 + 1e-12) * 100

    # Regime: trending (dx > 25) vs ranging (dx < 25)
    trending = (dx > 25).astype(int)
    trend_count = (np.diff(trending) != 0).sum()
    pct_trending = trending.mean() * 100

    # Bollinger Band width (volatility regime)
    bb_center = pd.Series(price).rolling(20, min_periods=5).mean()
    bb_width = pd.Series(price).rolling(20, min_periods=5).std() * 2
    bb_expanding = (bb_width / bb_width.shift(20)).dropna()
    vol_breakouts = (bb_expanding > 1.5).sum()  # vol expanded >50%

    # RSI overbought/oversold
    delta = pd.Series(price).diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rsi = 100 - 100/(1 + gain/(loss + 1e-12))
    overbought = (rsi > 70).sum()
    oversold = (rsi < 30).sum()

    print(f"\n  {name} ({len(sub):,} days):")
    print(f"    ADX: {pct_trending:.0f}% trending, {trend_count} trend/ranging switches")
    print(f"    BB: {vol_breakouts} volatility breakout days (BB width >1.5× 20d-ago)")
    print(f"    RSI: {overbought} overbought days, {oversold} oversold days")
    print(f"    Days per transition (lower = more frequent regime changes):")
    print(f"      Trend: {len(sub)/max(trend_count,1):.0f}d   Vol: {len(sub)/max(vol_quartiles.diff().abs().sum(),1):.0f}d")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
