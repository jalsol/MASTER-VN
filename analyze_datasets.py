#!/usr/bin/env python3
"""Deep-dive: explain VN100 model performance patterns with data properties."""
import pickle
from pathlib import Path
import numpy as np

datasets = {}
# 10-year
for split in ["train", "valid", "test"]:
    p = Path(f"data/vn100_10y/vn100_10y/vn100_10y_dl_{split}.pkl")
    with open(p, "rb") as f:
        datasets[f"10y_{split}"] = pickle.load(f)
# 5-year (legacy)
for split in ["train", "valid", "test"]:
    p = Path(f"data/vn100/vn100/vn100_dl_{split}.pkl")
    with open(p, "rb") as f:
        datasets[f"5y_{split}"] = pickle.load(f)

print("=" * 100)
print("KEY INSIGHT #1: GATE FEATURE (market data) QUALITY")
print("=" * 100)

for name in ["10y", "5y"]:
    train = datasets[f"{name}_train"]._data
    gate = train[:, -1, 158:221]  # last timestep, columns 158-220
    alpha = train[:, -1, :158]

    gate_active = np.mean(np.abs(gate) > 1e-8)
    gate_nonzero_cols = np.sum(np.std(gate, axis=0) > 1e-8)

    print(f"\n{name.upper()} Gate Features (63 market-derived features):")
    print(f"  Non-zero fraction: {gate_active:.4f} ({gate_active*100:.1f}%)")
    print(f"  Columns with variance > 0: {gate_nonzero_cols}/63")
    print(f"  Mean abs: {np.mean(np.abs(gate)):.6f}, Std: {np.mean(np.std(gate, axis=0)):.6f}")
    print(f"  Gate feature max abs: {np.max(np.abs(gate)):.4f}")

    if gate_nonzero_cols > 0:
        # Check which market features are alive
        gate_stds = np.std(gate, axis=0)
        alive = np.where(gate_stds > 1e-8)[0]
        print(f"  Alive indices: {alive.tolist()}")
        # Feature names for alive indices
        feature_names = datasets[f"{name}_train"].metadata.market_feature_names
        print(f"  Alive features: {[feature_names[i] for i in alive]}")
    else:
        print(f"  ⚠️ ALL GATE FEATURES ARE DEAD (zero variance)")

print("\n" + "=" * 100)
print("KEY INSIGHT #2: LABEL DISTRIBUTION — tail risk & predictability")
print("=" * 100)

for name in ["10y", "5y"]:
    train = datasets[f"{name}_train"]._data
    labels = train[:, -1, -1]
    labels = labels[~np.isnan(labels)]

    # Tail analysis
    p01, p05, p10, p50, p90, p95, p99 = np.percentile(labels, [1, 5, 10, 50, 90, 95, 99])
    tail_ratio = (p99 - p01) / (p95 - p05 + 1e-12)  # how fat are the extreme tails

    print(f"\n{name.upper()}:")
    print(f"  Percentiles: P1={p01:.4f} P5={p05:.4f} P10={p10:.4f} P50={p50:.4f} "
          f"P90={p90:.4f} P95={p95:.4f} P99={p99:.4f}")
    print(f"  Tail fatness ratio (P99-P1)/(P95-P5): {tail_ratio:.2f}")
    print(f"  Skewness: {np.mean((labels-np.mean(labels))**3)/(np.std(labels)**3+1e-12):.3f}")
    print(f"  Signal-to-noise: |mean|/std = {np.abs(np.mean(labels))/np.std(labels):.4f}")

print("\n" + "=" * 100)
print("KEY INSIGHT #3: REGIME DIVERSITY — yearly market returns")
print("=" * 100)

for name in ["10y", "5y"]:
    train = datasets[f"{name}_train"]._data
    idx = datasets[f"{name}_train"].get_index()
    dates = idx.get_level_values("datetime")
    vn100_ret = train[:, -1, 179]  # VN100_RET

    print(f"\n{name.upper()} Train — VN100 yearly returns:")
    years = sorted(set(d.year for d in dates))
    for year in years:
        mask = np.array([d.year == year for d in dates])
        yr_data = vn100_ret[mask]
        yr_data = yr_data[~np.isnan(yr_data)]
        if len(yr_data) < 500:
            continue
        cum = np.prod(1 + yr_data) - 1
        vol = np.std(yr_data) * np.sqrt(252)
        regime = "🐂 BULL" if cum > 0.15 else ("🐻 BEAR" if cum < -0.10 else "↔️ SIDEWAYS")
        print(f"  {year}: cum_return={cum*100:+.1f}%  ann_vol={vol*100:.1f}%  regime={regime}")

print("\n" + "=" * 100)
print("KEY INSIGHT #4: CROSS-SECTIONAL ALPHA OPPORTUNITY")
print("=" * 100)

for name in ["10y", "5y"]:
    train_ds = datasets[f"{name}_train"]
    data = train_ds._data
    idx = train_ds.get_index()
    dates = idx.get_level_values("datetime").unique()

    # Cross-sectional std of labels per day → measures stock differentiation
    cs_stds = []
    cs_ranges = []  # P95-P5 per day
    for date in dates:
        mask = idx.get_level_values("datetime") == date
        day_labels = data[mask, -1, -1]
        day_labels = day_labels[~np.isnan(day_labels)]
        if len(day_labels) > 20:
            cs_stds.append(np.std(day_labels))
            cs_ranges.append(np.percentile(day_labels, 95) - np.percentile(day_labels, 5))

    mean_cs_std = np.mean(cs_stds)
    mean_cs_range = np.mean(cs_ranges)

    # Also compute: fraction of days where top-bottom decile spread is large
    high_dispersion = np.mean(np.array(cs_ranges) > 0.05)

    print(f"\n{name.upper()}:")
    print(f"  Mean cross-sectional std of labels: {mean_cs_std:.5f}")
    print(f"  Mean P95-P5 spread: {mean_cs_range:.5f}")
    print(f"  % days with P95-P5 > 5%: {100*high_dispersion:.1f}%")
    print(f"  → {'High' if mean_cs_std > 0.03 else 'Moderate'} alpha dispersion")

print("\n" + "=" * 100)
print("KEY INSIGHT #5: FEATURE-LABEL RELATIONSHIP STABILITY")
print("=" * 100)

for name in ["10y", "5y"]:
    train = datasets[f"{name}_train"]._data
    mid = len(train) // 2
    first, second = train[:mid], train[mid:]

    def get_corrs(subset):
        feats = subset[:, -1, :158]
        labels = subset[:, -1, -1]
        valid = ~np.isnan(labels)
        return np.array([np.corrcoef(feats[valid, j], labels[valid])[0, 1]
                        for j in range(158)])

    c1, c2 = get_corrs(first), get_corrs(second)
    corr_stab = np.corrcoef(c1, c2)[0, 1]
    rank_stab = np.corrcoef(np.argsort(np.abs(c1)), np.argsort(np.abs(c2)))[0, 1]
    sign_flip = np.mean(c1 * c2 < 0)

    # Top features by mean abs correlation
    mean_abs_corr = (np.abs(c1) + np.abs(c2)) / 2
    top5 = np.argsort(mean_abs_corr)[-5:][::-1]

    print(f"\n{name.upper()}:")
    print(f"  Feature-label corr stability (Pearson r): {corr_stab:.4f}")
    print(f"  Feature importance rank stability: {rank_stab:.4f}")
    print(f"  Sign flip fraction: {sign_flip:.3f}")
    print(f"  Top-5 most predictive feature indices: {top5.tolist()}")

print("\n" + "=" * 100)
print("KEY INSIGHT #6: GATE FEATURE INFORMATION CONTENT")
print("=" * 100)

for name in ["10y", "5y"]:
    train = datasets[f"{name}_train"]._data
    gate = train[:, -1, 158:221]
    labels = train[:, -1, -1]
    valid = ~np.isnan(labels)
    gate = gate[valid]
    labels = labels[valid]

    # How well can each gate feature predict the label?
    gate_label_corrs = np.array([np.corrcoef(gate[:, j], labels)[0, 1] for j in range(63)])
    gate_label_corrs = np.nan_to_num(gate_label_corrs, nan=0.0)
    top_gate = np.argsort(np.abs(gate_label_corrs))[-5:][::-1]

    print(f"\n{name.upper()}:")
    print(f"  Gate→Label max |corr|: {np.max(np.abs(gate_label_corrs)):.4f}")
    print(f"  Gate→Label mean |corr|: {np.mean(np.abs(gate_label_corrs)):.4f}")
    if np.max(np.abs(gate_label_corrs)) > 1e-8:
        fnames = datasets[f"{name}_train"].metadata.market_feature_names
        print(f"  Top-5 informative gate features:")
        for i in top_gate:
            print(f"    [{i}] {fnames[i]}: corr={gate_label_corrs[i]:.4f}")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
