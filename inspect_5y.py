#!/usr/bin/env python3
"""Deep inspection of 5Y dataset gate features — why are they all zeros?"""
import pickle
from pathlib import Path
import numpy as np

# Load both datasets
for split in ["train", "valid", "test"]:
    for name, dir_ in [("10y", "data/vn100_10y/vn100_10y"), ("5y", "data/vn100/vn100")]:
        p = Path(dir_) / f"{dir_.split('/')[-1]}_dl_{split}.pkl"
        with open(p, "rb") as f:
            ds = pickle.load(f)
        globals()[f"{name}_{split}_ds"] = ds
        globals()[f"{name}_{split}"] = ds._data

print("=" * 90)
print("5Y DATASET DEEP INSPECTION")
print("=" * 90)

# 1. Check raw gate features BEFORE normalization
for name in ["10y", "5y"]:
    train_ds = globals()[f"{name}_train_ds"]
    train_data = globals()[f"{name}_train"]
    gate = train_data[:, -1, 158:221]

    print(f"\n── {name.upper()} Gate Features ──")
    print(f"  Shape: {gate.shape}")
    print(f"  Min: {np.min(gate):.6f}  Max: {np.max(gate):.6f}")
    print(f"  Fraction exactly zero: {np.mean(np.abs(gate) < 1e-12):.4f}")
    print(f"  Fraction near zero (|x|<0.001): {np.mean(np.abs(gate) < 0.001):.4f}")

    # Per-column stats
    col_stds = np.std(gate, axis=0)
    col_means = np.mean(np.abs(gate), axis=0)
    for j in range(63):
        if col_stds[j] > 1e-8:
            print(f"  Col[{j}] std={col_stds[j]:.6f} mean_abs={col_means[j]:.6f}")
    dead = np.sum(col_stds < 1e-8)
    print(f"  Dead columns: {dead}/63")

# 2. Check the normalization parameters
print("\n" + "=" * 90)
print("CHECKING: Are market features normalized away?")
print("=" * 90)

for name in ["10y", "5y"]:
    train_data = globals()[f"{name}_train"]
    gate = train_data[:, -1, 158:221]
    alpha = train_data[:, -1, :158]

    print(f"\n{name.upper()}:")
    print(f"  Alpha features: min={np.min(alpha):.3f} max={np.max(alpha):.3f} "
          f"mean_abs={np.mean(np.abs(alpha)):.4f} %nonzero={100*np.mean(np.abs(alpha)>1e-8):.1f}%")
    print(f"  Gate  features: min={np.min(gate):.6f} max={np.max(gate):.6f} "
          f"mean_abs={np.mean(np.abs(gate)):.6f} %nonzero={100*np.mean(np.abs(gate)>1e-8):.1f}%")

    # Check if gate features are all the SAME value (e.g. all 0.0)
    unique_vals_per_col = [len(np.unique(np.round(gate[:, j], 6))) for j in range(63)]
    print(f"  Unique values per gate column: min={min(unique_vals_per_col)} max={max(unique_vals_per_col)}")
    dead_cols = [j for j in range(63) if unique_vals_per_col[j] <= 1]
    print(f"  Columns with ≤1 unique value: {dead_cols}")

# 3. Check the metadata
print("\n" + "=" * 90)
print("DATASET METADATA COMPARISON")
print("=" * 90)
for name in ["10y", "5y"]:
    ds = globals()[f"{name}_train_ds"]
    meta = ds.metadata
    idx = ds.get_index()
    print(f"\n{name.upper()}:")
    print(f"  Lookback: {meta.lookback}")
    print(f"  Base features: {len(meta.base_feature_names)}")
    print(f"  Market features: {len(meta.market_feature_names)}")
    print(f"  Symbols: {len(idx.get_level_values('instrument').unique())}")
    print(f"  Date range: {idx.get_level_values('datetime').min()} → {idx.get_level_values('datetime').max()}")

# 4. Check market data across all splits (train+valid+test)
print("\n" + "=" * 90)
print("5Y: GATE FEATURES ACROSS ALL SPLITS")
print("=" * 90)
for split in ["train", "valid", "test"]:
    data = globals()[f"5y_{split}"]
    gate = data[:, -1, 158:221]
    nz = np.mean(np.abs(gate) > 1e-8)
    print(f"  {split}: nonzero={nz:.4f} min={np.min(gate):.6f} max={np.max(gate):.6f} "
          f"unique_total={len(np.unique(np.round(gate, 6)))}")

# 5. Compare with 10Y splits
print("\n" + "=" * 90)
print("10Y: GATE FEATURES ACROSS ALL SPLITS (reference)")
print("=" * 90)
for split in ["train", "valid", "test"]:
    data = globals()[f"10y_{split}"]
    gate = data[:, -1, 158:221]
    nz = np.mean(np.abs(gate) > 1e-8)
    print(f"  {split}: nonzero={nz:.4f} min={np.min(gate):.3f} max={np.max(gate):.3f}")

print("\nDONE")
