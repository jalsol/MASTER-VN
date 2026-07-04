#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_paths import LEGACY_VN100_PATH, RAW_CSV_VN100, UNIVERSE_EXPECTED_SYMBOLS, VN100_LB8


def copy_vn100_csvs(source_dir: Path, dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(source_dir.glob("*.csv")):
        if not src.stem.isupper():
            continue
        dst = dest_dir / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
        copied.append(dst)
    if not copied:
        raise FileNotFoundError(f"No stock CSV files found under {source_dir}")
    return copied


def validate_dataset(data_root: Path, prefix: str, min_symbols: int) -> None:
    train_path = data_root / f"{prefix}_dl_train.pkl"
    with train_path.open("rb") as fp:
        dataset = pickle.load(fp)
    symbols = dataset.get_index().get_level_values("instrument").unique()
    n_symbols = len(symbols)
    if n_symbols < min_symbols:
        raise ValueError(f"{train_path} has only {n_symbols} symbols, expected at least {min_symbols}")
    lookback = dataset.summary()["lookback"]
    print(
        f"validated {train_path}: symbols={n_symbols}, lookback={lookback}, "
        f"samples={len(dataset)}"
    )


def symlink_or_copy_legacy_vn100() -> None:
    if VN100_LB8.exists():
        return
    if not LEGACY_VN100_PATH.exists():
        raise FileNotFoundError(f"Legacy VN100 dataset missing at {LEGACY_VN100_PATH}")
    VN100_LB8.parent.mkdir(parents=True, exist_ok=True)
    VN100_LB8.symlink_to(LEGACY_VN100_PATH.resolve(), target_is_directory=True)
    print(f"linked {VN100_LB8} -> {LEGACY_VN100_PATH}")


def build_dataset(csv_dir: Path, output_dir: Path, universe: str, lookback: int) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "build_master_dataset.py"),
        "--csv-dir",
        str(csv_dir),
        "--output-dir",
        str(output_dir.parent),
        "--universe",
        universe,
        "--lookback",
        str(lookback),
        "--min-symbols",
        "90",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage VN100 CSVs and build/link canonical universe_vn100_lb8 dataset."
    )
    parser.add_argument(
        "--csv-source",
        type=Path,
        default=REPO_ROOT,
        help="Directory containing VN100 CSV files (default: repo root).",
    )
    parser.add_argument(
        "--link-legacy",
        action="store_true",
        help="Symlink data/universe_vn100_lb8/vn100 to existing data/vn100/vn100.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only copy CSVs into raw_csv/vn100.",
    )
    args = parser.parse_args()

    copied = copy_vn100_csvs(args.csv_source, RAW_CSV_VN100)
    print(f"copied {len(copied)} CSV files to {RAW_CSV_VN100}")

    if args.link_legacy:
        symlink_or_copy_legacy_vn100()
        validate_dataset(VN100_LB8, prefix="vn100", min_symbols=90)
        return

    if args.skip_build:
        return

    build_dataset(RAW_CSV_VN100, VN100_LB8, universe="vn100", lookback=8)
    validate_dataset(VN100_LB8, prefix="vn100", min_symbols=90)
    print(f"VN100 lb8 dataset ready at {VN100_LB8}")


if __name__ == "__main__":
    main()
