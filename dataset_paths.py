from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RAW_CSV_VN30 = REPO_ROOT / "raw_csv" / "vn30"
RAW_CSV_VN100 = REPO_ROOT / "raw_csv" / "vn100"

VN30_LB8 = REPO_ROOT / "data" / "universe_vn30_lb8" / "vn30"
VN100_LB8 = REPO_ROOT / "data" / "universe_vn100_lb8" / "vn100"
VN30_LB30 = REPO_ROOT / "data" / "vn30_t30" / "vn30_t30"

DEPRECATED_VN100_AT_VN30_PATH = REPO_ROOT / "data" / "vn30" / "vn30"
LEGACY_VN100_PATH = REPO_ROOT / "data" / "vn100" / "vn100"

UNIVERSE_EXPECTED_SYMBOLS = {
    "vn30": 30,
    "vn100": 100,
}


def resolve_pkl_paths(data_root: Path, prefix: str):
    data_root = Path(data_root)
    return (
        data_root / f"{prefix}_dl_train.pkl",
        data_root / f"{prefix}_dl_valid.pkl",
        data_root / f"{prefix}_dl_test.pkl",
    )


def default_data_root(universe: str, prefix: str | None = None) -> Path:
    prefix = prefix or universe
    if universe == "vn30" and prefix == "vn30":
        return VN30_LB8
    if universe == "vn100" and prefix == "vn100":
        if VN100_LB8.exists():
            return VN100_LB8
        return LEGACY_VN100_PATH
    if universe == "vn30_t30" and prefix == "vn30_t30":
        return VN30_LB30
    return REPO_ROOT / "data" / universe / prefix
