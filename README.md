# Integrating the Attention Mechanism in Predicting the Vietnamese Stock Market

This repository accompanies the thesis *"Integrating the Attention Mechanism in Predicting the Vietnamese Stock Market"*, extending the [MASTER](https://arxiv.org/abs/2312.15235) (AAAI 2024) architecture to Vietnamese VN100 stocks across three training horizons.

## Contents

### Analysis
- `analysis_report.md` — Full thesis (270+ pages of markdown): seven attention architectures evaluated on 3/5/10-year VN100 data with regime diversity quantification.
- `analyze_datasets.py` — Label distribution, feature stability, and market information content analysis.
- `analyze_regimes.py` — Technical regime detection (MA crossovers, volatility quartiles, drawdowns, RSI, Bollinger Bands, ADX).
- `analysis_output.txt` — Raw analysis outputs.
- `inspect_5y.py` — Gate feature inspection across datasets.

### Models (7 attention architectures)
All share a transformer backbone (temporal self-attention + spatial cross-attention, 8-day lookback, 158 Alpha158 features). They differ only in how 63 market features modulate attention via a *gate*:

| File | Architecture | Gate Mechanism |
|------|-------------|----------------|
| `master.py` | Base MASTER | Linear attention mask `w = A·m + b` |
| `master_moe_gate.py` | MoE | 3 expert gates + softmax router |
| `master_lgbm_gate.py` | LGBM-Gate | LightGBM tree gate blended with neural prior |
| `master_bilstm.py` | BiLSTM | Bidirectional LSTM replaces temporal attention; linear gate |
| `master_lgbm_leaf_input.py` | LGBM-LeafInput | LGBM leaf embeddings injected into spatial attention |
| `master_cross_attn_gate.py` | Cross-Attn Gate | Market features query stock features via cross-attention |
| `master_moe_lgbm.py` | MoE+LGBM Gate | Tree-routed multi-expert attention |
| `base_model.py` | Shared backbone | SequenceModel base class (training loop, early stopping) |

### Analysis Scripts
Scripts used to compute all data properties in `analysis_report.md` (§4):

- `analyze_datasets.py` — Label distribution (§4.1), cross-sectional dispersion (§4.3), feature-label stability (§4.4), temporal autocorrelation (§4.5), and market feature information content (§4.6).
- `analyze_regimes.py` — Regime diversity quantification (§4.2): trend regimes (MA crossovers), volatility regimes (rolling 20d quartiles), drawdown regimes, combined state entropy, and technical indicator extremes (RSI, Bollinger Bands, ADX).
- `inspect_5y.py` — Gate feature inspection and comparison between datasets.

### Tools
- `crawl.py` — Crawl VN100 stock data via vnstock API with rate-limit-aware batching.
- `tools/build_master_dataset.py` — Convert CSV data to MASTER-format PKL datasets with Alpha158 features and market information.
- `tools/setup_vn100_lb8.py` — Stage CSVs and build canonical VN100 lb8 dataset.
- `dataset_paths.py` — Centralized path configuration.

### Data Module
- `datasets/master_dataset.py` — `MasterTensorDataset` and `DatasetMetadata` classes.

### Reference Materials
- `paper.pdf` / `2312.15235.pdf` — MASTER paper (AAAI 2024)
- `MASTER-poster.pdf`, `MASTER-slides.pdf`, `MASTER-supplementary-materials.pdf`
- `framework.png` — MASTER architecture diagram
- `qlib-update/` — Qlib configuration reference files
- `LICENSE`

## Data

Datasets are **not** included (several GB of PKL files). To reproduce:
1. Crawl VN100 data: `uv run python crawl.py --start-date 2016-01-01 --end-date 2025-12-31 --output-dir raw_csv/vn100_10y`
2. Build dataset: `uv run python tools/build_master_dataset.py --csv-dir raw_csv/vn100_10y --output-dir data/vn100_10y --universe vn100_10y --segments "train:2016-01-01:2022-12-31" "valid:2023-01-01:2023-12-31" "test:2024-01-01:2025-12-31"`
3. Run tests: `uv run python test_moe_lgbm_vn100_10y_5seeds.py`
