import os
import pickle
from pathlib import Path

import numpy as np

from master_lgbm_leaf_input import MASTERLGBMLeafInputModel

os.environ.setdefault("BACKTEST_MARKET", "vn")
os.environ.setdefault("BACKTEST_HORIZON_DAYS", "10")
os.environ.setdefault("BACKTEST_RETURN_HORIZON_DAYS", "5")
os.environ.setdefault("BACKTEST_TOPK", "30")
os.environ.setdefault("BACKTEST_NDROP", "2")
os.environ.setdefault("BACKTEST_USE_INDEX_FEATURE_BENCHMARK", "0")
os.environ.setdefault("BACKTEST_BENCHMARK_GROUP", "0")
os.environ.setdefault("BACKTEST_IMPACT_COST", "0.002")

data_dir = Path("data/vn100/vn100")
with open(data_dir / "vn100_dl_train.pkl", "rb") as f:
    dl_train = pickle.load(f)
with open(data_dir / "vn100_dl_valid.pkl", "rb") as f:
    dl_valid = pickle.load(f)
with open(data_dir / "vn100_dl_test.pkl", "rb") as f:
    dl_test = pickle.load(f)

ic_list, icir_list, ric_list, ricir_list, ar_list, ir_list = [], [], [], [], [], []

for seed in range(5):
    print(f"\n{'=' * 80}")
    print(f"VN100 MASTER LGBM-LeafInput - Seed {seed}")
    print(f"{'=' * 80}")

    model = MASTERLGBMLeafInputModel(
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.5,
        S_dropout_rate=0.5,
        beta=5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        n_epochs=40,
        lr=1e-5,
        GPU=0,
        seed=seed,
        train_stop_loss_thred=0.95,
        save_path="model",
        save_prefix="vn100_lgbm_leaf_input",
        ffn_expand=1,
        lgbm_num_leaves=63,
        lgbm_learning_rate=0.03,
        lgbm_n_estimators=500,
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
        static_feature_end=158,
        embedding_stage="pre_spatial",
        leaf_integration_mode="add",
        use_amp=False,
    )

    model.fit(dl_train, dl_valid)
    _, metrics = model.predict(dl_test)

    print(f"\nSeed {seed} Results: {metrics}")

    ic_list.append(metrics["IC"])
    icir_list.append(metrics["ICIR"])
    ric_list.append(metrics["RIC"])
    ricir_list.append(metrics["RICIR"])
    ar_list.append(metrics["AR"])
    ir_list.append(metrics["IR"])

print(f"\n{'=' * 80}")
print("VN100 MASTER LGBM-LEAFINPUT - FINAL RESULTS (5 seeds)")
print(f"{'=' * 80}")
print(f"IC:     {np.nanmean(ic_list):.4f} ± {np.nanstd(ic_list):.4f}")
print(f"ICIR:   {np.nanmean(icir_list):.4f} ± {np.nanstd(icir_list):.4f}")
print(f"RIC:    {np.nanmean(ric_list):.4f} ± {np.nanstd(ric_list):.4f}")
print(f"RICIR:  {np.nanmean(ricir_list):.4f} ± {np.nanstd(ricir_list):.4f}")
print(f"AR:     {np.nanmean(ar_list):.4f} ± {np.nanstd(ar_list):.4f}")
print(f"IR:     {np.nanmean(ir_list):.4f} ± {np.nanstd(ir_list):.4f}")
