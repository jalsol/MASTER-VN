from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetMetadata:
    """Lightweight metadata container bundled with each dataset."""

    lookback: int
    base_feature_names: Sequence[str]
    market_feature_names: Sequence[str]
    label_name: str = "label"

    def __post_init__(self) -> None:
        self.base_feature_names = tuple(self.base_feature_names)
        self.market_feature_names = tuple(self.market_feature_names)

    @property
    def gate_feature_start(self) -> int:
        return len(self.base_feature_names)

    @property
    def gate_feature_end(self) -> int:
        return len(self.base_feature_names) + len(self.market_feature_names)

    @property
    def feature_dim(self) -> int:
        return len(self.base_feature_names) + len(self.market_feature_names)


class MasterTensorDataset:
    """
    Minimal dataset container that mimics the interface expected by MASTER's
    training loop (see `base_model.SequenceModel`).

    Each sample is a tensor with shape ``(lookback, feature_dim + 1)`` where
    the last column stores the label.  Remaining columns contain the stacked
    per-stock features followed by the market information features that are
    used by MASTER's gating module.
    """

    def __init__(
        self,
        data: np.ndarray,
        index: pd.MultiIndex,
        metadata: DatasetMetadata,
    ) -> None:
        if data.ndim != 3:
            raise ValueError("`data` must be 3-dimensional (samples, T, F)")
        if len(index) != data.shape[0]:
            raise ValueError("`index` length must match number of samples")
        if not isinstance(index, pd.MultiIndex):
            raise TypeError("`index` must be a pandas.MultiIndex")

        self._data = data.astype(np.float32, copy=False)
        self._index = index
        self.metadata = metadata

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._data[idx]

    def get_index(self) -> pd.MultiIndex:
        return self._index

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._data.shape

    @property
    def feature_dim(self) -> int:
        # subtract the label column
        return self._data.shape[-1] - 1

    @property
    def gate_feature_bounds(self) -> Tuple[int, int]:
        return (
            self.metadata.gate_feature_start,
            self.metadata.gate_feature_end,
        )

    def summary(self) -> dict:
        num_samples, lookback, feat = self._data.shape
        return {
            "num_samples": num_samples,
            "lookback": lookback,
            "feature_columns": feat - 1,
            "label_column": self.metadata.label_name,
            "gate_feature_start": self.metadata.gate_feature_start,
            "gate_feature_end": self.metadata.gate_feature_end,
            "base_features": list(self.metadata.base_feature_names),
            "market_features": list(self.metadata.market_feature_names),
        }

    def __repr__(self) -> str:
        summary = self.summary()
        return (
            f"MasterTensorDataset(samples={summary['num_samples']}, "
            f"lookback={summary['lookback']}, "
            f"features={summary['feature_columns']})"
        )

