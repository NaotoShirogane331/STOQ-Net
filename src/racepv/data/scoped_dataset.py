"""Lazy candidate adapter over already verified schema3 leaf datasets.

The scientific view is separate from storage schemas. This adapter grants no
source qualification and does not materialize overlapping model windows.
"""

import json
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from racepv.data import prepared_contract as c
from racepv.data.scoped_scientific_view import _revalidate


class ScopedCandidateDataset(Dataset):
    def __init__(self, loaded, view):
        _revalidate(view)
        c._exact(
            loaded["metadata"], json.loads(view.input_metadata_json), "adapter loaded metadata"
        )
        for flag in ("content_verified", "array_semantics_verified"):
            c._exact(loaded["metadata"][flag], True, "adapter " + flag)
        self.view = view
        # The validated view is immutable; keep only this scalar for per-sample use.
        self._source_timezone = view.feature_signature["source_clock"]["timezone"]
        self._datasets = loaded["datasets"]
        self._audits = loaded["causal_arrays"]
        leaves = loaded["metadata"]["leaves"]
        if len(leaves) != len(self._datasets):
            raise ValueError("adapter leaf coverage differs")
        self._site_ids = tuple(leaf["schema"]["site"]["id"] for leaf in leaves)
        self._ends = np.cumsum([len(dataset) for dataset in self._datasets]).tolist()
        if not self._ends or self._ends[-1] == 0:
            raise ValueError("candidate needs nonempty windows")
        self.content_identity = loaded["metadata"]["aggregate_content_sha256"]

    @property
    def site_id_map(self):
        return self.view.site_id_map

    @property
    def feature_signature(self):
        return self.view.feature_signature

    @property
    def export_metadata(self):
        signature = self.feature_signature
        return dict(
            site_id_map=self.site_id_map,
            dataset_kind=signature["dataset_kind"],
            simulated=signature["simulated"],
            ramp_method=self.view.ramp_method,
        )

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)):
            raise TypeError("window index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        leaf_index = bisect_right(self._ends, index)
        local_index = index - (self._ends[leaf_index - 1] if leaf_index else 0)
        leaf = self._datasets[leaf_index]
        sample = leaf[local_index]
        audits = self._audits[leaf_index]
        timestamps = audits["timestamps"][local_index].copy()
        timezone = self._source_timezone
        local_dates = pd.to_datetime(timestamps, utc=True).tz_convert(timezone).strftime("%Y%m%d")
        target_mask = leaf.sample_target_observed_mask(local_index)
        if target_mask.dtype != np.dtype(bool) or target_mask.shape != sample["targets"].shape:
            raise ValueError("invalid target observation mask")
        return {
            **sample,
            "daylight_mask": torch.from_numpy(leaf.sample_daylight_mask(local_index)),
            "target_mask": torch.from_numpy(target_mask),
            "timestamps": torch.from_numpy(timestamps),
            "target_interval_start_ns": torch.from_numpy(timestamps.copy()),
            "target_local_day_ids": torch.from_numpy(np.asarray(local_dates, dtype=np.int64)),
            "issue_ns": torch.tensor(int(audits["issue_ns"][local_index]), dtype=torch.int64),
            "day_ids": torch.from_numpy(audits["day_ids"][local_index].copy()),
            "site_ids": torch.tensor(self._site_ids[leaf_index], dtype=torch.int64),
        }
