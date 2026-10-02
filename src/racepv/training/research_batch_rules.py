"""Approved B0 reference statistics; deliberately separate from legacy persistence."""

import numpy as np
import pandas as pd

from racepv.training.research_baseline_bridge import _inputs, _numpy, _signature

STEP_NS = 900_000_000_000


class ProtocolSmartPersistence:
    """Train-only raw paired observations; no optimizer and no validation fitting."""

    def __init__(self, signature):
        history, future, _ = _signature(signature)
        if signature["clock"]["issue"] != "last_history_interval_end":
            raise ValueError("B0 requires last-history-interval-end issue clock")
        self.p = history.index("power_norm")
        self.c = history.index("clear_sky_power_norm")
        self.f = future.index("clear_sky_power_norm")
        self.groups = {}
        self.support_counts = {}
        self.fallback_counts = dict(history=0, site_hour=0, site_global=0)

    def _pairs(self, batch):
        inputs = _inputs(batch)
        h, mask = _numpy(inputs["history_values"]), _numpy(inputs["history_mask"])
        sites, issues = _numpy(batch["site_ids"]), _numpy(batch["issue_ns"])
        if sites.shape != (len(h),) or issues.shape != sites.shape or sites.dtype != np.int64 or issues.dtype != np.int64:
            raise ValueError("B0 requires int64 site and issue keys")
        # Stored observations are left-labelled intervals; the last starts issue-step.
        times = issues[:, None] - np.arange(192, 0, -1, dtype=np.int64)[None] * STEP_NS
        hours = np.asarray(pd.to_datetime(times.reshape(-1), utc=True).tz_convert("Australia/Darwin").hour).reshape(times.shape)
        return h, mask, sites, times, hours

    def fit(self, batches, *, split):
        if split != "train":
            raise ValueError("B0 fallback requires train only")
        records, sites_seen = {}, set()
        for batch in batches:
            h, mask, sites, times, hours = self._pairs(batch)
            for row, site in enumerate(sites):
                site = int(site)
                sites_seen.add(site)
                for index, stamp in enumerate(times[row]):
                    pm, cm = bool(mask[row, index, self.p]), bool(mask[row, index, self.c])
                    p = float(h[row, index, self.p]) if pm else None
                    c = float(h[row, index, self.c]) if cm else None
                    value = (pm, cm, p, c, int(hours[row, index]))
                    key = site, int(stamp)
                    if key in records and records[key] != value:
                        raise ValueError("conflicting raw observations at one physical site/time")
                    records[key] = value
        return self._finalize(records, sites_seen)

    def fit_store(self, store, *, split, chunk_size=8192):
        """Fit every unique raw row of the already-authorized complete train split."""
        if split != "train":
            raise ValueError("B0 fallback requires train only")
        records, sites_seen = {}, set()
        for leaf, arrays in zip(store["metadata"]["leaves"], store["causal_arrays"], strict=True):
            if leaf["schema"]["split"] != "train":
                raise ValueError("B0 raw store is not train")
            site = int(leaf["schema"]["site"]["id"])
            sites_seen.add(site)
            for start in range(0, len(arrays["labels_ns"]), chunk_size):
                stop = start + chunk_size
                times = arrays["labels_ns"][start:stop]
                power = arrays["base_values_raw"][start:stop, self.p]
                clear = arrays["clear_sky_norm"][start:stop]
                observed = arrays["power_observed"][start:stop]
                if not np.isin(observed, (0, 1)).all():
                    raise ValueError("raw power mask must be binary")
                observed = observed.astype(bool)
                if not np.isfinite(power[observed]).all():
                    raise ValueError("nonfinite raw observed training power")
                hours = pd.to_datetime(times, utc=True).tz_convert("Australia/Darwin").hour
                for index, stamp in enumerate(times):
                    pm, cm = bool(observed[index]), bool(np.isfinite(clear[index]))
                    value = (pm, cm, float(power[index]) if pm else None,
                             float(clear[index]) if cm else None, int(hours[index]))
                    key = site, int(stamp)
                    if key in records and records[key] != value:
                        raise ValueError("conflicting raw observations at one physical site/time")
                    records[key] = value
        return self._finalize(records, sites_seen)

    def _finalize(self, records, sites_seen):
        groups = {}
        for (site, _), (pm, cm, power, clear, hour) in records.items():
            if pm and cm and clear > .001:
                index = float(np.clip(power / clear, 0., 2.))
                groups.setdefault((site, hour), []).append(index)
                groups.setdefault((site,), []).append(index)
        if not sites_seen or any((site,) not in groups for site in sites_seen):
            raise ValueError("B0 requires raw train daylight pair support for every site")
        self.groups = {key: float(np.median(values)) for key, values in groups.items()}
        self.support_counts = {":".join(map(str, key)): len(values) for key, values in groups.items()}
        return self

    def predict(self, batch):
        if not self.groups:
            raise ValueError("B0 train reference statistics are not fitted")
        h, mask, sites, _, _ = self._pairs(batch)
        clear = _numpy(batch["future_known"])[..., self.f]
        issues = _numpy(batch["issue_ns"])
        issue_hours = np.asarray(pd.to_datetime(issues, utc=True).tz_convert("Australia/Darwin").hour)
        indices = np.empty(len(h), dtype=np.float64)
        for row, site in enumerate(sites):
            valid = mask[row, -96:, self.p] & mask[row, -96:, self.c] & (h[row, -96:, self.c] > .001)
            if valid.any():
                last = int(np.flatnonzero(valid)[-1]) + 96
                indices[row] = np.clip(h[row, last, self.p] / h[row, last, self.c], 0., 2.)
                self.fallback_counts["history"] += 1
            else:
                key = (int(site), int(issue_hours[row]))
                if key in self.groups:
                    indices[row] = self.groups[key]
                    self.fallback_counts["site_hour"] += 1
                elif (int(site),) in self.groups:
                    indices[row] = self.groups[(int(site),)]
                    self.fallback_counts["site_global"] += 1
                else:
                    raise ValueError("B0 requested site has no train support")
        day = _numpy(batch["daylight_mask"])
        return np.where(day, np.clip(clear * indices[:, None], 0., 1.05), 0.)

    def fitted_identity(self):
        return dict(version="protocol_B0_train_raw_fallback_v1", support_counts=self.support_counts,
            reference_medians={":".join(map(str, key)): value for key, value in self.groups.items()},
            fallback_counts=self.fallback_counts, unique_physical_pairs=True, fit_split="train")
