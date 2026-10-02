"""Bounded feature generation, one-lead mmap fitting, durable subestimators."""

import gc
import os
import time
import uuid
from contextlib import contextmanager

import numpy as np
from torch.utils.data import DataLoader

from racepv.baselines.research_registry import QUANTILES
from racepv.training.exploratory_contract import equal
from racepv.training.recoverable_run import _canonical_json, _write_exclusive
from racepv.training.research_baseline_bridge import _inputs, _numpy, _target_mask
from racepv.training.research_batch_storage import (
    commit_unit,
    require_space,
    stage_directory,
    write_array,
)

FEATURE_DIMENSION = 5391


def normalized_batch(batch):
    result = dict(batch)
    for key in ("history_mask", "target_mask", "daylight_mask"):
        mask = result[key]
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(key + " must be binary before bool conversion")
        result[key] = mask.bool()
    return result


def batches(dataset, size):
    for batch in DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0, drop_last=False):
        yield normalized_batch(batch)


def lead_features(batch, lead):
    """Exactly the legacy feature order without constructing N x 24 x 5391."""
    if type(lead) is not int or not 0 <= lead < 24:
        raise ValueError("lead must be in [0,24)")
    x = _inputs(normalized_batch(batch))
    h, mask, future, site = (_numpy(x[key]).astype(np.float32) for key in x)
    result = np.concatenate((h.reshape(len(h), -1), mask.reshape(len(h), -1), site, future[:, lead]), axis=1)
    if result.shape != (len(h), FEATURE_DIMENSION) or not np.isfinite(result).all():
        raise ValueError("invalid full-information lead features")
    return result


class OwnedLeadCache:
    """One pair of exclusively created files, reused only through their owned handles.

    Feature generation is batch bounded. sklearn/LightGBM can copy their fitting
    matrix and allocate large trees; mmap does NOT promise constant-memory fit.
    """

    def __init__(self, binding, train_count, validation_count):
        self.binding = binding
        self.reserve = binding.raw["shared"]["resources"]["scratch_reserve_bytes"]
        self.maximum_bytes = (train_count + validation_count) * FEATURE_DIMENSION * 4
        self.capacity = require_space(binding.run_root, self.maximum_bytes, self.reserve)
        self.root = binding.run_root / "scratch" / ("owned-" + uuid.uuid4().hex)
        self.root.mkdir(mode=0o700)
        _write_exclusive(self.root / "owner.json", _canonical_json(dict(binding_sha256=binding.sha256,
            maximum_feature_bytes=self.maximum_bytes, feature_dimension=FEATURE_DIMENSION,
            warning="fit libraries may copy the matrix; host RAM requirement is additional")))
        self.handles = {}
        try:
            for split in ("train", "validation"):
                self.handles[split] = (self.root / (split + ".f32")).open("x+b")
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles.values():
            if not handle.closed:
                # Only the exclusive session-owned scratch inode is released.
                handle.truncate(0)
                handle.close()

    @contextmanager
    def matrix(self, dataset, split, lead, batch_size):
        count = sum(int(_numpy(_target_mask(batch))[:, lead].sum()) for batch in batches(dataset, batch_size))
        if count == 0:
            raise ValueError(f"lead {lead} has no legal daylight {split} support")
        handle = self.handles[split]
        requested = count * FEATURE_DIMENSION * 4
        require_space(self.root, max(0, requested - os.fstat(handle.fileno()).st_size), self.reserve)
        handle.truncate(requested)
        handle.flush()
        matrix = np.memmap(handle, mode="r+", dtype=np.float32, shape=(count, FEATURE_DIMENSION))
        targets = np.empty(count, dtype=np.float32)
        offset = 0
        try:
            for batch in batches(dataset, batch_size):
                legal = _numpy(_target_mask(batch))[:, lead]
                n = int(legal.sum())
                matrix[offset:offset + n] = lead_features(batch, lead)[legal]
                targets[offset:offset + n] = _numpy(batch["targets"])[legal, lead]
                offset += n
            if offset != count:
                raise ValueError("lead population changed during cache construction")
            matrix.flush()
            yield matrix, targets
        finally:
            matrix.flush()
            matrix._mmap.close()
            handle.truncate(0)
            handle.flush()


def model_capacity(model_id, config, samples):
    """Conservative serialized-size planning before expensive fitting."""
    if model_id == "random_forest_mondrian":
        from sklearn.tree import _tree
        nodes = max(1, 2 * (samples // config["min_samples_leaf"]) - 1)
        # node struct + scalar value, plus per-tree pickle bookkeeping reserve.
        return config["n_estimators"] * (nodes * (_tree.NODE_DTYPE.itemsize + 8) + 65536) + 1024**2
    return config["n_estimators"] * config["num_leaves"] * 512 + 1024**2


def _estimator(model_id, config, quantile):
    if model_id == "quantile_lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**config, objective="quantile", alpha=quantile)
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**config)


def run_trees(binding, datasets, existing, *, on_unit_committed=None):
    """Already committed estimators are never deserialized or fitted again."""
    import joblib
    identity = binding.raw["model_identity"]
    config, name = identity["resolved_config"], identity["model_id"]
    limits = binding.raw["shared"]["resources"]
    size = limits["batch_size"]
    quantiles = QUANTILES if name == "quantile_lightgbm" else (None,)
    done = {receipt["unit"] for receipt in existing}
    previous = existing[-1]["commit_sha256"] if existing else None
    expected_count = 24 * len(quantiles)
    if len(done) == expected_count:
        return existing
    cache = OwnedLeadCache(binding, len(datasets["train"]), len(datasets["validation"]))
    try:
        for lead in range(24):
            names = [f"l{lead:02d}-q{q:02d}" for q in range(len(quantiles))]
            if all(unit in done for unit in names):
                continue
            with cache.matrix(datasets["train"], "train", lead, size) as (x, y):
                if name == "quantile_lightgbm":
                    validation_context = cache.matrix(datasets["validation"], "validation", lead, size)
                else:
                    from contextlib import nullcontext
                    validation_context = nullcontext((None, None))
                with validation_context as (vx, vy):
                    for q, unit in enumerate(names):
                        if unit in done:
                            continue
                        started = time.perf_counter()
                        predicted_size = len(datasets["validation"]) * 4 + 1024
                        planned = model_capacity(name, config, len(y))
                        if planned > limits["model_file_limit_bytes"]:
                            raise RuntimeError(f"RESOURCE_FAILED: conservative model bound {planned} exceeds model-file-limit {limits['model_file_limit_bytes']}; increase explicit bound before preparing campaign")
                        space = require_space(binding.run_root, planned + predicted_size, limits["scratch_reserve_bytes"])
                        estimator = _estimator(name, config, quantiles[q])
                        equal(estimator.get_params()["random_state"], binding.raw["job"]["seed"], "effective tree random seed")
                        effective_estimator_params = estimator.get_params()
                        if name == "quantile_lightgbm":
                            from lightgbm import early_stopping
                            estimator.fit(x, y, eval_set=[(vx, vy)], eval_metric="quantile",
                                callbacks=[early_stopping(50, first_metric_only=True, verbose=False)])
                            best_iteration = int(estimator.best_iteration_)
                            if best_iteration < 1:
                                raise ValueError("LightGBM did not fix a positive best iteration")
                        else:
                            estimator.fit(x, y)
                            best_iteration = None
                        prediction = np.empty(len(datasets["validation"]), dtype=np.float32)
                        offset = 0
                        for batch in batches(datasets["validation"], size):
                            features = lead_features(batch, lead)
                            values = estimator.predict(features, num_iteration=best_iteration) if best_iteration is not None else estimator.predict(features)
                            if np.shape(values) != (len(features),) or not np.isfinite(values).all():
                                raise ValueError("nonfinite or malformed subestimator predictions")
                            prediction[offset:offset + len(features)] = values
                            offset += len(features)
                        if offset != len(prediction):
                            raise ValueError("validation prediction coverage changed")
                        stage = stage_directory(binding, unit)
                        # A real file stream avoids the neural 2GiB safe_read/BytesIO path.
                        require_space(stage, planned + predicted_size, limits["scratch_reserve_bytes"])
                        with (stage / "model.joblib").open("xb") as handle:
                            joblib.dump(estimator, handle, compress=0, protocol=5)
                            handle.flush()
                            os.fsync(handle.fileno())
                        write_array(stage / "prediction.npy", prediction)
                        del estimator
                        gc.collect()
                        metadata = dict(lead=lead, quantile=quantiles[q], seed=binding.raw["job"]["seed"],
                            train_legal_count=len(y), validation_early_stopping_count=len(vy) if vy is not None else None,
                            best_iteration=best_iteration, elapsed_seconds=time.perf_counter() - started,
                            effective_estimator_params=effective_estimator_params,
                            conservative_model_bytes=planned, resource_preflight=space,
                            cache_maximum_feature_bytes=cache.maximum_bytes,
                            memory_limit="mmap feature generation only; fit library copies and model RAM are additional")
                        receipt = commit_unit(binding, stage, unit, metadata=metadata, previous=previous)
                        previous = receipt["commit_sha256"]
                        existing.append(receipt)
                        if on_unit_committed:
                            on_unit_committed(receipt)
    finally:
        cache.close()
    return existing
