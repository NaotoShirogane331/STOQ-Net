"""Portable NPZ interchange for pre-windowed datasets and predictions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import zipfile
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .audit import sha256_file
from .utils import atomic_json_dump

if TYPE_CHECKING:
    from racepv.data.dataset import WindowedPVDataset

MODEL_ARRAYS = (
    "history_values",
    "history_mask",
    "future_known",
    "site_metadata",
    "targets",
    "state_labels",
)

REGISTERED_HISTORY_FEATURES = (
    "power_norm",
    "clear_sky_power_norm",
    "clear_sky_residual",
    "power_diff_1",
    "power_diff_2",
    "rolling_volatility_1h",
    "rolling_volatility_3h",
    "clear_sky_index",
    "power_observed",
    "power_imputed",
    "ghi_z",
    "temperature_z",
    "humidity_z",
    "wind_speed_z",
)
REGISTERED_FUTURE_FEATURES = (
    "clear_sky_power_norm",
    "solar_elevation",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "lead_time_hours",
)
REGISTERED_SITE_FEATURES = (
    "latitude_scaled",
    "longitude_scaled",
    "log_capacity_kw_scaled",
    "surface_tilt_scaled",
    "surface_azimuth_scaled",
    "has_geometry",
)
REGISTERED_STATE_LABELS = {
    "ignore": -100,
    "stable": 0,
    "ramp_up": 1,
    "ramp_down": 2,
}
REGISTERED_SPATIAL_ROLES = {
    "standard",
    "source_train",
    "validation_calibration",
    "zero_parameter_test",
}
REGISTERED_DATASET_KINDS = {
    "dkasc",
    "nrel",
    "solar_energy_137",
    "synthetic",
}

WINDOW_STORE_FORMAT = "racepv-window-store-v1"
WINDOW_SEQUENCE_STORE_FORMAT = "racepv-window-sequence-store-v1"
WINDOW_SHARD_INDEX_FORMAT = "racepv-window-shard-index-v1"
REGISTERED_LOOKBACK_STEPS = 192
REGISTERED_FREQUENCY_NS = 15 * 60 * 1_000_000_000

AUDIT_ARRAYS = (
    "daylight_mask",
    "timestamps",
    "day_ids",
    "site_ids",
    "target_observed_mask",
    "issue_ns",
)


def _window_indices(indices, length):
    """Validate before narrowing or normalizing Python-style window indices."""
    if isinstance(indices, slice):
        for value in (indices.start, indices.stop, indices.step):
            if value is not None and (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            ):
                raise TypeError("slice bounds must be integers, not bool")
        indices = range(*indices.indices(length))
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if indices.dtype.kind != "i":
            raise TypeError("indices must have signed integer dtype")
    elif not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise TypeError("indices must be a sequence of signed integers or a slice")
    values = []
    for value in indices:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or (isinstance(value, np.integer) and value.dtype.kind != "i")
        ):
            raise TypeError("indices must contain only signed integers")
        value = int(value)
        if value < -length or value >= length:
            raise IndexError("window index is out of bounds")
        values.append(value + length if value < 0 else value)
    return np.asarray(values, dtype=np.int64)


def _window_names(names):
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError("names must be a sequence of array names")
    names = tuple(names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("names must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError("names must not contain duplicates")
    if any(name not in (*MODEL_ARRAYS, *AUDIT_ARRAYS) for name in names):
        raise KeyError("names must be registered model or audit arrays")
    return names


def _read_array_batch(values, positions):
    if len(positions) == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    if isinstance(values, (_SequenceWindowArray, _ShardedArray)):
        return values.read_batch(positions)
    if type(values) is _contract().ResolverModelArray:
        return values.read_batch(positions)
    return np.array(values[positions], copy=True)


def _array_selection(values, index):
    """Apply the sample-axis selection before any trailing-axis selection."""
    selectors = list(index) if isinstance(index, tuple) else [index]
    ellipses = [i for i, item in enumerate(selectors) if item is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("an index can only have a single ellipsis")
    consumed = sum(item is not None and item is not Ellipsis for item in selectors)
    if consumed > values.ndim:
        raise IndexError("too many indices for array")
    if ellipses:
        position = ellipses[0]
        selectors[position : position + 1] = [slice(None)] * (values.ndim - consumed)
    else:
        selectors.extend([slice(None)] * (values.ndim - consumed))
    sample_axis = next(i for i, item in enumerate(selectors) if item is not None)
    first = selectors[sample_axis]
    scalar = isinstance(first, (int, np.integer, bool, np.bool_))
    positions = _window_indices([first] if scalar else first, len(values))
    batch = values.read_batch(positions)
    selectors[sample_axis] = (
        0 if scalar else slice(None) if isinstance(first, slice) else np.arange(len(positions))
    )
    return batch[tuple(selectors)]


def _materialize_batches(values):
    """Explicit full conversion only; intermediate reads stay bounded."""
    result = np.empty(values.shape, dtype=values.dtype)
    for start in range(0, len(values), 64):
        stop = min(start + 64, len(values))
        result[start:stop] = values.read_batch(range(start, stop))
    return result


def read_model_window_batch(dataset, indices, names=MODEL_ARRAYS):
    """Read independent window batches; audit fields require explicit names."""
    from racepv.data.dataset import WindowedPVDataset

    positions = _window_indices(indices, len(dataset))
    requested = _window_names(names)
    if isinstance(dataset, NpzWindowDataset):
        return dataset.read_model_window_batch(positions, requested)
    if not isinstance(dataset, WindowedPVDataset):
        raise TypeError("unsupported window dataset")
    model_names = tuple(name for name in requested if name in MODEL_ARRAYS)
    audit_names = tuple(name for name in requested if name in AUDIT_ARRAYS)
    audits = {}
    if audit_names:
        audits = _contract().read_qualified_memory_audits(dataset, positions, audit_names)
    result = dataset.read_model_window_batch(positions, model_names)
    result.update(audits)
    return {name: result[name] for name in requested}


def _contract():
    from racepv.data import prepared_contract

    return prepared_contract


def _v2_claim(schema):
    return isinstance(schema, Mapping) and (
        schema.get("schema_version") not in (None, 1)
        or type(schema.get("schema_version", 1)) is not int
        or "schema_kind" in schema
        or "causality_profile" in schema
        or str(schema.get("format", "")).endswith("-v2")
        or "artifact_kind" in schema
    )


def _admit_v2(path, *, expected_version, expected_profile, formats):
    contract = _contract()
    if expected_version is None or expected_profile is None:
        raise ValueError("v2 admission requires explicit expected_version and expected_profile")
    contract._exact(expected_version, 2, "explicit expected_version")
    contract._exact(expected_profile, contract.PROFILE, "explicit expected_profile")
    metadata = contract.validate_v2_metadata(path)
    if metadata["descriptor"]["format"] not in formats:
        raise ValueError("descriptor format does not match this constructor")
    return metadata


def _legacy_expectation(expected_version, expected_profile):
    if expected_version is not None and (
        type(expected_version) is not int or expected_version != 1
    ):
        raise ValueError("legacy data cannot satisfy the requested protocol version")
    if expected_profile is not None:
        raise ValueError("legacy data cannot satisfy an origin-safe profile")


def _load_v2_leaf(metadata, cache):
    contract = _contract()
    root = Path(metadata["artifact_root"])
    physical = {}
    for name, ref in metadata["descriptor"]["arrays"].items():
        if "resolver" in ref:
            continue
        full = contract.contained_artifact_file(root, ref["path"])
        key = (str(full), ref["member"])
        if key not in cache:
            if ref["member"] is not None:
                with np.load(full, allow_pickle=False) as archive:
                    if set(archive.files) != set(contract.ARRAY_ROLES):
                        raise ValueError("NPZ members must exactly match registered logical roles")
                    for member in archive.files:
                        cache[(str(full), member)] = np.array(archive[member], copy=True)
            else:
                cache[key] = np.load(full, mmap_mode="r", allow_pickle=False)
        physical[name] = cache[key]
    if metadata["descriptor"]["format"] == contract.V2_WINDOW_SEQUENCE_STORE_FORMAT:
        physical = contract.virtual_model_arrays(metadata["schema"], physical)
    contract.verify_v2_semantics(metadata["schema"], physical)
    return physical


def _install_v2(dataset, metadata, *, split=None):
    dataset._v2_metadata = metadata
    dataset._v2_descriptor_path = metadata["descriptor_path"]
    views = metadata.get("views", [metadata])
    cache = {}
    # The constructor has already validated every node, path and hash before
    # this first array open, even when only one split will be selected below.
    dataset._v2_views = [(view, _load_v2_leaf(view, cache)) for view in views]
    selected = [
        (view, arrays)
        for view, arrays in dataset._v2_views
        if split is None or view["schema"]["split"] == split
    ]
    if not selected:
        raise ValueError("no matching split in origin-safe store")
    if split is not None and len(selected) != 1:
        raise ValueError("sequence split must select exactly one view")
    dataset._v2_selected = selected
    dataset.schema = copy.deepcopy(selected[0][0]["schema"])
    dataset.content_identity = metadata["descriptor"]["aggregate_content_sha256"]
    dataset.forecast_population_identity = metadata["descriptor"]["forecast_population_sha256"]
    dataset.causal_arrays = (
        selected[0][1]
        if len(selected) == 1
        else {f"{v['schema']['site']['id']}/{v['schema']['split']}": a for v, a in selected}
    )
    dataset.arrays = {
        name: selected[0][1][name]
        if len(selected) == 1
        else _ShardedArray([arrays[name] for _, arrays in selected])
        for name in (*MODEL_ARRAYS, *AUDIT_ARRAYS)
    }
    dataset.aux_arrays = {
        f"{v['schema']['site']['id']}/{v['schema']['split']}": {
            name: array for name, array in arrays.items() if name not in MODEL_ARRAYS
        }
        for v, arrays in selected
    }
    dataset._archive_names = frozenset(dataset.arrays)
    dataset._offsets = np.cumsum(
        [0, *[a["window_starts"].shape[0] for _, a in selected]], dtype=np.int64
    )
    if split is not None:
        dataset.split = split
        dataset.window_starts = selected[0][1]["window_starts"]
        dataset.virtual_shapes = {name: values.shape for name, values in dataset.arrays.items()}
        dataset.virtual_dtypes = {name: values.dtype for name, values in dataset.arrays.items()}
        dataset.arrays = {name: _SequenceWindowArray(dataset, name) for name in dataset.arrays}
    dataset.collection_split = metadata["descriptor"].get("selected_split", dataset.schema["split"])


def _verify_stored_v2(dataset, *, version, profile):
    contract = _contract()
    if not hasattr(dataset, "_v2_descriptor_path"):
        raise ValueError("stored dataset has no validated v2 publication")
    metadata = contract.validate_v2_metadata(dataset._v2_descriptor_path)
    contract._exact(dataset._v2_metadata, metadata, "stored published metadata")
    views = metadata.get("views", [metadata])
    if len(views) != len(dataset._v2_views):
        raise ValueError("stored views changed")
    for current, (old, arrays) in zip(views, dataset._v2_views, strict=True):
        contract._exact(old, current, "stored view identity")
        contract.verify_v2_semantics(current["schema"], arrays)
    selected = dataset._v2_selected
    expected_selected = [
        (view, arrays)
        for view, arrays in dataset._v2_views
        if not isinstance(dataset, NpySequenceWindowDataset)
        or view["schema"]["split"] == dataset.split
    ]
    if len(selected) != len(expected_selected) or any(
        view != expected_view or arrays is not expected_arrays
        for (view, arrays), (expected_view, expected_arrays) in zip(
            selected, expected_selected, strict=True
        )
    ):
        raise ValueError("stored selection changed")
    if (
        dataset.content_identity != metadata["descriptor"]["aggregate_content_sha256"]
        or dataset.forecast_population_identity
        != metadata["descriptor"]["forecast_population_sha256"]
    ):
        raise ValueError("stored public content/population identity changed")
    contract._keys(dataset.arrays, (*MODEL_ARRAYS, *AUDIT_ARRAYS), "stored public array roles")
    public_causal = (
        selected[0][1]
        if len(selected) == 1
        else {
            f"{view['schema']['site']['id']}/{view['schema']['split']}": arrays
            for view, arrays in selected
        }
    )
    contract._keys(dataset.causal_arrays, public_causal, "stored causal arrays")
    for key, value in public_causal.items():
        if dataset.causal_arrays[key] is not value:
            raise ValueError("stored causal array binding changed")
    expected_aux = {
        f"{view['schema']['site']['id']}/{view['schema']['split']}": {
            name: array for name, array in arrays.items() if name not in MODEL_ARRAYS
        }
        for view, arrays in selected
    }
    contract._keys(dataset.aux_arrays, expected_aux, "stored auxiliary views")
    for key, expected in expected_aux.items():
        contract._keys(dataset.aux_arrays[key], expected, "stored auxiliary roles")
        if any(dataset.aux_arrays[key][name] is not array for name, array in expected.items()):
            raise ValueError("stored auxiliary array binding changed")
    if dataset.schema != selected[0][0]["schema"]:
        raise ValueError("stored schema changed")
    if len(dataset) != sum(len(a["window_starts"]) for _, a in selected):
        raise ValueError("stored sample count changed")
    offset = 0
    for _view, arrays in selected:
        for start in range(0, len(arrays["window_starts"]), 64):
            indices = np.arange(start, min(start + 64, len(arrays["window_starts"])))
            actual = dataset.read_model_window_batch(indices + offset)
            for name in MODEL_ARRAYS:
                array = arrays[name]
                expected = (
                    array.read_batch(indices)
                    if type(array) is contract.ResolverModelArray
                    else array[indices]
                )
                contract._same(actual[name], expected, "stored exposed " + name)
            for name in AUDIT_ARRAYS:
                exposed = np.stack([dataset.arrays[name][int(i + offset)] for i in indices])
                contract._same(exposed, arrays[name][indices], "stored audit " + name)
        offset += len(arrays["window_starts"])
    identity = contract.require_origin_safe_history_contract(
        dataset._v2_descriptor_path, version=version, profile=profile
    )
    identity["array_semantics_verified"] = True
    if metadata["descriptor"]["format"] == contract.V2_WINDOW_SEQUENCE_STORE_FORMAT:
        identity["split"] = dataset.split
        identity["selected_forecast_population_sha256"] = dataset.schema[
            "forecast_population_sha256"
        ]
    return identity


def _v2_write_capability():
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
    ):
        raise NotImplementedError("v2 publication requires directory-relative no-follow writes")


def _v2_absolute(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("v2 output path must not contain parent traversal")
    return path.absolute()


@contextmanager
def _v2_directory(path, *, create=False):
    """Walk from / with held directory descriptors, never traversing links.

    A replacement symlink cannot redirect any subsequent mkdir/open. An actor
    able to rename an already open directory elsewhere may relocate that inode;
    this is not a namespace lock against arbitrary privileged directory moves.
    """
    _v2_write_capability()
    path = _v2_absolute(path)
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass  # Concurrent creation is checked by the no-follow open.
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _v2_check_destination(path):
    path = _v2_absolute(path)
    # Preflight is read-only and catches dangling links as well as live links.
    for component in reversed((path, *path.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("v2 output must not contain symlink components")
        if component != path and not stat.S_ISDIR(mode):
            raise ValueError("v2 output ancestor must be a directory")


@contextmanager
def _v2_open_new(path):
    path = _v2_absolute(path)
    with _v2_directory(path.parent, create=True) as parent:
        descriptor = os.open(
            path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            yield handle


def _v2_write_bytes(path, data):
    with _v2_open_new(path) as handle:
        handle.write(data)


def _v2_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _v2_file(path, root):
    digest = hashlib.sha256()
    with _v2_directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            size = os.fstat(handle.fileno()).st_size
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


def _v2_publish(path, descriptor):
    contract = _contract()
    encoded = _v2_json_bytes(descriptor)
    receipt = {
        "artifact_kind": "racepv.prepared-receipt",
        "schema_version": 2,
        "descriptor": {
            "path": path.name,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": len(encoded),
        },
    }
    _v2_write_bytes(path.with_name(contract._receipt_name(path)), _v2_json_bytes(receipt))
    _v2_write_bytes(path, encoded)  # Completion descriptor is the last exclusive write.


def _v2_attempt(path, *, directory, references=()):
    _v2_write_capability()  # Unsupported platforms fail before even an attempt.
    _v2_check_destination(path)
    if not directory:
        _v2_check_destination(path.with_name(_contract()._receipt_name(path)))
    for name in references:
        _contract()._relative_name(name)
        _v2_check_destination(path.parent / name)
    attempt = path if directory else path.with_name(path.name + ".attempt")
    _v2_check_destination(attempt)
    with _v2_directory(attempt.parent, create=True) as parent:
        os.mkdir(attempt.name, dir_fd=parent)
    if not directory and path.exists():
        raise FileExistsError(path)
    return attempt


def _v2_failure(attempt, error):
    try:
        _v2_write_bytes(
            attempt / "failure.json",
            _v2_json_bytes({"exception": type(error).__name__, "message": str(error)}),
        )
    except (OSError, ValueError) as failure:
        error.add_note(f"failure evidence destination became unavailable: {failure}")


def _v2_prepared(dataset, schema, site_index):
    contract = _contract()
    contract.require_origin_safe_history_contract(dataset)
    prepared = dataset.prepared_contract
    contract._exact(dict(schema), prepared["schema"], "writer schema")
    contract._exact(site_index, prepared["schema"]["site"]["id"], "writer site_index")
    return prepared


def _v2_save_array(handle, array):
    if type(array) is not _contract().ResolverModelArray:
        np.save(handle, array, allow_pickle=False)
        return
    np.lib.format.write_array_header_2_0(
        handle,
        {
            "descr": np.lib.format.dtype_to_descr(array.dtype),
            "fortran_order": False,
            "shape": array.shape,
        },
    )
    for start in range(0, len(array), 64):
        handle.write(
            array.read_batch(np.arange(start, min(start + 64, len(array)))).tobytes(order="C")
        )


def _v2_leaf_write(path, prepared, *, fmt, shared=None):
    contract = _contract()
    root = path.parent
    schema = prepared["schema"]
    arrays = contract.virtual_model_arrays(
        schema,
        {
            name: array
            for name, array in prepared["causal_arrays"].items()
            if name not in MODEL_ARRAYS
        },
    )
    files = {}
    shared = {} if shared is None else shared

    def put(relative, writer):
        full = root / relative
        if relative not in shared:
            writer(full)
            shared[relative] = _v2_file(full, root)
        files[relative] = shared[relative]

    for name, data in prepared["evidence"].items():
        put(name, lambda full, data=data: _v2_write_bytes(full, data))
    stem = (
        path.name.removesuffix(".descriptor.json")
        if path.name.endswith(".descriptor.json")
        else path.stem
    )
    schema_name = stem + ".schema.json"
    put(schema_name, lambda full: _v2_write_bytes(full, _v2_json_bytes(schema)))
    refs = {}
    if fmt == "npz":

        def save_npz(full):
            with _v2_open_new(full) as handle:
                with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name, array in arrays.items():
                        with archive.open(name + ".npy", "w", force_zip64=True) as member:
                            _v2_save_array(member, array)

        put(stem + ".npz", save_npz)
        refs = {name: {"path": stem + ".npz", "member": name} for name in contract.ARRAY_ROLES}
    else:
        for name, array in arrays.items():
            if fmt == contract.V2_WINDOW_SEQUENCE_STORE_FORMAT and name in MODEL_ARRAYS:
                refs[name] = contract.resolver_reference(name)
                continue
            prefix = (
                "arrays"
                if fmt != contract.V2_WINDOW_SEQUENCE_STORE_FORMAT
                or name in contract.POPULATION_ARRAYS[:13]
                else stem
            )
            relative = f"{prefix}/{name}.npy"

            def save_npy(full, array=array):
                with _v2_open_new(full) as handle:
                    _v2_save_array(handle, array)

            put(relative, save_npy)
            refs[name] = {"path": relative, "member": None}
    descriptor = {
        "artifact_kind": "racepv.prepared-descriptor",
        "schema_version": 2,
        "causality_profile": contract.PROFILE,
        "format": fmt,
        "schema": schema_name,
        "files": list(files.values()),
        "arrays": refs,
        "aggregate_content_sha256": _aggregate_file_identities(list(files.values())),
        "forecast_population_sha256": schema["forecast_population_sha256"],
    }
    _v2_publish(path, descriptor)
    return contract.validate_v2_metadata(path)


def _v2_index_write(path, children, *, fmt, split):
    contract = _contract()
    root = _v2_absolute(path.parent)
    files, entries = {}, []
    representative = None
    for child, selected_split in children:
        selected = [
            v for v in child.get("views", [child]) if v["schema"]["split"] == selected_split
        ]
        if len(selected) != 1:
            raise ValueError("index must select one child view")
        view = selected[0]
        representative = (
            representative or Path(view["artifact_root"]) / view["descriptor"]["schema"]
        )
        for node in contract._graph_nodes(child):
            for item in node["descriptor"]["files"]:
                full = Path(node["artifact_root"]) / item["path"]
                relative = full.relative_to(root).as_posix()
                value = {**item, "path": relative}
                if relative in files and files[relative] != value:
                    raise ValueError("conflicting index file identity")
                files[relative] = value
            for full in (
                Path(node["descriptor_path"]),
                Path(node["descriptor_path"]).with_name(
                    contract._receipt_name(Path(node["descriptor_path"]))
                ),
            ):
                item = _v2_file(full, root)
                files[item["path"]] = item
        entries.append(
            {
                "descriptor": _v2_file(Path(child["descriptor_path"]), root),
                "site_id": view["schema"]["site"]["id"],
                "split": selected_split,
                "aggregate_content_sha256": child["descriptor"]["aggregate_content_sha256"],
                "forecast_population_sha256": view["schema"]["forecast_population_sha256"],
            }
        )
    population = {
        "causality_profile": contract.PROFILE,
        "format": fmt,
        "selected_split": split,
        "leaves": [
            {key: entry[key] for key in ("site_id", "split", "forecast_population_sha256")}
            for entry in entries
        ],
    }
    descriptor = {
        "artifact_kind": "racepv.prepared-index",
        "schema_version": 2,
        "causality_profile": contract.PROFILE,
        "format": fmt,
        "schema": representative.relative_to(root).as_posix(),
        "selected_split": split,
        "leaves": entries,
        "files": list(files.values()),
        "aggregate_content_sha256": _aggregate_file_identities(list(files.values())),
        "forecast_population_sha256": contract.canonical_json_sha256(population),
    }
    _v2_publish(path, descriptor)
    return contract.validate_v2_metadata(path)


def _aggregate_file_identities(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for state in sorted(files, key=lambda value: str(value["path"])):
        digest.update(str(state["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(state["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(state["bytes"])).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class NpzWindowDataset(Dataset[dict[str, Tensor]]):
    """Memory-resident, validated experiment interchange dataset."""

    def __init__(self, path: str | Path, *, expected_version=None, expected_profile=None) -> None:
        self.path = Path(path)
        descriptor = self.path.with_suffix(".descriptor.json")
        schema_path = self.path.with_suffix(".schema.json")
        schema_header = _contract().read_strict_json(schema_path) if schema_path.exists() else None
        if (
            descriptor.exists()
            or _v2_claim(schema_header)
            or self.path.with_name(self.path.name + ".attempt").exists()
        ):
            metadata = _admit_v2(
                descriptor,
                expected_version=expected_version,
                expected_profile=expected_profile,
                formats={"npz"},
            )
            _install_v2(self, metadata)
            return
        _legacy_expectation(expected_version, expected_profile)
        with np.load(self.path, allow_pickle=False) as archive:
            self._archive_names = frozenset(archive.files)
            missing = [name for name in MODEL_ARRAYS if name not in archive]
            if missing:
                raise KeyError(f"{self.path} is missing required arrays: {missing}")
            self.arrays = {name: np.asarray(archive[name]) for name in archive.files}
        schema_path = self.path.with_suffix(".schema.json")
        if schema_path.exists():
            with schema_path.open(encoding="utf-8") as handle:
                schema = json.load(handle)
            if not isinstance(schema, dict):
                raise TypeError(f"{schema_path} must contain a JSON object")
            self.schema: dict[str, Any] | None = schema
        else:
            self.schema = None
        self._validate()

    def _validate(self, *, verify_values: bool = True) -> None:
        sizes = {name: self.arrays[name].shape[0] for name in MODEL_ARRAYS}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"all arrays must share a sample dimension: {sizes}")
        history = self.arrays["history_values"]
        mask = self.arrays["history_mask"]
        future = self.arrays["future_known"]
        targets = self.arrays["targets"]
        labels = self.arrays["state_labels"]
        site_metadata = self.arrays["site_metadata"]
        if history.ndim != 3 or history.shape[1] != 192:
            raise ValueError("history_values must have shape [N,192,C]")
        if mask.shape != history.shape:
            raise ValueError("history_mask must match history_values")
        if future.ndim != 3 or future.shape[1] != 24:
            raise ValueError("future_known must have shape [N,24,F]")
        if site_metadata.ndim != 2 or site_metadata.shape[0] != history.shape[0]:
            raise ValueError("site_metadata must have shape [N,S]")
        if targets.shape != (history.shape[0], 24):
            raise ValueError("targets must have shape [N,24]")
        if labels.shape != targets.shape:
            raise ValueError("state_labels must match targets")
        for name in ("history_values", "future_known", "site_metadata", "targets"):
            values = self.arrays[name]
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"{name} must be numeric")
            if verify_values and not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if not np.issubdtype(mask.dtype, np.number) and mask.dtype != np.bool_:
            raise TypeError("history_mask must be numeric or boolean")
        if verify_values and (not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all()):
            raise ValueError("history_mask must be a finite binary array")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("state_labels must use an integer dtype")
        if verify_values and not np.isin(labels, (-100, 0, 1, 2)).all():
            raise ValueError(
                "state_labels must be stable=0, ramp_up=1, ramp_down=2, or -100 for ignored night"
            )
        if "daylight_mask" in self.arrays:
            raw_daylight = np.asarray(self.arrays["daylight_mask"])
            if raw_daylight.shape != targets.shape:
                raise ValueError("daylight_mask must match targets")
            if (
                not np.issubdtype(raw_daylight.dtype, np.number)
            ) and raw_daylight.dtype != np.bool_:
                raise TypeError("daylight_mask must be numeric or boolean")
            if verify_values and (
                not np.isfinite(raw_daylight).all() or not np.isin(raw_daylight, (0, 1)).all()
            ):
                raise ValueError("daylight_mask must be a finite binary array")
            daylight = raw_daylight.astype(bool)
        else:
            if future.shape[-1] < 1:
                raise ValueError("cannot derive daylight_mask without future clear-sky power")
            daylight = future[..., 0] >= 0.01
            self.arrays["daylight_mask"] = daylight.astype(np.float32)
        if verify_values and np.any(labels[~daylight] != -100):
            raise ValueError("night state_labels must be -100 so night is excluded from ramp loss")
        if verify_values and np.any(np.abs(targets[~daylight]) > 1e-7):
            raise ValueError("night targets must be exactly zero under the registered physics rule")

    def require_registered_schema(self) -> None:
        """Reject archives whose feature indices cannot be audited."""

        if hasattr(self, "_v2_metadata"):
            raise ValueError("v2 requires require_origin_safe_history_contract")
        audit_arrays = ("daylight_mask", "timestamps", "day_ids", "site_ids")
        missing_audit = [name for name in audit_arrays if name not in self._archive_names]
        if missing_audit:
            raise ValueError(f"registered archive is missing audit arrays: {missing_audit}")
        target_shape = self.arrays["targets"].shape
        for name in ("daylight_mask", "timestamps", "day_ids"):
            if self.arrays[name].shape != target_shape:
                raise ValueError(f"{name} must match the target shape")
        if self.arrays["site_ids"].shape not in {
            (target_shape[0],),
            (target_shape[0], 1),
        }:
            raise ValueError("site_ids must contain one value per forecast origin")
        for name in ("timestamps", "day_ids", "site_ids"):
            values = self.arrays[name]
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must use an integer dtype")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if self.schema is None:
            raise ValueError(
                f"{self.path} has no .schema.json sidecar; regenerate it with racepv-prepare"
            )
        history = tuple(self.schema.get("history_columns", ()))
        future = tuple(self.schema.get("future_known_columns", ()))
        site = tuple(self.schema.get("site_metadata_names", ()))
        if self.schema.get("schema_version") != 1:
            raise ValueError("unsupported or missing prepared schema_version")
        if history != REGISTERED_HISTORY_FEATURES:
            raise ValueError("history schema does not match the registered feature order")
        if future != REGISTERED_FUTURE_FEATURES:
            raise ValueError(
                "future-known schema does not match the audited deterministic feature contract"
            )
        if site != REGISTERED_SITE_FEATURES:
            raise ValueError("site metadata schema does not match the registered feature order")
        if self.schema.get("target") != "power_norm":
            raise ValueError("registered target must be power_norm")
        if self.schema.get("state_labels") != REGISTERED_STATE_LABELS:
            raise ValueError("state-label mapping does not match the registered contract")
        if len(history) != self.arrays["history_values"].shape[-1]:
            raise ValueError("history schema width does not match history_values")
        if len(future) != self.arrays["future_known"].shape[-1]:
            raise ValueError("future schema width does not match future_known")
        if len(site) != self.arrays["site_metadata"].shape[-1]:
            raise ValueError("site schema width does not match site_metadata")
        site_id_map = self.schema.get("site_id_map")
        if not isinstance(site_id_map, dict) or not site_id_map:
            raise ValueError("schema must contain a non-empty site_id_map")
        canonical_map: dict[str, str] = {}
        for raw_key, raw_value in site_id_map.items():
            try:
                key = str(int(raw_key))
            except (TypeError, ValueError) as error:
                raise ValueError("schema.site_id_map keys must be integer strings") from error
            if str(raw_key) != key:
                raise ValueError("schema.site_id_map keys must use canonical integer strings")
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError("schema.site_id_map values must be non-empty strings")
            canonical_map[key] = raw_value
        if len(set(canonical_map.values())) != len(canonical_map):
            raise ValueError("schema.site_id_map values must be unique")
        observed_ids = {str(int(value)) for value in np.unique(self.arrays["site_ids"])}
        if observed_ids != set(canonical_map):
            raise ValueError("schema.site_id_map keys must exactly match observed site_ids")
        spatial_roles = self.schema.get("spatial_roles")
        if not isinstance(spatial_roles, dict) or set(spatial_roles) != observed_ids:
            raise ValueError("schema.spatial_roles must describe every observed site")
        invalid_roles = {
            str(role) for role in spatial_roles.values() if role not in REGISTERED_SPATIAL_ROLES
        }
        if invalid_roles:
            raise ValueError(f"schema contains invalid spatial roles: {sorted(invalid_roles)}")
        ramp_thresholds = self.schema.get("ramp_thresholds")
        if not isinstance(ramp_thresholds, dict) or set(ramp_thresholds) != observed_ids:
            raise ValueError(
                "schema.ramp_thresholds must describe every observed site using train-only values"
            )
        for site_id in observed_ids:
            state = ramp_thresholds[site_id]
            if not isinstance(state, dict) or "threshold" not in state:
                raise ValueError(f"ramp threshold metadata is incomplete for site {site_id}")
            threshold = float(state["threshold"])
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"ramp threshold must be finite and positive for site {site_id}")

    def registered_schema_signature(self) -> dict[str, Any]:
        """Feature contract excluding dataset-specific site identities."""

        self.require_registered_schema()
        if self.schema is None:  # pragma: no cover - narrowed by the call above
            raise RuntimeError("schema unexpectedly missing")
        keys = (
            "schema_version",
            "dataset_kind",
            "simulated",
            "ramp_method",
            "history_columns",
            "future_known_columns",
            "site_metadata_names",
            "target",
            "state_labels",
        )
        # Legacy/development containers may omit the evidence fields and remain
        # loadable for diagnostics. Registered entry points separately call
        # require_registered_evidence(), which rejects those omissions.
        return {key: self.schema.get(key) for key in keys}

    def require_registered_evidence(self, *, allow_ineligible: bool = False) -> None:
        """Fail closed unless preparation provenance supports registered evidence.

        ``allow_ineligible`` exists only for explicitly labelled development and
        synthetic smoke runs. Missing provenance is never accepted, even there.
        """

        self.require_registered_schema()
        if self.schema is None:  # pragma: no cover - narrowed above
            raise RuntimeError("schema unexpectedly missing")
        dataset_kind = self.schema.get("dataset_kind")
        if dataset_kind not in REGISTERED_DATASET_KINDS:
            raise ValueError("schema has no supported dataset_kind provenance")
        simulated = self.schema.get("simulated")
        if not isinstance(simulated, bool):
            raise ValueError("schema.simulated must be an explicit boolean")
        expected_simulated = dataset_kind in {"nrel", "synthetic"}
        if simulated != expected_simulated:
            raise ValueError("schema.simulated disagrees with dataset_kind")
        eligible = self.schema.get("registered_evidence_eligible")
        if not isinstance(eligible, bool):
            raise ValueError("schema.registered_evidence_eligible must be an explicit boolean")
        provenance = self.schema.get("preparation_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("schema is missing preparation_provenance")
        manifest_hash = provenance.get("source_manifest_sha256")
        if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
            raise ValueError("preparation provenance has no valid manifest SHA-256")
        try:
            int(manifest_hash, 16)
        except ValueError as error:
            raise ValueError("manifest SHA-256 must be hexadecimal") from error
        raw_input_hash = provenance.get("raw_input_sha256")
        if not isinstance(raw_input_hash, str) or len(raw_input_hash) != 64:
            raise ValueError("preparation provenance has no valid raw-input SHA-256")
        try:
            int(raw_input_hash, 16)
        except ValueError as error:
            raise ValueError("raw-input SHA-256 must be hexadecimal") from error
        raw_file_count = provenance.get("raw_file_count")
        if not isinstance(raw_file_count, int) or isinstance(raw_file_count, bool):
            raise ValueError("preparation provenance raw_file_count must be an integer")
        if raw_file_count <= 0:
            raise ValueError("preparation provenance raw_file_count must be positive")
        raw_identities = self.schema.get("raw_source_identities")
        if not isinstance(raw_identities, list) or len(raw_identities) != raw_file_count:
            raise ValueError("schema.raw_source_identities must contain one record per raw file")
        normalized_raw_identities: list[dict[str, Any]] = []
        seen_raw_ordinals: set[tuple[str, str, int]] = set()
        for record in raw_identities:
            if not isinstance(record, dict) or set(record) != {
                "dataset",
                "site_id",
                "file_ordinal",
                "size_bytes",
                "sha256",
            }:
                raise ValueError("raw source identity record has an invalid schema")
            dataset = record["dataset"]
            site_id = record["site_id"]
            ordinal = record["file_ordinal"]
            size_bytes = record["size_bytes"]
            content_hash = record["sha256"]
            if dataset != dataset_kind or not isinstance(site_id, str) or not site_id:
                raise ValueError("raw source identity has an invalid dataset/site")
            if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
                raise ValueError("raw source file_ordinal must be a non-negative integer")
            if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
                raise ValueError("raw source size_bytes must be a non-negative integer")
            if not isinstance(content_hash, str) or len(content_hash) != 64:
                raise ValueError("raw source identity has no valid content SHA-256")
            try:
                int(content_hash, 16)
            except ValueError as error:
                raise ValueError("raw source content SHA-256 must be hexadecimal") from error
            ordinal_key = (dataset, site_id, ordinal)
            if ordinal_key in seen_raw_ordinals:
                raise ValueError("raw source identity repeats a site/file ordinal")
            seen_raw_ordinals.add(ordinal_key)
            normalized_raw_identities.append(
                {
                    "dataset": dataset,
                    "site_id": site_id,
                    "file_ordinal": ordinal,
                    "size_bytes": size_bytes,
                    "sha256": content_hash,
                }
            )
        canonical_raw_payload = json.dumps(
            normalized_raw_identities,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(canonical_raw_payload).hexdigest() != raw_input_hash:
            raise ValueError("raw source identity records disagree with raw-input SHA-256")
        if provenance.get("selected_dataset") != dataset_kind:
            raise ValueError("preparation provenance dataset disagrees with schema")
        ramp_method = self.schema.get("ramp_method")
        if ramp_method not in {"q85", "q90", "q95", "fixed0.1"}:
            raise ValueError("schema has no registered ramp_method")
        if provenance.get("ramp_method") != ramp_method:
            raise ValueError("preparation provenance ramp method disagrees with schema")
        for name in (
            "require_observed_targets",
            "allow_approximate_solar",
            "dkasc_quality_enforced",
            "nrel_selection_enforced",
        ):
            if not isinstance(provenance.get(name), bool):
                raise ValueError(f"preparation provenance {name} must be boolean")
        bypasses = provenance.get("development_bypasses")
        if not isinstance(bypasses, list) or any(
            not isinstance(value, str) or not value for value in bypasses
        ):
            raise ValueError("preparation provenance development_bypasses must be a string list")
        if eligible:
            if not provenance["require_observed_targets"]:
                raise ValueError("eligible evidence cannot use imputed daylight targets")
            if provenance["allow_approximate_solar"]:
                raise ValueError("eligible evidence cannot use approximate solar features")
            if bypasses:
                raise ValueError("eligible evidence cannot contain development bypasses")
            if dataset_kind in {"synthetic", "solar_energy_137"}:
                raise ValueError(
                    "synthetic and appendix-only benchmark data cannot be registered main evidence"
                )
            if dataset_kind == "dkasc" and not provenance["dkasc_quality_enforced"]:
                raise ValueError("eligible DKASC evidence must enforce quality selection")
            if dataset_kind == "nrel" and not provenance["nrel_selection_enforced"]:
                raise ValueError("eligible NREL evidence must enforce cohort selection")
            if dataset_kind == "nrel":
                cohort = self.schema.get("registered_nrel_cohort")
                if not isinstance(cohort, dict):
                    raise ValueError("eligible NREL schema is missing its frozen cohort")
                full_by_state = cohort.get("full_site_ids_by_state")
                if not isinstance(full_by_state, dict) or set(full_by_state) != {
                    "AL",
                    "AZ",
                    "CA",
                    "NY",
                    "OK",
                }:
                    raise ValueError("NREL cohort must describe all five registered states")
                if any(
                    not isinstance(values, list)
                    or len(values) != 20
                    or len(set(values)) != 20
                    or any(not isinstance(value, str) or not value for value in values)
                    for values in full_by_state.values()
                ):
                    raise ValueError("NREL cohort must contain 20 unique site IDs per state")
                if len({value for values in full_by_state.values() for value in values}) != 100:
                    raise ValueError("NREL full cohort site IDs must be globally unique")
                canonical = json.dumps(full_by_state, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                if hashlib.sha256(canonical).hexdigest() != cohort.get("full_cohort_sha256"):
                    raise ValueError("NREL full cohort hash does not match its site map")
                site_states = self.schema.get("site_states")
                site_id_map = self.schema.get("site_id_map")
                if (
                    not isinstance(site_states, dict)
                    or not isinstance(site_id_map, dict)
                    or set(site_states) != set(site_id_map)
                ):
                    raise ValueError("NREL schema must map every site index to a state")
                for key, site_name in site_id_map.items():
                    state = site_states[key]
                    if state not in full_by_state or site_name not in full_by_state[state]:
                        raise ValueError("NREL site/state is absent from the frozen cohort")
        elif not allow_ineligible:
            raise ValueError(
                "prepared data are ineligible for registered evidence; use an explicit "
                "development-data option only for smoke/debug runs"
            )

    def __len__(self) -> int:
        return int(self.arrays["targets"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample = {}
        for name, values in self.arrays.items():
            value = values[index]
            if hasattr(self, "_v2_metadata"):
                value = np.array(value, copy=True)
            if name == "state_labels":
                sample[name] = torch.as_tensor(value, dtype=torch.long)
            elif name in {"timestamps", "day_ids", "site_ids"} or (
                hasattr(self, "_v2_metadata") and name in AUDIT_ARRAYS
            ):
                sample[name] = torch.as_tensor(value)
            else:
                sample[name] = torch.as_tensor(value, dtype=torch.float32)
        return sample

    def verify_content(self) -> str:
        if hasattr(self, "_v2_metadata"):
            return _verify_stored_v2(self, version=2, profile=_contract().PROFILE)[
                "aggregate_content_sha256"
            ]
        return sha256_file(self.path)

    def read_model_window_batch(self, indices, names=MODEL_ARRAYS):
        positions = _window_indices(indices, len(self))
        requested = _window_names(names)
        if isinstance(self, NpySequenceWindowDataset):
            if hasattr(self, "_v2_metadata"):
                arrays = self._v2_selected[0][1]
                result = {}
                for name in requested:
                    exposed = self.arrays[name]
                    # Resolve our wrapper without recursion; replacements must remain
                    # observable to the stored object's semantic mutation guard.
                    values = (
                        arrays[name]
                        if (
                            isinstance(exposed, _SequenceWindowArray)
                            and exposed.dataset is self
                            and exposed.name == name
                        )
                        else exposed
                    )
                    result[name] = _read_array_batch(values, positions)
                return result
            return {
                name: np.stack([self._legacy_numpy_value(name, int(i)) for i in positions])
                if len(positions)
                else np.empty((0, *self.virtual_shapes[name][1:]), dtype=self.virtual_dtypes[name])
                for name in requested
            }
        return {name: _read_array_batch(self.arrays[name], positions) for name in requested}


class NpyWindowDataset(NpzWindowDataset):
    """Memory-mapped window store produced without materializing a giant NPZ."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_values: bool = True,
        verify_content: bool = False,
        expected_version=None,
        expected_profile=None,
    ) -> None:
        self.path = Path(path)
        manifest_path = self.path / "store.json"
        header = _contract().read_strict_json(manifest_path)
        if _v2_claim(header):
            metadata = _admit_v2(
                manifest_path,
                expected_version=expected_version,
                expected_profile=expected_profile,
                formats={_contract().V2_WINDOW_STORE_FORMAT},
            )
            _install_v2(self, metadata)
            return
        _legacy_expectation(expected_version, expected_profile)
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("format") != WINDOW_STORE_FORMAT:
            raise ValueError(f"{manifest_path} is not a registered window store")
        self.store_manifest = manifest
        self.content_identity = manifest.get("aggregate_content_sha256")
        if not isinstance(self.content_identity, str) or len(self.content_identity) != 64:
            raise ValueError(f"{manifest_path} has no valid aggregate content identity")
        raw_arrays = manifest.get("arrays")
        if not isinstance(raw_arrays, dict):
            raise ValueError(f"{manifest_path} must contain an arrays mapping")
        self._archive_names = frozenset(str(name) for name in raw_arrays)
        missing = [name for name in MODEL_ARRAYS if name not in self._archive_names]
        if missing:
            raise KeyError(f"{self.path} is missing required arrays: {missing}")
        arrays: dict[str, np.ndarray] = {}
        for name, state in raw_arrays.items():
            if not isinstance(state, dict) or not isinstance(state.get("file"), str):
                raise ValueError(f"invalid array metadata for {name!r}")
            array_path = self.path / state["file"]
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(value) for value in state.get("shape", ()))
            if expected_shape and values.shape != expected_shape:
                raise ValueError(f"{name} shape differs from store.json")
            if state.get("dtype") and str(values.dtype) != str(state["dtype"]):
                raise ValueError(f"{name} dtype differs from store.json")
            arrays[str(name)] = values
        self.arrays = arrays
        schema_path = self.path / "schema.json"
        if schema_path.exists():
            with schema_path.open(encoding="utf-8") as handle:
                schema = json.load(handle)
            if not isinstance(schema, dict):
                raise TypeError(f"{schema_path} must contain a JSON object")
            self.schema: dict[str, Any] | None = schema
        else:
            self.schema = None
        if not verify_values and manifest.get("value_validation") != "complete_on_write":
            raise ValueError(
                f"{manifest_path} cannot skip value verification without a write-time audit"
            )
        self._validate(verify_values=verify_values)
        if verify_content:
            self.verify_content()

    def verify_content(self) -> str:
        if hasattr(self, "_v2_metadata"):
            return NpzWindowDataset.verify_content(self)
        files = []
        for name, state in self.store_manifest["arrays"].items():
            path = self.path / str(state["file"])
            actual = {
                "path": str(state["file"]),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            if actual["sha256"] != state.get("sha256") or actual["bytes"] != int(
                state.get("bytes", -1)
            ):
                raise ValueError(f"content hash mismatch for window array {name!r}")
            files.append(actual)
        identity = _aggregate_file_identities(files)
        if identity != self.content_identity:
            raise ValueError("window store aggregate content identity does not match")
        return identity


class _ShardedArray:
    """Array-like concatenation that leaves every NPY shard memory mapped."""

    def __init__(self, parts: Sequence[np.ndarray]) -> None:
        if not parts:
            raise ValueError("a sharded array needs at least one part")
        tails = {tuple(part.shape[1:]) for part in parts}
        dtypes = {str(part.dtype) for part in parts}
        if len(tails) != 1 or len(dtypes) != 1:
            raise ValueError("sharded arrays must share trailing shapes and dtypes")
        self.parts = tuple(parts)
        self.offsets = np.cumsum([0, *(len(part) for part in self.parts)], dtype=np.int64)
        self.shape = (int(self.offsets[-1]), *self.parts[0].shape[1:])
        self.dtype = self.parts[0].dtype
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def _locate(self, index: int) -> tuple[int, int]:
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        part_index = bisect_right(self.offsets, normalized) - 1
        return part_index, int(normalized - self.offsets[part_index])

    def __getitem__(self, index: Any) -> np.ndarray | np.generic:
        return _array_selection(self, index)

    def read_batch(self, indices):
        positions = _window_indices(indices, len(self))
        result = np.empty((len(positions), *self.shape[1:]), dtype=self.dtype)
        part_ids = np.searchsorted(self.offsets, positions, side="right") - 1
        for part in np.unique(part_ids):
            destinations = np.flatnonzero(part_ids == part)
            local = positions[destinations] - self.offsets[part]
            result[destinations] = _read_array_batch(self.parts[part], local)
        return result

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        values = _materialize_batches(self)
        return values.astype(dtype, copy=False) if dtype is not None else values

    def astype(self, dtype: np.dtype[Any] | str, **kwargs: Any) -> np.ndarray:
        return np.asarray(self).astype(dtype, **kwargs)

    def reshape(self, *shape: int) -> np.ndarray:
        return np.asarray(self).reshape(*shape)


class _SequenceWindowArray:
    """Virtual pre-windowed array backed by one non-duplicated site sequence."""

    def __init__(self, dataset: NpySequenceWindowDataset, name: str) -> None:
        self.dataset = dataset
        self.name = name
        self.shape = dataset.virtual_shapes[name]
        self.dtype = dataset.virtual_dtypes[name]
        self.ndim = len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: Any) -> np.ndarray | np.generic:
        return _array_selection(self, index)

    def read_batch(self, indices):
        return self.dataset.read_model_window_batch(indices, [self.name])[self.name]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> np.ndarray:
        values = self.dataset._materialize_array(self.name)
        return values.astype(dtype, copy=False) if dtype is not None else values

    def astype(self, dtype: np.dtype[Any] | str, **kwargs: Any) -> np.ndarray:
        return np.asarray(self).astype(dtype, **kwargs)

    def reshape(self, *shape: int) -> np.ndarray:
        return np.asarray(self).reshape(*shape)


class NpySequenceWindowDataset(NpzWindowDataset):
    """Lazy windows over one site sequence; overlapping histories are not stored."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        verify_content: bool = False,
        expected_version=None,
        expected_profile=None,
    ) -> None:
        self.path = Path(path)
        manifest_path = self.path / "store.json"
        header = _contract().read_strict_json(manifest_path)
        if _v2_claim(header):
            metadata = _admit_v2(
                manifest_path,
                expected_version=expected_version,
                expected_profile=expected_profile,
                formats={_contract().V2_WINDOW_SEQUENCE_STORE_FORMAT},
            )
            if "views" not in metadata:
                raise ValueError("sequence constructor requires a complete four-view index")
            _install_v2(self, metadata, split=split)
            return
        _legacy_expectation(expected_version, expected_profile)
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("format") != WINDOW_SEQUENCE_STORE_FORMAT:
            raise ValueError(f"{manifest_path} is not a registered sequence store")
        self.store_manifest = manifest
        self.content_identity = manifest.get("aggregate_content_sha256")
        if not isinstance(self.content_identity, str) or len(self.content_identity) != 64:
            raise ValueError(f"{manifest_path} has no valid aggregate content identity")
        raw_splits = manifest.get("splits")
        if not isinstance(raw_splits, dict) or split not in raw_splits:
            raise ValueError(f"sequence store has no split {split!r}")
        self.split = split
        self.lookback = int(manifest.get("lookback", 192))
        self.horizon = int(manifest.get("horizon", 24))
        if self.lookback != 192 or self.horizon != 24:
            raise ValueError("registered sequence stores require lookback=192 and horizon=24")
        self.future_known_names = tuple(manifest.get("future_known_names", ()))
        self.base_future_names = tuple(manifest.get("base_future_names", ()))
        self.site_index = int(manifest["site_index"])

        raw_arrays = manifest.get("arrays")
        if not isinstance(raw_arrays, dict):
            raise ValueError("sequence store must contain an arrays mapping")
        self.sequence_arrays: dict[str, np.ndarray] = {}
        for name, state in raw_arrays.items():
            if not isinstance(state, dict) or not isinstance(state.get("file"), str):
                raise ValueError(f"invalid sequence array metadata for {name!r}")
            values = np.load(self.path / state["file"], mmap_mode="r", allow_pickle=False)
            if values.shape != tuple(int(value) for value in state["shape"]):
                raise ValueError(f"sequence array {name} shape differs from store.json")
            if str(values.dtype) != str(state["dtype"]):
                raise ValueError(f"sequence array {name} dtype differs from store.json")
            self.sequence_arrays[str(name)] = values
        required_sequence = {
            "history_values",
            "history_mask",
            "base_future",
            "site_metadata",
            "targets",
            "state_labels",
            "night",
            "timestamps",
            "local_day_ids",
            f"window_starts__{split}",
        }
        missing = sorted(required_sequence.difference(self.sequence_arrays))
        if missing:
            raise ValueError(f"sequence store is missing arrays: {missing}")
        self.window_starts = self.sequence_arrays[f"window_starts__{split}"]
        declared_count = int(raw_splits[split].get("sample_count", -1))
        if len(self.window_starts) != declared_count or declared_count <= 0:
            raise ValueError("sequence split count differs from window-start index")

        rows = len(self.sequence_arrays["targets"])
        for name in (
            "history_values",
            "history_mask",
            "base_future",
            "state_labels",
            "night",
            "timestamps",
            "local_day_ids",
        ):
            if len(self.sequence_arrays[name]) != rows:
                raise ValueError(f"sequence array {name} has a different row count")
        history_features = self.sequence_arrays["history_values"].shape[-1]
        future_features = len(self.future_known_names)
        site_features = self.sequence_arrays["site_metadata"].shape[-1]
        n = len(self.window_starts)
        self.virtual_shapes = {
            "history_values": (n, self.lookback, history_features),
            "history_mask": (n, self.lookback, history_features),
            "future_known": (n, self.horizon, future_features),
            "site_metadata": (n, site_features),
            "targets": (n, self.horizon),
            "state_labels": (n, self.horizon),
            "daylight_mask": (n, self.horizon),
            "timestamps": (n, self.horizon),
            "day_ids": (n, self.horizon),
            "site_ids": (n,),
        }
        self.virtual_dtypes = {
            "history_values": np.dtype("float32"),
            "history_mask": np.dtype("float32"),
            "future_known": np.dtype("float32"),
            "site_metadata": np.dtype("float32"),
            "targets": np.dtype("float32"),
            "state_labels": np.dtype("int64"),
            "daylight_mask": np.dtype("float32"),
            "timestamps": np.dtype("int64"),
            "day_ids": np.dtype("int32"),
            "site_ids": np.dtype("int64"),
        }
        self._archive_names = frozenset(self.virtual_shapes)
        self.arrays = {name: _SequenceWindowArray(self, name) for name in self.virtual_shapes}
        schema_path = self.path / "schema.json"
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        if not isinstance(schema, dict):
            raise TypeError(f"{schema_path} must contain a JSON object")
        self.schema: dict[str, Any] | None = schema
        if verify_content:
            self.verify_content()

    def __len__(self) -> int:
        return int(len(self.window_starts))

    def _positions(self, item: int) -> tuple[int, int, int]:
        normalized = item + len(self) if item < 0 else item
        if normalized < 0 or normalized >= len(self):
            raise IndexError(item)
        start = int(self.window_starts[normalized])
        target_start = start + self.lookback
        stop = target_start + self.horizon
        return start, target_start, stop

    def _future_matrix(self, target_start: int, stop: int) -> np.ndarray:
        base = self.sequence_arrays["base_future"][target_start:stop]
        base_lookup = {name: base[:, index] for index, name in enumerate(self.base_future_names)}
        lead_hours = np.arange(1, self.horizon + 1, dtype=np.float32) * 0.25
        columns = []
        for name in self.future_known_names:
            if name == "lead_time_hours":
                values = lead_hours
            elif name == "lead_time_sin":
                values = np.sin(2 * np.pi * lead_hours / 6.0).astype(np.float32)
            elif name == "lead_time_cos":
                values = np.cos(2 * np.pi * lead_hours / 6.0).astype(np.float32)
            else:
                values = base_lookup[name]
            columns.append(np.asarray(values, dtype=np.float32))
        return (
            np.stack(columns, axis=-1) if columns else np.empty((self.horizon, 0), dtype=np.float32)
        )

    def _numpy_value(self, name: str, item: int) -> np.ndarray | np.generic:
        return self.read_model_window_batch([item], [name])[name][0]

    def _legacy_numpy_value(self, name: str, item: int) -> np.ndarray | np.generic:
        start, target_start, stop = self._positions(item)
        if name in {"history_values", "history_mask"}:
            return np.array(self.sequence_arrays[name][start:target_start], copy=True)
        if name == "future_known":
            return self._future_matrix(target_start, stop)
        if name == "site_metadata":
            return np.array(self.sequence_arrays[name], copy=True)
        if name in {"targets", "state_labels"}:
            return np.array(self.sequence_arrays[name][target_start:stop], copy=True)
        if name == "daylight_mask":
            return (~self.sequence_arrays["night"][target_start:stop].astype(bool)).astype(
                np.float32
            )
        if name == "timestamps":
            return np.array(self.sequence_arrays["timestamps"][target_start:stop], copy=True)
        if name == "day_ids":
            return np.array(self.sequence_arrays["local_day_ids"][target_start:stop], copy=True)
        if name == "site_ids":
            return np.int64(self.site_index)
        raise KeyError(name)

    def _materialize_array(self, name: str) -> np.ndarray:
        return _materialize_batches(self.arrays[name])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if hasattr(self, "_v2_metadata"):
            return NpzWindowDataset.__getitem__(self, index)
        sample = {}
        for name in self.virtual_shapes:
            value = self._numpy_value(name, index)
            if name == "state_labels":
                sample[name] = torch.as_tensor(value, dtype=torch.long)
            elif name in {"timestamps", "day_ids", "site_ids"}:
                sample[name] = torch.as_tensor(value)
            else:
                sample[name] = torch.as_tensor(value, dtype=torch.float32)
        return sample

    def verify_content(self) -> str:
        if hasattr(self, "_v2_metadata"):
            return _verify_stored_v2(self, version=2, profile=_contract().PROFILE)[
                "aggregate_content_sha256"
            ]
        files = []
        for name, state in self.store_manifest["arrays"].items():
            path = self.path / str(state["file"])
            actual = {
                "path": str(state["file"]),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            if actual["sha256"] != state.get("sha256") or actual["bytes"] != int(
                state.get("bytes", -1)
            ):
                raise ValueError(f"content hash mismatch for sequence array {name!r}")
            files.append(actual)
        identity = _aggregate_file_identities(files)
        if identity != self.content_identity:
            raise ValueError("sequence store aggregate content identity does not match")
        return identity


class ShardedWindowDataset(NpzWindowDataset):
    """Global dataset index over per-site memory-mapped window stores."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_content: bool = False,
        expected_version=None,
        expected_profile=None,
    ) -> None:
        self.path = Path(path)
        header = _contract().read_strict_json(self.path)
        if _v2_claim(header):
            metadata = _admit_v2(
                self.path,
                expected_version=expected_version,
                expected_profile=expected_profile,
                formats={_contract().V2_WINDOW_SHARD_INDEX_FORMAT},
            )
            _install_v2(self, metadata)
            return
        _legacy_expectation(expected_version, expected_profile)
        with self.path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        if not isinstance(index, dict) or index.get("format") != WINDOW_SHARD_INDEX_FORMAT:
            raise ValueError(f"{self.path} is not a registered shard index")
        self.index_manifest = index
        self.content_identity = index.get("aggregate_content_sha256")
        if not isinstance(self.content_identity, str) or len(self.content_identity) != 64:
            raise ValueError(f"{self.path} has no valid aggregate content identity")
        self.forecast_population_identity = index.get("forecast_population_sha256")
        raw_shards = index.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("window shard index must contain at least one shard")
        declared_split = index.get("split")
        if declared_split is not None and not isinstance(declared_split, str):
            raise ValueError("window shard index split must be a string")
        self.collection_split: str | None = declared_split
        loaded_shards = []
        for state in raw_shards:
            if not isinstance(state, dict) or "path" not in state:
                raise ValueError("every shard entry must contain a path")
            store_path = (self.path.parent / str(state["path"])).resolve()
            with (store_path / "store.json").open(encoding="utf-8") as handle:
                store_header = json.load(handle)
            if store_header.get("format") == WINDOW_SEQUENCE_STORE_FORMAT:
                if not isinstance(state.get("split"), str):
                    raise ValueError("sequence-store shard entries must declare split")
                loaded_shards.append(NpySequenceWindowDataset(store_path, split=state["split"]))
            elif store_header.get("format") == WINDOW_STORE_FORMAT:
                loaded_shards.append(NpyWindowDataset(store_path, verify_values=False))
            else:
                raise ValueError(f"unsupported shard store format at {store_path}")
        self.shards = tuple(loaded_shards)
        declared_lengths = [int(state.get("length", -1)) for state in raw_shards]
        actual_lengths = [len(shard) for shard in self.shards]
        if declared_lengths != actual_lengths:
            raise ValueError(
                f"shard lengths differ from index: declared={declared_lengths}, "
                f"actual={actual_lengths}"
            )
        names = [shard._archive_names for shard in self.shards]
        if any(value != names[0] for value in names[1:]):
            raise ValueError("all window shards must contain the same arrays")
        self._archive_names = names[0]
        self.arrays = {
            name: _ShardedArray([shard.arrays[name] for shard in self.shards])
            for name in sorted(self._archive_names)
        }
        self._offsets = np.cumsum([0, *actual_lengths], dtype=np.int64)
        schema_path = self.path.with_suffix(".schema.json")
        if not schema_path.exists():
            raise FileNotFoundError(schema_path)
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        if not isinstance(schema, dict):
            raise TypeError(f"{schema_path} must contain a JSON object")
        self.schema: dict[str, Any] | None = schema

        signatures = []
        for shard in self.shards:
            shard.require_registered_schema()
            signatures.append(shard.registered_schema_signature())
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("window shards do not share the same registered feature schema")
        if verify_content:
            self.verify_content()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if hasattr(self, "_v2_metadata"):
            return NpzWindowDataset.__getitem__(self, index)
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._offsets, normalized) - 1
        local_index = int(normalized - self._offsets[shard_index])
        return self.shards[shard_index][local_index]

    def require_registered_schema(self) -> None:
        """Validate the collection contract without expanding every virtual window."""

        if hasattr(self, "_v2_metadata"):
            raise ValueError("v2 requires require_origin_safe_history_contract")
        audit_arrays = {"daylight_mask", "timestamps", "day_ids", "site_ids"}
        missing = sorted(audit_arrays.difference(self._archive_names))
        if missing:
            raise ValueError(f"registered archive is missing audit arrays: {missing}")
        if self.schema is None:
            raise ValueError("shard index has no registered schema")
        if self.schema.get("schema_version") != 1:
            raise ValueError("unsupported or missing prepared schema_version")
        history = tuple(self.schema.get("history_columns", ()))
        future = tuple(self.schema.get("future_known_columns", ()))
        site = tuple(self.schema.get("site_metadata_names", ()))
        if history != REGISTERED_HISTORY_FEATURES:
            raise ValueError("history schema does not match the registered feature order")
        if future != REGISTERED_FUTURE_FEATURES:
            raise ValueError(
                "future-known schema does not match the audited deterministic feature contract"
            )
        if site != REGISTERED_SITE_FEATURES:
            raise ValueError("site metadata schema does not match the registered feature order")
        if self.schema.get("target") != "power_norm":
            raise ValueError("registered target must be power_norm")
        if self.schema.get("state_labels") != REGISTERED_STATE_LABELS:
            raise ValueError("state-label mapping does not match the registered contract")
        if len(history) != self.arrays["history_values"].shape[-1]:
            raise ValueError("history schema width does not match history_values")
        if len(future) != self.arrays["future_known"].shape[-1]:
            raise ValueError("future schema width does not match future_known")
        if len(site) != self.arrays["site_metadata"].shape[-1]:
            raise ValueError("site schema width does not match site_metadata")
        target_shape = self.arrays["targets"].shape
        for name in ("daylight_mask", "timestamps", "day_ids"):
            if self.arrays[name].shape != target_shape:
                raise ValueError(f"{name} must match the target shape")

        if self.arrays["site_ids"].shape != (len(self),):
            raise ValueError("site_ids must contain one value per forecast origin")

        site_id_map = self.schema.get("site_id_map")
        if not isinstance(site_id_map, dict) or not site_id_map:
            raise ValueError("schema must contain a non-empty site_id_map")
        expected_ids = {
            key
            for shard in self.shards
            if shard.schema is not None
            for key in shard.schema["site_id_map"]
        }
        if set(site_id_map) != expected_ids:
            raise ValueError("collection site_id_map differs from its shard identities")
        values = list(site_id_map.values())
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("schema.site_id_map values must be non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError("schema.site_id_map values must be unique")
        spatial_roles = self.schema.get("spatial_roles")
        if not isinstance(spatial_roles, dict) or set(spatial_roles) != expected_ids:
            raise ValueError("schema.spatial_roles must describe every observed site")
        invalid_roles = set(spatial_roles.values()).difference(REGISTERED_SPATIAL_ROLES)
        if invalid_roles:
            raise ValueError(f"schema contains invalid spatial roles: {sorted(invalid_roles)}")
        ramp_thresholds = self.schema.get("ramp_thresholds")
        if not isinstance(ramp_thresholds, dict) or set(ramp_thresholds) != expected_ids:
            raise ValueError(
                "schema.ramp_thresholds must describe every observed site using train-only values"
            )
        for site_id, state in ramp_thresholds.items():
            if not isinstance(state, dict) or "threshold" not in state:
                raise ValueError(f"ramp threshold metadata is incomplete for site {site_id}")
            threshold = float(state["threshold"])
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"ramp threshold must be finite and positive for site {site_id}")
        for shard in self.shards:
            if shard.schema is None:  # pragma: no cover - checked by shard validation
                raise RuntimeError("validated shard unexpectedly has no schema")
            for site_id, site_name in shard.schema["site_id_map"].items():
                if site_id_map[site_id] != site_name:
                    raise ValueError("collection site_id_map conflicts with a shard schema")
                if spatial_roles[site_id] != shard.schema["spatial_roles"][site_id]:
                    raise ValueError("collection spatial role conflicts with a shard schema")
                shard_threshold = float(shard.schema["ramp_thresholds"][site_id]["threshold"])
                if not np.isclose(float(ramp_thresholds[site_id]["threshold"]), shard_threshold):
                    raise ValueError("collection ramp threshold conflicts with a shard schema")

    def require_registered_evidence(self, *, allow_ineligible: bool = False) -> None:
        """Validate both the collection sidecar and every referenced site store."""

        super().require_registered_evidence(allow_ineligible=allow_ineligible)
        if (
            not isinstance(self.forecast_population_identity, str)
            or len(self.forecast_population_identity) != 64
        ):
            raise ValueError("registered shard index has no forecast-population identity")
        assert self.schema is not None
        root_state = {
            key: self.schema[key]
            for key in (
                "dataset_kind",
                "simulated",
                "ramp_method",
                "registered_evidence_eligible",
                "preparation_provenance",
                "registered_nrel_cohort",
            )
        }
        for shard in self.shards:
            shard.require_registered_evidence(allow_ineligible=allow_ineligible)
            assert shard.schema is not None
            shard_state = {key: shard.schema[key] for key in root_state}
            if shard_state != root_state:
                raise ValueError("collection and site-store evidence provenance are inconsistent")

    def verify_content(self) -> str:
        if hasattr(self, "_v2_metadata"):
            return NpzWindowDataset.verify_content(self)
        files: list[dict[str, Any]] = []
        for shard in self.index_manifest["shards"]:
            for expected in shard.get("files", ()):
                path = self.path.parent / str(expected["path"])
                actual = {
                    "path": str(expected["path"]),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                if actual["sha256"] != expected.get("sha256") or actual["bytes"] != int(
                    expected.get("bytes", -1)
                ):
                    raise ValueError(f"content hash mismatch for shard file {path}")
                files.append(actual)
        root_expected = self.index_manifest.get("root_schema")
        if not isinstance(root_expected, dict):
            raise ValueError("shard index has no root schema identity")
        root_path = self.path.parent / str(root_expected["path"])
        root_actual = {
            "path": str(root_expected["path"]),
            "sha256": sha256_file(root_path),
            "bytes": root_path.stat().st_size,
        }
        if root_actual["sha256"] != root_expected.get("sha256") or root_actual["bytes"] != int(
            root_expected.get("bytes", -1)
        ):
            raise ValueError("content hash mismatch for shard-index schema")
        files.append(root_actual)
        identity = _aggregate_file_identities(files)
        if identity != self.content_identity:
            raise ValueError("shard-index aggregate content identity does not match")
        return identity


WindowDataset = Union[
    NpzWindowDataset,
    NpyWindowDataset,
    NpySequenceWindowDataset,
    ShardedWindowDataset,
    "WindowedPVDataset",
]


def window_dataset_temporal_bounds(dataset: WindowDataset) -> dict[str, int]:
    """Return global UTC bounds without materialising a lazy shard collection."""

    if (
        _v2_claim(getattr(dataset, "schema", None))
        or getattr(dataset, "prepared_contract", None) is not None
    ):
        raise ValueError("origin-safe chronology requires a qualified v2 consumer bridge")
    leaves = dataset.shards if isinstance(dataset, ShardedWindowDataset) else (dataset,)
    first_targets: list[int] = []
    last_targets: list[int] = []
    for leaf in leaves:
        if len(leaf) <= 0:
            raise ValueError("registered split contains no forecast windows")
        first = np.asarray(leaf.arrays["timestamps"][0], dtype=np.int64).reshape(-1)
        last = np.asarray(leaf.arrays["timestamps"][len(leaf) - 1], dtype=np.int64).reshape(-1)
        if first.size == 0 or last.size == 0:
            raise ValueError("registered timestamps cannot be empty")
        first_targets.append(int(np.min(first)))
        last_targets.append(int(np.max(last)))
    target_start = min(first_targets)
    target_end = max(last_targets)
    return {
        "history_start_ns": (target_start - REGISTERED_LOOKBACK_STEPS * REGISTERED_FREQUENCY_NS),
        "target_start_ns": target_start,
        "target_end_ns": target_end,
    }


def require_global_split_chronology(
    datasets: Mapping[str, WindowDataset],
) -> dict[str, Any]:
    """Require every later split to begin after all earlier split targets.

    This is deliberately global across sites. It prevents a spatial-transfer run
    from calling itself historical extrapolation when source data extend beyond
    the target test issue period.
    """

    ordered = [
        split for split in ("train", "validation", "calibration", "test") if split in datasets
    ]
    bounds = {split: window_dataset_temporal_bounds(datasets[split]) for split in ordered}
    checks = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        prior_end = bounds[earlier]["target_end_ns"]
        next_start = bounds[later]["history_start_ns"]
        passed = prior_end < next_start
        checks.append(
            {
                "earlier_split": earlier,
                "later_split": later,
                "earlier_target_end_ns": prior_end,
                "later_history_start_ns": next_start,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                f"global historical boundary failed: {earlier} targets end at "
                f"{prior_end}, but {later} histories begin at {next_start}; trim all "
                "sites to common calendar cutoffs before claiming historical extrapolation"
            )
    return {"bounds": bounds, "checks": checks, "passed": True}


def require_registered_spatial_protocol(
    datasets: Mapping[str, WindowDataset],
) -> dict[str, list[str]]:
    """Enforce the preregistered NREL state-role split at every model entry point."""

    if not datasets:
        raise ValueError("spatial protocol requires at least one split")
    kinds = {
        str(dataset.schema.get("dataset_kind"))
        for dataset in datasets.values()
        if dataset.schema is not None
    }
    if len(kinds) != 1:
        raise ValueError("all registered splits must use one dataset kind")
    roles: dict[str, list[str]] = {}
    for split, dataset in datasets.items():
        if dataset.schema is None:
            raise ValueError("spatial protocol requires registered schemas")
        raw_roles = dataset.schema.get("spatial_roles")
        if not isinstance(raw_roles, dict) or not raw_roles:
            raise ValueError("registered schema is missing spatial roles")
        roles[split] = sorted({str(value) for value in raw_roles.values()})
    if kinds == {"nrel"}:
        if not all(
            dataset.schema is not None
            and dataset.schema.get("registered_evidence_eligible") is True
            for dataset in datasets.values()
        ):
            return roles
        expected = {
            "train": ("source_train", "source_train_60", {"AL": 20, "CA": 20, "NY": 20}),
            "validation": (
                "validation_calibration",
                "validation_calibration_20",
                {"OK": 20},
            ),
            "calibration": (
                "validation_calibration",
                "validation_calibration_20",
                {"OK": 20},
            ),
            "test": (
                "zero_parameter_test",
                "zero_parameter_test_20",
                {"AZ": 20},
            ),
        }
        cohort_hashes = set()
        for split, observed in roles.items():
            if split not in expected:
                continue
            expected_role, expected_profile, expected_counts = expected[split]
            if observed != [expected_role]:
                raise ValueError(
                    f"registered NREL {split} requires role {[expected_role]}, received {observed}"
                )
            schema = datasets[split].schema
            assert schema is not None
            cohort = schema.get("registered_nrel_cohort")
            if not isinstance(cohort, dict):
                raise ValueError("registered NREL split has no frozen cohort metadata")
            if cohort.get("output_profile") != expected_profile:
                raise ValueError(f"registered NREL {split} requires profile {expected_profile!r}")
            observed_counts = cohort.get("output_state_counts")
            if observed_counts != expected_counts:
                raise ValueError(
                    f"registered NREL {split} requires state counts {expected_counts}, "
                    f"received {observed_counts}"
                )
            site_id_map = schema.get("site_id_map")
            site_states = schema.get("site_states")
            if not isinstance(site_id_map, dict) or not isinstance(site_states, dict):
                raise ValueError("registered NREL split lacks site/state mappings")
            actual_ids = sorted(str(value) for value in site_id_map.values())
            if actual_ids != sorted(str(value) for value in cohort.get("output_site_ids", ())):
                raise ValueError("registered NREL split differs from its frozen output IDs")
            actual_counts: dict[str, int] = {}
            for state in site_states.values():
                actual_counts[str(state)] = actual_counts.get(str(state), 0) + 1
            if dict(sorted(actual_counts.items())) != expected_counts:
                raise ValueError("registered NREL site-state mapping has wrong counts")
            cohort_hashes.add(str(cohort.get("full_cohort_sha256")))
        if len(cohort_hashes) != 1:
            raise ValueError("NREL splits were not derived from one frozen 100-site cohort")
    return roles


def load_window_dataset(
    path: str | Path, *, verify_content: bool = False, expected_version=None, expected_profile=None
) -> WindowDataset:
    """Load legacy NPZ, a single mmap store, or a global shard index."""

    source = Path(path)
    if source.is_dir():
        with (source / "store.json").open(encoding="utf-8") as handle:
            store_header = json.load(handle)
        if store_header.get("format") == WINDOW_SEQUENCE_STORE_FORMAT:
            raise ValueError(
                "a sequence-store directory contains four splits; load its split .index.json"
            )
        return NpyWindowDataset(
            source,
            verify_content=verify_content,
            expected_version=expected_version,
            expected_profile=expected_profile,
        )
    if source.suffix.lower() == ".json":
        return ShardedWindowDataset(
            source,
            verify_content=verify_content,
            expected_version=expected_version,
            expected_profile=expected_profile,
        )
    dataset = NpzWindowDataset(
        source, expected_version=expected_version, expected_profile=expected_profile
    )
    if verify_content and not hasattr(dataset, "_v2_metadata"):
        dataset.content_identity = sha256_file(source)
    return dataset


def window_dataset_content_identity(path: str | Path, *, verify: bool = True) -> str:
    """Return an identity covering all underlying shard files, not just the index."""

    source = Path(path)
    descriptor = (
        source / "store.json"
        if source.is_dir()
        else source
        if source.suffix == ".json"
        else source.with_suffix(".descriptor.json")
    )
    if descriptor.exists() and _v2_claim(_contract().read_strict_json(descriptor)):
        return _contract().validate_v2_metadata(descriptor)["descriptor"][
            "aggregate_content_sha256"
        ]
    if source.suffix == ".npz":
        sidecar = source.with_suffix(".schema.json")
        if (
            descriptor.exists()
            or (sidecar.exists() and _v2_claim(_contract().read_strict_json(sidecar)))
            or source.with_name(source.name + ".attempt").exists()
        ):
            return _contract().validate_v2_metadata(descriptor)["descriptor"][
                "aggregate_content_sha256"
            ]
    if source.is_dir():
        with (source / "store.json").open(encoding="utf-8") as handle:
            store_header = json.load(handle)
        if store_header.get("format") == WINDOW_SEQUENCE_STORE_FORMAT:
            raw = store_header.get("aggregate_content_sha256")
            if not verify:
                return str(raw)
            # Split choice is irrelevant to the underlying content identity.
            split = next(iter(store_header.get("splits", {})), None)
            if split is None:
                raise ValueError("sequence store declares no splits")
            return NpySequenceWindowDataset(source, split=split).verify_content()
        dataset = NpyWindowDataset(source, verify_values=False)
        return dataset.verify_content() if verify else dataset.content_identity
    if source.suffix.lower() == ".json":
        dataset = ShardedWindowDataset(source)
        return dataset.verify_content() if verify else dataset.content_identity
    return sha256_file(source)


def save_window_dataset_store(
    path: str | Path,
    dataset: Any,
    *,
    site_index: int,
    schema: Mapping[str, Any],
) -> int:
    """Stream one window dataset into NPY memmaps with near-constant RAM use."""

    if _v2_claim(schema) or getattr(dataset, "prepared_contract", None) is not None:
        destination = Path(path)
        attempt = _v2_attempt(destination, directory=True)
        try:
            prepared = _v2_prepared(dataset, schema, site_index)
            _v2_leaf_write(
                destination / "store.json", prepared, fmt=_contract().V2_WINDOW_STORE_FORMAT
            )
        except Exception as error:
            _v2_failure(attempt, error)
            raise
        return len(dataset)

    length = int(len(dataset))
    if length <= 0:
        raise ValueError(f"split {getattr(dataset, 'split_bounds', None)} has no valid windows")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    first = dataset[0]
    missing = [name for name in MODEL_ARRAYS if name not in first]
    if missing:
        raise KeyError(f"window dataset sample is missing arrays: {missing}")

    shapes = {name: (length, *tuple(first[name].shape)) for name in MODEL_ARRAYS}
    dtypes = {name: np.asarray(first[name].detach().cpu().numpy()).dtype for name in MODEL_ARRAYS}
    shapes.update(
        {
            "daylight_mask": (length, 24),
            "timestamps": (length, 24),
            "day_ids": (length, 24),
            "site_ids": (length,),
        }
    )
    dtypes.update(
        {
            "daylight_mask": np.dtype("float32"),
            "timestamps": np.dtype("int64"),
            "day_ids": np.dtype("int32"),
            "site_ids": np.dtype("int64"),
        }
    )
    temporary_paths = {name: destination / f".{name}.tmp.npy" for name in shapes}
    memmaps = {
        name: np.lib.format.open_memmap(
            temporary_paths[name], mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }
    try:
        for item in range(length):
            sample = first if item == 0 else dataset[item]
            for name in MODEL_ARRAYS:
                values = np.asarray(sample[name].detach().cpu().numpy())
                if values.shape != shapes[name][1:]:
                    raise ValueError(f"sample {item} changed the shape of {name}")
                if name != "state_labels" and not np.isfinite(values).all():
                    raise ValueError(f"sample {item} contains non-finite {name}")
                memmaps[name][item] = values
            history_mask = np.asarray(sample["history_mask"].detach().cpu().numpy())
            if not np.isin(history_mask, (0, 1)).all():
                raise ValueError(f"sample {item} has a non-binary history_mask")
            labels = np.asarray(sample["state_labels"].detach().cpu().numpy())
            if not np.isin(labels, (-100, 0, 1, 2)).all():
                raise ValueError(f"sample {item} has invalid state_labels")
            daylight = np.asarray(dataset.sample_daylight_mask(item), dtype=np.float32)
            if daylight.shape != (24,) or not np.isin(daylight, (0, 1)).all():
                raise ValueError(f"sample {item} has an invalid daylight mask")
            _, target_time = dataset.sample_timestamps(item)
            if len(target_time) != 24:
                raise ValueError(f"sample {item} has an invalid target timestamp horizon")
            targets = np.asarray(sample["targets"].detach().cpu().numpy())
            if np.any(labels[~daylight.astype(bool)] != -100):
                raise ValueError(f"sample {item} has non-ignored night state labels")
            if np.any(np.abs(targets[~daylight.astype(bool)]) > 1e-7):
                raise ValueError(f"sample {item} has non-zero night targets")
            memmaps["daylight_mask"][item] = daylight
            memmaps["timestamps"][item] = target_time.as_unit("ns").asi8
            memmaps["day_ids"][item] = (
                target_time.year.to_numpy() * 10_000
                + target_time.month.to_numpy() * 100
                + target_time.day.to_numpy()
            ).astype(np.int32)
            memmaps["site_ids"][item] = int(site_index)
        for values in memmaps.values():
            values.flush()
            mmap_handle = getattr(values, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        del values
        memmaps.clear()
        arrays_state = {}
        file_identities = []
        for name, temporary in temporary_paths.items():
            final_path = destination / f"{name}.npy"
            temporary.replace(final_path)
            identity = {
                "path": final_path.name,
                "sha256": sha256_file(final_path),
                "bytes": final_path.stat().st_size,
            }
            file_identities.append(identity)
            arrays_state[name] = {
                "file": final_path.name,
                "shape": list(shapes[name]),
                "dtype": str(dtypes[name]),
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
            }
        atomic_json_dump(dict(schema), destination / "schema.json")
        atomic_json_dump(
            {
                "format": WINDOW_STORE_FORMAT,
                "sample_count": length,
                "arrays": arrays_state,
                "value_validation": "complete_on_write",
                "aggregate_content_sha256": _aggregate_file_identities(file_identities),
            },
            destination / "store.json",
        )
    except Exception:
        for values in memmaps.values():
            mmap_handle = getattr(values, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        memmaps.clear()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise
    return length


def save_window_sequence_store(
    path: str | Path,
    datasets: Mapping[str, Any],
    *,
    site_index: int,
    schema: Mapping[str, Any],
) -> dict[str, int]:
    """Store one site sequence once and index all split windows lazily."""

    expected_splits = ("train", "validation", "calibration", "test")
    if tuple(datasets) != expected_splits:
        raise ValueError(f"datasets must be ordered as {expected_splits}")
    if any(len(datasets[name]) <= 0 for name in expected_splits):
        empty = [name for name in expected_splits if len(datasets[name]) <= 0]
        raise ValueError(f"sequence store cannot contain empty splits: {empty}")
    if _v2_claim(schema) or any(
        getattr(dataset, "prepared_contract", None) is not None for dataset in datasets.values()
    ):
        contract = _contract()
        destination = Path(path)
        attempt = _v2_attempt(destination, directory=True)
        try:
            prepared = {}
            for split, dataset in datasets.items():
                part_schema = schema if split == "train" else dataset.prepared_contract["schema"]
                prepared[split] = _v2_prepared(dataset, part_schema, site_index)
                contract._exact(
                    prepared[split]["schema"]["split"], split, "sequence selected split"
                )
            primary = prepared["train"]
            for part in prepared.values():
                for key in ("site", "sources", "fit_states"):
                    contract._exact(
                        part["schema"][key], primary["schema"][key], "shared sequence " + key
                    )
                for name in contract.POPULATION_ARRAYS[:13]:
                    contract._exact(
                        part["schema"]["arrays"][name],
                        primary["schema"]["arrays"][name],
                        "shared static " + name,
                    )
            shared, children = {}, []
            for split, part in prepared.items():
                child = _v2_leaf_write(
                    destination / (split + ".descriptor.json"),
                    part,
                    fmt=contract.V2_WINDOW_SEQUENCE_STORE_FORMAT,
                    shared=shared,
                )
                children.append((child, split))
            _v2_index_write(
                destination / "store.json",
                children,
                fmt=contract.V2_WINDOW_SEQUENCE_STORE_FORMAT,
                split=None,
            )
        except Exception as error:
            _v2_failure(attempt, error)
            raise
        return {name: len(dataset) for name, dataset in datasets.items()}
    primary = datasets["train"]
    frame_index = primary.frame.index
    for name in expected_splits[1:]:
        candidate = datasets[name]
        if (
            not candidate.frame.index.equals(frame_index)
            or candidate.history_columns != primary.history_columns
            or candidate.future_known_columns != primary.future_known_columns
            or candidate.site_metadata_names != primary.site_metadata_names
        ):
            raise ValueError("all split datasets must reference the same site sequence schema")

    history_values = np.nan_to_num(primary._history_values, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    history_mask = np.asarray(primary._history_mask, dtype=np.float32)
    # Canonicalize every unavailable historical cell to zero before it reaches
    # the reusable sequence store. This makes the future-safe mask contract
    # enforceable even for consumers that do not explicitly multiply by it.
    history_values = np.where(history_mask.astype(bool), history_values, 0.0).astype(
        np.float32, copy=False
    )
    base_future = np.nan_to_num(primary._base_future, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    targets = np.nan_to_num(primary._targets, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    states = np.asarray(primary._states, dtype=np.int64)
    night = np.asarray(primary._night, dtype=np.uint8)
    if not np.isfinite(history_values).all() or not np.isfinite(base_future).all():
        raise ValueError("sequence values must be finite after deterministic sanitization")
    if not np.isin(history_mask, (0, 1)).all():
        raise ValueError("sequence history mask must be binary")
    if np.any(history_values[~history_mask.astype(bool)] != 0.0):
        raise RuntimeError("masked sequence history values must be canonical zeros")
    if not np.isin(states, (-100, 0, 1, 2)).all():
        raise ValueError("sequence state labels violate the registered mapping")
    if np.any(states[night.astype(bool)] != -100) or np.any(
        np.abs(targets[night.astype(bool)]) > 1e-7
    ):
        raise ValueError("sequence night targets/labels violate the physics rule")

    local_day_ids = (
        frame_index.year.to_numpy() * 10_000
        + frame_index.month.to_numpy() * 100
        + frame_index.day.to_numpy()
    ).astype(np.int32)
    arrays: dict[str, np.ndarray] = {
        "history_values": history_values,
        "history_mask": history_mask,
        "base_future": base_future,
        "site_metadata": np.asarray(primary._site_metadata, dtype=np.float32),
        "targets": targets,
        "state_labels": states,
        "night": night,
        "timestamps": frame_index.as_unit("ns").asi8.astype(np.int64),
        "local_day_ids": local_day_ids,
    }
    split_state = {}
    counts = {}
    for name in expected_splits:
        starts = np.asarray(datasets[name].window_starts, dtype=np.int64)
        arrays[f"window_starts__{name}"] = starts
        counts[name] = int(len(starts))
        split_state[name] = {
            "sample_count": int(len(starts)),
            "bounds": list(datasets[name].split_bounds),
        }

    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    arrays_state = {}
    file_identities = []
    temporary_paths = []
    try:
        for name, values in arrays.items():
            temporary = destination / f".{name}.tmp.npy"
            final_path = destination / f"{name}.npy"
            temporary_paths.append(temporary)
            np.save(temporary, values, allow_pickle=False)
            temporary.replace(final_path)
            identity = {
                "path": final_path.name,
                "sha256": sha256_file(final_path),
                "bytes": final_path.stat().st_size,
            }
            file_identities.append(identity)
            arrays_state[name] = {
                "file": final_path.name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": identity["sha256"],
                "bytes": identity["bytes"],
            }
        atomic_json_dump(dict(schema), destination / "schema.json")
        atomic_json_dump(
            {
                "format": WINDOW_SEQUENCE_STORE_FORMAT,
                "site_index": int(site_index),
                "sequence_rows": int(len(frame_index)),
                "lookback": int(primary.lookback),
                "horizon": int(primary.horizon),
                "future_known_names": list(primary.future_known_columns),
                "base_future_names": list(primary._base_future_names),
                "splits": split_state,
                "arrays": arrays_state,
                "value_validation": "complete_on_write",
                "aggregate_content_sha256": _aggregate_file_identities(file_identities),
            },
            destination / "store.json",
        )
    except Exception:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise
    return counts


def save_window_shard_index(
    path: str | Path,
    shards: Sequence[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any],
) -> str:
    """Write a lightweight global index over per-site memory-mapped stores."""

    if not shards:
        raise ValueError("cannot create an empty window shard index")
    if _v2_claim(schema):
        contract = _contract()
        destination = Path(path)
        attempt = _v2_attempt(destination, directory=False)
        try:
            schema = contract.validate_v2_schema(dict(schema))
            children = []
            for entry in shards:
                contract._relative_name(entry["path"])
                descriptor = contract.contained_artifact_file(
                    destination.parent, entry["path"] + "/store.json"
                )
                child = contract.validate_v2_metadata(descriptor)
                split = entry.get("split")
                contract._exact(split, schema["split"], "shard selected split")
                views = [v for v in child.get("views", [child]) if v["schema"]["split"] == split]
                if len(views) != 1:
                    raise ValueError("shard requires one selected view")
                contract._exact(
                    entry["length"],
                    views[0]["schema"]["arrays"]["window_starts"]["shape"][0],
                    "shard length",
                )
                children.append((child, split))
            selected_schema = next(
                v
                for v in children[0][0].get("views", [children[0][0]])
                if v["schema"]["split"] == schema["split"]
            )["schema"]
            contract._exact(schema, selected_schema, "shard representative schema")
            metadata = _v2_index_write(
                destination,
                children,
                fmt=contract.V2_WINDOW_SHARD_INDEX_FORMAT,
                split=schema["split"],
            )
        except Exception as error:
            _v2_failure(attempt, error)
            raise
        return metadata["descriptor"]["aggregate_content_sha256"]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schema_path = destination.with_suffix(".schema.json")
    atomic_json_dump(dict(schema), schema_path)
    normalized = []
    all_files: list[dict[str, Any]] = []
    population_files: list[dict[str, Any]] = []
    for shard in shards:
        relative = str(shard.get("path", ""))
        length = int(shard.get("length", 0))
        if not relative or Path(relative).is_absolute() or length <= 0:
            raise ValueError("shard paths must be relative and lengths must be positive")
        store_path = (destination.parent / relative).resolve()
        store_manifest_path = store_path / "store.json"
        with store_manifest_path.open(encoding="utf-8") as handle:
            store_manifest = json.load(handle)
        if not isinstance(store_manifest, dict):
            raise ValueError(f"invalid shard store: {store_path}")
        store_format = store_manifest.get("format")
        if store_format == WINDOW_STORE_FORMAT:
            stored_length = int(store_manifest.get("sample_count", -1))
        elif store_format == WINDOW_SEQUENCE_STORE_FORMAT:
            split = shard.get("split")
            split_state = store_manifest.get("splits", {}).get(split, {})
            stored_length = int(split_state.get("sample_count", -1))
        else:
            stored_length = -1
        if stored_length != length:
            raise ValueError(f"invalid or length-mismatched shard store: {store_path}")
        shard_files = []
        for array_name, state in store_manifest.get("arrays", {}).items():
            file_relative = str(Path(relative) / str(state["file"]))
            file_state = {
                "path": file_relative,
                "sha256": str(state["sha256"]),
                "bytes": int(state["bytes"]),
            }
            shard_files.append(file_state)
            all_files.append(file_state)
            if array_name != "state_labels":
                population_files.append(file_state)
        for metadata_name in ("store.json", "schema.json"):
            metadata_path = store_path / metadata_name
            file_state = {
                "path": str(Path(relative) / metadata_name),
                "sha256": sha256_file(metadata_path),
                "bytes": metadata_path.stat().st_size,
            }
            shard_files.append(file_state)
            all_files.append(file_state)
        normalized.append(
            {
                **dict(shard),
                "path": relative,
                "length": length,
                "store_content_sha256": store_manifest["aggregate_content_sha256"],
                "files": shard_files,
            }
        )
    root_schema_state = {
        "path": schema_path.name,
        "sha256": sha256_file(schema_path),
        "bytes": schema_path.stat().st_size,
    }
    all_files.append(root_schema_state)
    split_names = {state.get("split") for state in normalized}
    collection_split = split_names.pop() if len(split_names) == 1 else None
    content_identity = _aggregate_file_identities(all_files)
    population_digest = hashlib.sha256()
    population_digest.update(_aggregate_file_identities(population_files).encode("ascii"))
    population_digest.update(
        json.dumps(
            {
                "split": collection_split,
                "history_columns": schema.get("history_columns"),
                "future_known_columns": schema.get("future_known_columns"),
                "site_metadata_names": schema.get("site_metadata_names"),
                "target": schema.get("target"),
                "site_id_map": schema.get("site_id_map"),
                "spatial_roles": schema.get("spatial_roles"),
                "site_states": schema.get("site_states"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    forecast_population_identity = population_digest.hexdigest()
    atomic_json_dump(
        {
            "format": WINDOW_SHARD_INDEX_FORMAT,
            "split": collection_split,
            "sample_count": int(sum(state["length"] for state in normalized)),
            "shards": normalized,
            "root_schema": root_schema_state,
            "aggregate_content_sha256": content_identity,
            "forecast_population_sha256": forecast_population_identity,
            "forecast_population_excludes": [
                "state_labels",
                "ramp_threshold_metadata",
                "ramp_method",
            ],
        },
        destination,
    )
    return content_identity


def save_windows_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    schema: Mapping[str, Any] | None = None,
    causal_arrays=None,
    prepared_contract=None,
) -> None:
    if _v2_claim(schema) or causal_arrays is not None or prepared_contract is not None:
        contract = _contract()
        destination = Path(path)
        evidence = (
            prepared_contract.get("evidence", {}) if isinstance(prepared_contract, Mapping) else {}
        )
        references = [
            *evidence,
            destination.with_suffix(".schema.json").name,
            destination.with_suffix(".descriptor.json").name,
            destination.with_suffix(".receipt.json").name,
        ]
        attempt = _v2_attempt(destination, directory=False, references=references)
        try:
            contract._keys(
                prepared_contract, ("schema", "causal_arrays", "evidence"), "NPZ prepared_contract"
            )
            contract._exact(dict(schema), prepared_contract["schema"], "NPZ schema")
            contract.validate_preparation_contract(
                dict(schema), causal_arrays, evidence=prepared_contract["evidence"]
            )
            contract.validate_preparation_contract(
                dict(schema),
                prepared_contract["causal_arrays"],
                evidence=prepared_contract["evidence"],
            )
            if not set(MODEL_ARRAYS) <= set(arrays) <= set(MODEL_ARRAYS) | set(AUDIT_ARRAYS):
                raise ValueError("NPZ model/audit arrays must use the public whitelist")
            for name, array in arrays.items():
                if contract._array_identity(array) != contract._array_identity(causal_arrays[name]):
                    raise ValueError("NPZ model input identity mismatch: " + name)
            _v2_leaf_write(
                destination.with_suffix(".descriptor.json"),
                {**prepared_contract, "causal_arrays": causal_arrays},
                fmt="npz",
            )
        except Exception as error:
            _v2_failure(attempt, error)
            raise
        return
    missing = [name for name in MODEL_ARRAYS if name not in arrays]
    if missing:
        raise KeyError(f"cannot save an incomplete window archive: {missing}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)
    if schema is not None:
        atomic_json_dump(dict(schema), destination.with_suffix(".schema.json"))


def save_predictions_npz(
    path: str | Path,
    tensors: Mapping[str, Tensor | np.ndarray],
    metadata: Mapping[str, object] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
        for name, value in tensors.items()
    }
    np.savez_compressed(destination, **arrays)
    if metadata is not None:
        with destination.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(dict(metadata), handle, ensure_ascii=False, indent=2)
