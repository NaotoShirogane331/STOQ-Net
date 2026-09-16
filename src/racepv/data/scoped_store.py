"""Pinned physical NPY storage for one complete schema-3 split."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import stat
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import racepv.io as window_io
from racepv.data import prepared_contract as contract
from racepv.data import scoped_prepared

LEAF_FORMAT = "racepv-scoped-window-store-v3"
INDEX_FORMAT = "racepv-scoped-window-shard-index-v3"
_LEAF_FIELDS = (
    "artifact_kind",
    "schema_version",
    "causality_profile",
    "format",
    "schema",
    "files",
    "arrays",
    "file_scopes",
    "read_scope_sha256",
    "aggregate_content_sha256",
    "forecast_population_sha256",
)
_INDEX_FIELDS = (
    "artifact_kind",
    "schema_version",
    "causality_profile",
    "format",
    "selected_split",
    "leaves",
    "files",
    "file_scopes",
    "read_scope_sha256",
    "aggregate_content_sha256",
    "forecast_population_sha256",
)
_RECEIPT_FIELDS = ("artifact_kind", "schema_version", "descriptor")
_SCOPE_FIELDS = ("path", "role", "splits")


def _allowed_splits(value):
    if type(value) is not tuple or not value:
        raise TypeError("allowed_splits must be a nonempty tuple")
    if any(type(item) is not str for item in value):
        raise TypeError("allowed_splits entries must be strict strings")
    if len(set(value)) != len(value):
        raise ValueError("allowed_splits must be unique")
    try:
        positions = [contract.SPLITS.index(item) for item in value]
    except ValueError as error:
        raise ValueError("allowed_splits contains an unknown split") from error
    if positions != sorted(positions):
        raise ValueError("allowed_splits must use canonical order")
    return value


def _pins(descriptor_sha256, read_scope_sha256, allowed_splits):
    contract._sha(descriptor_sha256, "expected_descriptor_sha256")
    contract._sha(read_scope_sha256, "expected_read_scope_sha256")
    return _allowed_splits(allowed_splits)


def _absolute(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("artifact path must not contain parent traversal")
    return path.absolute()


def _reject_link_components(path):
    path = _absolute(path)
    for component in reversed((path, *path.parents)):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("artifact path contains a symlink component")
    return path


@contextmanager
def _open_held(path):
    path = _reject_link_components(path)
    with window_io._v2_directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            state = os.fstat(descriptor)
            if not stat.S_ISREG(state.st_mode):
                raise ValueError("artifact member is not a regular file")
            if state.st_nlink != 1:
                raise ValueError("artifact member is multiply linked")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                yield handle, state
        finally:
            os.close(descriptor)


@contextmanager
def _open_content(path):
    """Held, verified no-follow content seam used by full loading."""
    with _open_held(path) as opened:
        yield opened


def _hash_handle(handle):
    digest = hashlib.sha256()
    handle.seek(0)
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _read_metadata(path, *, expected=None, expected_sha256=None, cache):
    path = _absolute(path)
    cached = cache.get(path)
    if cached is None:
        with _open_held(path) as (handle, state):
            data = handle.read()
        cached = (data, state.st_size, state.st_dev, state.st_ino)
        cache[path] = cached
    data, size, _, _ = cached
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("external descriptor pin mismatch")
    if expected is not None:
        contract._file_identity(expected)
        if expected["bytes"] != size or expected["sha256"] != digest:
            raise ValueError("metadata file identity mismatch")
    try:
        return contract._strict_json_value(data.decode("utf-8")), data
    except UnicodeDecodeError as error:
        raise ValueError("metadata is not UTF-8 JSON") from error


def _member(root, relative):
    contract._relative_name(relative)
    return _reject_link_components(_absolute(root) / Path(*relative.split("/")))


def _stat_member(root, relative):
    path = _member(root, relative)
    with window_io._v2_directory(path.parent) as parent:
        state = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISLNK(state.st_mode):
        raise ValueError("artifact member is a symlink")
    if not stat.S_ISREG(state.st_mode):
        raise ValueError("artifact member is not a regular file")
    if state.st_nlink != 1:
        raise ValueError("artifact member is multiply linked")
    return path, (state.st_dev, state.st_ino)


def _file_list(value, name):
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    result = []
    for item in value:
        contract._file_identity(item)
        result.append(copy.deepcopy(item))
    paths = [item["path"] for item in result]
    if paths != sorted(paths):
        raise ValueError(f"{name} must be sorted by path")
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError(f"{name} contains duplicate or case-aliased paths")
    return result


def _scope_list(value, files, name):
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    result = []
    for entry in value:
        contract._keys(entry, _SCOPE_FIELDS, "file scope")
        contract._relative_name(entry["path"])
        if type(entry["role"]) is not str or not entry["role"]:
            raise TypeError("file scope role must be a strict nonempty string")
        if type(entry["splits"]) is not list:
            raise TypeError("file scope splits must be a list")
        if any(type(split) is not str or split not in contract.SPLITS for split in entry["splits"]):
            raise ValueError("file scope contains an invalid split")
        if len(set(entry["splits"])) != len(entry["splits"]):
            raise ValueError("file scope splits must be unique")
        result.append(copy.deepcopy(entry))
    paths = [entry["path"] for entry in result]
    if paths != sorted(paths) or paths != [item["path"] for item in files]:
        raise ValueError(f"{name} must correspond exactly to sorted files")
    return result


def _read_scope_sha256(files, scopes):
    by_path = {item["path"]: item for item in files}
    return contract.canonical_json_sha256(
        {
            "schema_version": 3,
            "files": [
                {
                    "file": by_path[entry["path"]],
                    "role": entry["role"],
                    "splits": entry["splits"],
                }
                for entry in scopes
            ],
        }
    )


def _derived_leaf_roles(schema, arrays):
    split = schema["split"]
    roles = {"store.schema.json": ("schema", [])}
    for name, reference in arrays.items():
        roles[reference["path"]] = (f"array:{name}", [split])
    source_roles = {
        "time_evidence": ("time_evidence", []),
        "producer": ("producer", []),
        "site_metadata": ("site_metadata", []),
        "raw_input": ("raw_input", [split]),
        "frozen_fit_raw_input": ("frozen_fit_raw_input", ["train"]),
    }
    for name, identity in schema["sources"].items():
        roles[identity["path"]] = source_roles[name]
    for name, state in schema["fit_states"].items():
        roles[state["file"]["path"]] = (f"fit_state:{name}", [])
        for field, prefix in (("fit", "fit_input"), ("training_audit", "training_audit")):
            record = state[field]
            if record is not None:
                roles[record["input_file"]["path"]] = (f"{prefix}:{name}", ["train"])
    expected_count = 1 + len(arrays) + len(contract._schema_evidence(schema))
    if len(roles) != expected_count:
        raise ValueError("physical file roles alias")
    return roles


def _leaf_scopes(schema, arrays, files):
    roles = _derived_leaf_roles(schema, arrays)
    if set(roles) != {item["path"] for item in files}:
        raise ValueError("leaf files do not match the complete role graph")
    return [
        {"path": item["path"], "role": roles[item["path"]][0], "splits": roles[item["path"]][1]}
        for item in files
    ]


def _check_receipt(path, descriptor_identity, cache, expected_identity=None):
    receipt_path = path.with_name(contract._receipt_name(path))
    receipt, _ = _read_metadata(receipt_path, expected=expected_identity, cache=cache)
    contract._keys(receipt, _RECEIPT_FIELDS, "receipt")
    contract._exact(receipt["artifact_kind"], "racepv.prepared-receipt", "receipt kind")
    contract._exact(receipt["schema_version"], 3, "receipt schema_version")
    contract._file_identity(receipt["descriptor"])
    contract._exact(receipt["descriptor"], descriptor_identity, "receipt descriptor")
    return receipt_path, receipt


def _check_physical_graph(root, files):
    inodes = {}
    resolved = {}
    for item in files:
        path, inode = _stat_member(root, item["path"])
        if inode in inodes:
            raise ValueError("physical file roles alias")
        inodes[inode] = item["path"]
        resolved[item["path"]] = path
    return resolved


def _actual_files(root):
    root = _reject_link_components(root)
    result = set()
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory = Path(directory)
        for name in names:
            if (directory / name).is_symlink():
                raise ValueError("artifact graph contains a symlink")
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                raise ValueError("artifact graph contains a symlink")
            result.add(path.relative_to(root).as_posix())
    return result


def _authorize(scopes, allowed_splits):
    required = {split for entry in scopes for split in entry["splits"]}
    if not required.issubset(set(allowed_splits)):
        raise ValueError("scoped graph includes unauthorized content dependencies")


def _admit_leaf(
    path,
    descriptor,
    descriptor_identity,
    allowed_splits,
    cache,
    expected_receipt_identity=None,
):
    if path.name != "store.json":
        raise ValueError("leaf completion descriptor must be store.json")
    contract._keys(descriptor, _LEAF_FIELDS, "scoped leaf descriptor")
    contract._exact(descriptor["artifact_kind"], "racepv.prepared-descriptor", "artifact kind")
    contract._exact(descriptor["schema_version"], 3, "schema_version")
    contract._exact(descriptor["causality_profile"], contract.PROFILE, "causality_profile")
    contract._exact(descriptor["format"], LEAF_FORMAT, "format")
    contract._exact(descriptor["schema"], "store.schema.json", "schema reference")
    receipt_path, receipt = _check_receipt(
        path, descriptor_identity, cache, expected_receipt_identity
    )
    files = _file_list(descriptor["files"], "leaf files")
    contract._keys(descriptor["arrays"], contract.ARRAY_ROLES, "descriptor arrays")
    arrays = copy.deepcopy(descriptor["arrays"])
    array_paths = []
    for role, reference in arrays.items():
        contract._keys(reference, ("path", "member"), f"array reference {role}")
        contract._relative_name(reference["path"])
        if reference["member"] is not None or reference["path"] != f"arrays/{role}.npy":
            raise ValueError("scoped arrays require one canonical NPY file per role")
        array_paths.append(reference["path"])
    if len(set(array_paths)) != len(array_paths):
        raise ValueError("array file roles alias")
    file_by_path = {item["path"]: item for item in files}
    schema_identity = file_by_path.get("store.schema.json")
    if schema_identity is None:
        raise ValueError("leaf schema is absent from files")
    schema, _ = _read_metadata(
        path.parent / "store.schema.json", expected=schema_identity, cache=cache
    )
    schema = scoped_prepared.validate_v3_schema(schema)
    for identity in contract._schema_evidence(schema):
        if file_by_path.get(identity["path"]) != identity:
            raise ValueError("schema evidence identity mismatch")
    scopes = _scope_list(descriptor["file_scopes"], files, "leaf file_scopes")
    if scopes != _leaf_scopes(schema, arrays, files):
        raise ValueError("leaf file scopes disagree with derived roles")
    contract._sha(descriptor["read_scope_sha256"], "read_scope_sha256")
    if descriptor["read_scope_sha256"] != _read_scope_sha256(files, scopes):
        raise ValueError("leaf read_scope_sha256 mismatch")
    contract._sha(descriptor["aggregate_content_sha256"], "aggregate_content_sha256")
    if descriptor["aggregate_content_sha256"] != window_io._aggregate_file_identities(files):
        raise ValueError("leaf aggregate_content_sha256 mismatch")
    contract._sha(descriptor["forecast_population_sha256"], "forecast_population_sha256")
    if descriptor["forecast_population_sha256"] != schema["forecast_population_sha256"]:
        raise ValueError("leaf forecast population mismatch")
    resolved = _check_physical_graph(path.parent, files)
    metadata_paths = [schema["sources"]["site_metadata"]["path"]]
    metadata_paths.extend(state["file"]["path"] for state in schema["fit_states"].values())
    documents = {}
    for relative in metadata_paths:
        value, _ = _read_metadata(resolved[relative], expected=file_by_path[relative], cache=cache)
        documents[relative] = value
    contract._validate_state_documents(schema, documents)
    _authorize(scopes, allowed_splits)
    expected_actual = {item["path"] for item in files} | {path.name, receipt_path.name}
    if _actual_files(path.parent) != expected_actual:
        raise ValueError("leaf contains orphan or missing files")
    return {
        "kind": "leaf",
        "path": path,
        "root": path.parent,
        "descriptor": descriptor,
        "descriptor_identity": descriptor_identity,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "schema": schema,
        "arrays": arrays,
        "files": files,
        "scopes": scopes,
        "resolved": resolved,
        "children": [],
    }


def _relative_to_root(path, root):
    path = _absolute(path)
    root = _absolute(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("index child escapes its parent") from error


def _admit_index(path, descriptor, descriptor_identity, allowed_splits, cache, ancestors):
    contract._keys(descriptor, _INDEX_FIELDS, "scoped index descriptor")
    contract._exact(descriptor["artifact_kind"], "racepv.prepared-index", "artifact kind")
    contract._exact(descriptor["schema_version"], 3, "schema_version")
    contract._exact(descriptor["causality_profile"], contract.PROFILE, "causality_profile")
    contract._exact(descriptor["format"], INDEX_FORMAT, "format")
    split = descriptor["selected_split"]
    if type(split) is not str or split not in contract.SPLITS:
        raise ValueError("index selected_split is invalid")
    receipt_path, receipt = _check_receipt(path, descriptor_identity, cache)
    files = _file_list(descriptor["files"], "index files")
    scopes = _scope_list(descriptor["file_scopes"], files, "index file_scopes")
    contract._sha(descriptor["read_scope_sha256"], "read_scope_sha256")
    if descriptor["read_scope_sha256"] != _read_scope_sha256(files, scopes):
        raise ValueError("index read_scope_sha256 mismatch")
    contract._sha(descriptor["aggregate_content_sha256"], "aggregate_content_sha256")
    if descriptor["aggregate_content_sha256"] != window_io._aggregate_file_identities(files):
        raise ValueError("index aggregate_content_sha256 mismatch")
    file_by_path = {item["path"]: item for item in files}
    if type(descriptor["leaves"]) is not list or not descriptor["leaves"]:
        raise ValueError("index leaves must be a nonempty list")
    children = []
    site_ids = []
    descriptor_paths = []
    for entry in descriptor["leaves"]:
        contract._keys(
            entry,
            (
                "descriptor",
                "site_id",
                "split",
                "aggregate_content_sha256",
                "forecast_population_sha256",
                "read_scope_sha256",
            ),
            "index leaf",
        )
        contract._file_identity(entry["descriptor"])
        if type(entry["site_id"]) is not int:
            raise TypeError("leaf site_id must be an integer")
        contract._exact(entry["split"], split, "leaf split")
        for field in (
            "aggregate_content_sha256",
            "forecast_population_sha256",
            "read_scope_sha256",
        ):
            contract._sha(entry[field], f"leaf {field}")
        child_path = _member(path.parent, entry["descriptor"]["path"])
        if child_path == path or child_path in ancestors:
            raise ValueError("descriptor reference cycle")
        receipt_relative = _relative_to_root(
            child_path.with_name(contract._receipt_name(child_path)), path.parent
        )
        expected_receipt_identity = file_by_path.get(receipt_relative)
        if expected_receipt_identity is None:
            raise ValueError("child receipt identity is missing from parent catalog")
        child = _admit(
            child_path,
            entry["descriptor"]["sha256"],
            entry["read_scope_sha256"],
            allowed_splits,
            cache,
            (*ancestors, path),
            require_leaf=True,
            expected_receipt_identity=expected_receipt_identity,
        )
        if entry["descriptor"]["bytes"] != child["descriptor_identity"]["bytes"]:
            raise ValueError("child descriptor byte identity mismatch")
        if entry["site_id"] != child["schema"]["site"]["id"]:
            raise ValueError("child site ID mismatch")
        if entry["split"] != child["schema"]["split"]:
            raise ValueError("child split mismatch")
        if entry["aggregate_content_sha256"] != child["descriptor"]["aggregate_content_sha256"]:
            raise ValueError("child aggregate identity mismatch")
        if entry["forecast_population_sha256"] != child["descriptor"]["forecast_population_sha256"]:
            raise ValueError("child population identity mismatch")
        children.append(child)
        site_ids.append(entry["site_id"])
        descriptor_paths.append(entry["descriptor"]["path"])
    if site_ids != sorted(site_ids) or len(set(site_ids)) != len(site_ids):
        raise ValueError("index sites must be unique and ordered")
    if len({item.casefold() for item in descriptor_paths}) != len(descriptor_paths):
        raise ValueError("index child descriptor paths alias")
    expected_files = {}
    expected_scopes = {}
    for child in children:
        for item, scope in zip(child["files"], child["scopes"], strict=True):
            relative = _relative_to_root(child["root"] / item["path"], path.parent)
            expected_files[relative] = {**item, "path": relative}
            expected_scopes[relative] = {**scope, "path": relative}
        for child_path, role in (
            (child["path"], "child_descriptor"),
            (child["receipt_path"], "child_receipt"),
        ):
            relative = _relative_to_root(child_path, path.parent)
            identity = _identity_for_path(child_path, path.parent)
            expected_files[relative] = identity
            expected_scopes[relative] = {"path": relative, "role": role, "splits": []}
    expected_file_list = [expected_files[name] for name in sorted(expected_files)]
    expected_scope_list = [expected_scopes[name] for name in sorted(expected_scopes)]
    if files != expected_file_list or scopes != expected_scope_list:
        raise ValueError("index catalog is not the exact transformed child union")
    population = {
        "schema_version": 3,
        "causality_profile": contract.PROFILE,
        "format": INDEX_FORMAT,
        "selected_split": split,
        "leaves": [
            {
                "site_id": entry["site_id"],
                "split": entry["split"],
                "forecast_population_sha256": entry["forecast_population_sha256"],
            }
            for entry in descriptor["leaves"]
        ],
    }
    if descriptor["forecast_population_sha256"] != contract.canonical_json_sha256(population):
        raise ValueError("index forecast population mismatch")
    resolved = _check_physical_graph(path.parent, files)
    _authorize(scopes, allowed_splits)
    expected_actual = {item["path"] for item in files} | {path.name, receipt_path.name}
    if _actual_files(path.parent) != expected_actual:
        raise ValueError("index contains orphan or missing files")
    return {
        "kind": "index",
        "path": path,
        "root": path.parent,
        "descriptor": descriptor,
        "descriptor_identity": descriptor_identity,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "files": files,
        "scopes": scopes,
        "resolved": resolved,
        "children": children,
    }


def _admit(
    descriptor_path,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    allowed_splits,
    cache,
    ancestors=(),
    *,
    require_leaf=False,
    expected_receipt_identity=None,
):
    path = _absolute(descriptor_path)
    descriptor, encoded = _read_metadata(
        path, expected_sha256=expected_descriptor_sha256, cache=cache
    )
    descriptor_identity = {
        "path": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
    if descriptor_identity["sha256"] != expected_descriptor_sha256:
        raise ValueError("external descriptor pin mismatch")
    if descriptor.get("read_scope_sha256") != expected_read_scope_sha256:
        raise ValueError("external read-scope pin mismatch")
    kind = descriptor.get("artifact_kind")
    if kind == "racepv.prepared-descriptor":
        node = _admit_leaf(
            path,
            descriptor,
            descriptor_identity,
            allowed_splits,
            cache,
            expected_receipt_identity,
        )
    elif kind == "racepv.prepared-index" and not require_leaf:
        node = _admit_index(path, descriptor, descriptor_identity, allowed_splits, cache, ancestors)
    else:
        raise ValueError("unknown or nested artifact kind; legacy fallback is forbidden")
    return node


def _public_metadata(node, *, verified=False):
    leaves = node["children"] if node["kind"] == "index" else [node]
    public_leaves = []
    for leaf in leaves:
        public_leaves.append(
            {
                "descriptor_path": str(leaf["path"]),
                "descriptor_sha256": leaf["descriptor_identity"]["sha256"],
                "read_scope_sha256": leaf["descriptor"]["read_scope_sha256"],
                "aggregate_content_sha256": leaf["descriptor"]["aggregate_content_sha256"],
                "forecast_population_sha256": leaf["descriptor"]["forecast_population_sha256"],
                "schema": copy.deepcopy(leaf["schema"]),
                "arrays": copy.deepcopy(leaf["arrays"]),
            }
        )
    by_scope = {item["path"]: item for item in node["scopes"]}
    files = [
        {
            **copy.deepcopy(item),
            "resolved_path": str(node["resolved"][item["path"]]),
            "role": by_scope[item["path"]]["role"],
            "splits": copy.deepcopy(by_scope[item["path"]]["splits"]),
        }
        for item in node["files"]
    ]
    return {
        "descriptor_path": str(node["path"]),
        "descriptor_sha256": node["descriptor_identity"]["sha256"],
        "read_scope_sha256": node["descriptor"]["read_scope_sha256"],
        "aggregate_content_sha256": node["descriptor"]["aggregate_content_sha256"],
        "forecast_population_sha256": node["descriptor"]["forecast_population_sha256"],
        "selected_split": (
            node["descriptor"]["selected_split"]
            if node["kind"] == "index"
            else node["schema"]["split"]
        ),
        "leaves": public_leaves,
        "files": files,
        "content_verified": verified,
        "array_semantics_verified": verified,
        "source_interpretation_verified": False,
        "registered_evidence_eligible": False,
    }


def inspect_scoped_store(
    descriptor_path,
    *,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    allowed_splits,
) -> dict:
    """Inspect a pinned strict metadata graph without opening content dependencies."""
    allowed = _pins(expected_descriptor_sha256, expected_read_scope_sha256, allowed_splits)
    node = _admit(
        descriptor_path,
        expected_descriptor_sha256,
        expected_read_scope_sha256,
        allowed,
        {},
    )
    return _public_metadata(node)


def _verify_current(identity, path):
    with _open_content(path) as (handle, state):
        digest = _hash_handle(handle)
        if state.st_size != identity["bytes"] or digest != identity["sha256"]:
            raise ValueError("content file identity mismatch")


def _load_npy(identity, path, specification):
    with _open_content(path) as (handle, state):
        digest = _hash_handle(handle)
        if state.st_size != identity["bytes"] or digest != identity["sha256"]:
            raise ValueError("NPY file identity mismatch")
        handle.seek(0)
        try:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError("unsupported NPY version")
        except (EOFError, OSError, TypeError, ValueError) as error:
            raise ValueError("invalid NPY header") from error
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise ValueError("object, structured, and subarray NPY dtypes are forbidden")
        if type(shape) is not tuple or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("invalid NPY shape")
        if list(shape) != specification["shape"] or str(dtype) != specification["dtype"]:
            raise ValueError("NPY header disagrees with schema")
        count = math.prod(shape)
        payload_size = count * dtype.itemsize
        if count > sys.maxsize or dtype.itemsize > sys.maxsize or payload_size > sys.maxsize:
            raise ValueError("NPY extent exceeds the addressable platform range")
        offset = handle.tell()
        expected_size = offset + payload_size
        if expected_size > sys.maxsize:
            raise ValueError("NPY extent exceeds the addressable platform range")
        if state.st_size != expected_size:
            raise ValueError("NPY payload is truncated or has trailing bytes")
        if count == 0:
            result = np.empty(shape, dtype=dtype, order="F" if fortran_order else "C")
            result.setflags(write=False)
        else:
            result = np.memmap(
                handle,
                dtype=dtype,
                mode="r",
                offset=offset,
                shape=shape,
                order="F" if fortran_order else "C",
            )
        if contract._bounded_array_identity(result) != specification:
            raise ValueError("logical array identity mismatch")
        return result


def _load_leaf(leaf):
    file_by_path = {item["path"]: item for item in leaf["files"]}
    array_paths = {reference["path"] for reference in leaf["arrays"].values()}
    for item in leaf["files"]:
        if item["path"] not in array_paths:
            _verify_current(item, leaf["resolved"][item["path"]])
    arrays = {
        role: _load_npy(
            file_by_path[reference["path"]],
            leaf["resolved"][reference["path"]],
            leaf["schema"]["arrays"][role],
        )
        for role, reference in leaf["arrays"].items()
    }
    dataset = scoped_prepared.build_scoped_window_dataset(leaf["schema"], arrays)
    return dataset, arrays


def load_scoped_store(
    descriptor_path,
    *,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    allowed_splits,
) -> dict:
    """Reinspect, verify all current bytes, and reconstruct every selected leaf."""
    allowed = _pins(expected_descriptor_sha256, expected_read_scope_sha256, allowed_splits)
    node = _admit(
        descriptor_path,
        expected_descriptor_sha256,
        expected_read_scope_sha256,
        allowed,
        {},
    )
    leaves = node["children"] if node["kind"] == "index" else [node]
    datasets = []
    arrays = []
    for leaf in leaves:
        dataset, causal_arrays = _load_leaf(leaf)
        datasets.append(dataset)
        arrays.append(causal_arrays)
    return {
        "metadata": _public_metadata(node, verified=True),
        "datasets": tuple(datasets),
        "causal_arrays": tuple(arrays),
    }


def _identity_for_path(path, root):
    return window_io._v2_file(_absolute(path), _absolute(root))


def _write_member(root, relative, data):
    contract._relative_name(relative)
    if type(data) is not bytes:
        raise TypeError("artifact member data must be bytes")
    window_io._v2_write_bytes(_absolute(root) / Path(*relative.split("/")), data)


def _write_npy(root, relative, array):
    contract._relative_name(relative)
    path = _absolute(root) / Path(*relative.split("/"))
    with window_io._v2_open_new(path) as handle:
        np.save(handle, array, allow_pickle=False)


def _validate_publication_metadata(schema, evidence):
    schema = scoped_prepared.validate_v3_schema(schema)
    expected_evidence = contract._schema_evidence(schema)
    contract._keys(evidence, [item["path"] for item in expected_evidence], "preparation evidence")
    for item in expected_evidence:
        value = evidence[item["path"]]
        if type(value) is not bytes:
            raise TypeError("preparation evidence must contain bytes")
        contract._verify_bytes(item, value)
    state_paths = [
        schema["sources"]["site_metadata"]["path"],
        *(state["file"]["path"] for state in schema["fit_states"].values()),
    ]
    documents = {
        path: contract._strict_json_value(evidence[path].decode("utf-8")) for path in state_paths
    }
    contract._validate_state_documents(schema, documents)
    return schema


def _validate_publication(schema, causal_arrays, evidence):
    schema = _validate_publication_metadata(schema, evidence)
    contract._keys(causal_arrays, contract.ARRAY_ROLES, "causal_arrays")
    scoped_prepared.verify_v3_semantics(schema, causal_arrays)
    return schema


def _publication_result(path, descriptor):
    encoded = window_io._v2_json_bytes(descriptor)
    return {
        "descriptor_path": str(path),
        "descriptor_sha256": hashlib.sha256(encoded).hexdigest(),
        "read_scope_sha256": descriptor["read_scope_sha256"],
        "aggregate_content_sha256": descriptor["aggregate_content_sha256"],
        "forecast_population_sha256": descriptor["forecast_population_sha256"],
        "registered_evidence_eligible": False,
    }


def _publish_descriptor(path, descriptor):
    encoded = window_io._v2_json_bytes(descriptor)
    identity = {
        "path": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
    receipt = {
        "artifact_kind": "racepv.prepared-receipt",
        "schema_version": 3,
        "descriptor": identity,
    }
    receipt_path = path.with_name(contract._receipt_name(path))
    window_io._v2_write_bytes(receipt_path, window_io._v2_json_bytes(receipt))
    window_io._v2_write_bytes(path, encoded)


def publish_scoped_leaf(output_directory, *, schema, causal_arrays, evidence) -> dict:
    """Exclusively publish one complete split as physical NPY files."""
    schema = _validate_publication(schema, causal_arrays, evidence)
    output = _absolute(output_directory)
    attempt = window_io._v2_attempt(output, directory=True)
    try:
        _write_member(output, "store.schema.json", window_io._v2_json_bytes(schema))
        for role in contract.ARRAY_ROLES:
            _write_npy(output, f"arrays/{role}.npy", causal_arrays[role])
        for relative in sorted(evidence):
            _write_member(output, relative, evidence[relative])
        identities = [
            _identity_for_path(output / Path(*relative.split("/")), output)
            for relative in _declared_leaf_paths(schema)
        ]
        identities.sort(key=lambda item: item["path"])
        arrays = {
            role: {"path": f"arrays/{role}.npy", "member": None} for role in contract.ARRAY_ROLES
        }
        scopes = _leaf_scopes(schema, arrays, identities)
        descriptor = {
            "artifact_kind": "racepv.prepared-descriptor",
            "schema_version": 3,
            "causality_profile": contract.PROFILE,
            "format": LEAF_FORMAT,
            "schema": "store.schema.json",
            "files": identities,
            "arrays": arrays,
            "file_scopes": scopes,
            "read_scope_sha256": _read_scope_sha256(identities, scopes),
            "aggregate_content_sha256": window_io._aggregate_file_identities(identities),
            "forecast_population_sha256": schema["forecast_population_sha256"],
        }
        path = output / "store.json"
        _publish_descriptor(path, descriptor)
        return _publication_result(path, descriptor)
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise


def _declared_leaf_paths(schema):
    yield "store.schema.json"
    for role in contract.ARRAY_ROLES:
        yield f"arrays/{role}.npy"
    yield from (item["path"] for item in contract._schema_evidence(schema))


def _check_unused_destination(path):
    window_io._v2_check_destination(path)
    receipt = path.with_name(contract._receipt_name(path))
    window_io._v2_check_destination(receipt)
    if path.exists() or path.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise FileExistsError(path)


def publish_scoped_index(descriptor_path, *, leaves, allowed_splits) -> dict:
    """Exclusively publish a pinned one-level, same-split leaf index."""
    allowed = _allowed_splits(allowed_splits)
    if not isinstance(leaves, Sequence) or isinstance(leaves, (str, bytes)) or not leaves:
        raise ValueError("leaves must be a nonempty sequence")
    path = _absolute(descriptor_path)
    _check_unused_destination(path)
    admitted = []
    seen_paths = set()
    for leaf in leaves:
        if type(leaf) is not dict:
            raise TypeError("each leaf pin must be a dict")
        contract._keys(
            leaf,
            ("descriptor_path", "expected_descriptor_sha256", "expected_read_scope_sha256"),
            "leaf pin",
        )
        child_path = _absolute(leaf["descriptor_path"])
        _relative_to_root(child_path, path.parent)
        if child_path.parent == path.parent:
            raise ValueError("index leaves must be in strict child directories")
        if str(child_path).casefold() in seen_paths:
            raise ValueError("duplicate leaf path")
        seen_paths.add(str(child_path).casefold())
        _pins(
            leaf["expected_descriptor_sha256"],
            leaf["expected_read_scope_sha256"],
            allowed,
        )
        child = _admit(
            child_path,
            leaf["expected_descriptor_sha256"],
            leaf["expected_read_scope_sha256"],
            allowed,
            {},
            require_leaf=True,
        )
        _load_leaf(child)
        admitted.append(child)
    admitted.sort(key=lambda child: child["schema"]["site"]["id"])
    site_ids = [child["schema"]["site"]["id"] for child in admitted]
    splits = {child["schema"]["split"] for child in admitted}
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("index leaf sites must be unique")
    if len(splits) != 1:
        raise ValueError("index leaves must have one selected split")
    split = next(iter(splits))
    if split not in allowed:
        raise ValueError("selected split is not allowed")
    files = {}
    scopes = {}
    entries = []
    for child in admitted:
        for item, scope in zip(child["files"], child["scopes"], strict=True):
            relative = _relative_to_root(child["root"] / item["path"], path.parent)
            files[relative] = {**item, "path": relative}
            scopes[relative] = {**scope, "path": relative}
        for child_path, role in (
            (child["path"], "child_descriptor"),
            (child["receipt_path"], "child_receipt"),
        ):
            relative = _relative_to_root(child_path, path.parent)
            files[relative] = _identity_for_path(child_path, path.parent)
            scopes[relative] = {"path": relative, "role": role, "splits": []}
        descriptor_identity = _identity_for_path(child["path"], path.parent)
        entries.append(
            {
                "descriptor": descriptor_identity,
                "site_id": child["schema"]["site"]["id"],
                "split": split,
                "aggregate_content_sha256": child["descriptor"]["aggregate_content_sha256"],
                "forecast_population_sha256": child["descriptor"]["forecast_population_sha256"],
                "read_scope_sha256": child["descriptor"]["read_scope_sha256"],
            }
        )
    if len(files) != sum(len(child["files"]) + 2 for child in admitted):
        raise ValueError("index child file paths alias")
    if len({name.casefold() for name in files}) != len(files):
        raise ValueError("index child file paths case-alias")
    file_list = [files[name] for name in sorted(files)]
    scope_list = [scopes[name] for name in sorted(scopes)]
    if _actual_files(path.parent) != set(files):
        raise ValueError("prospective index parent contains unrelated or missing files")
    population = {
        "schema_version": 3,
        "causality_profile": contract.PROFILE,
        "format": INDEX_FORMAT,
        "selected_split": split,
        "leaves": [
            {
                "site_id": entry["site_id"],
                "split": entry["split"],
                "forecast_population_sha256": entry["forecast_population_sha256"],
            }
            for entry in entries
        ],
    }
    descriptor = {
        "artifact_kind": "racepv.prepared-index",
        "schema_version": 3,
        "causality_profile": contract.PROFILE,
        "format": INDEX_FORMAT,
        "selected_split": split,
        "leaves": entries,
        "files": file_list,
        "file_scopes": scope_list,
        "read_scope_sha256": _read_scope_sha256(file_list, scope_list),
        "aggregate_content_sha256": window_io._aggregate_file_identities(file_list),
        "forecast_population_sha256": contract.canonical_json_sha256(population),
    }
    attempt = window_io._v2_attempt(path, directory=False)
    try:
        _publish_descriptor(path, descriptor)
        return _publication_result(path, descriptor)
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise
