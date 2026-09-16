"""Bounded resolver-to-NPY bridge; never grants scientific qualification."""

from __future__ import annotations

import copy
import hashlib

import numpy as np

from racepv import io as window_io
from racepv.data import prepared_contract as contract
from racepv.data import scoped_store as store


def _write_resolver(path, array):
    """Exclusive physical NPY, at most 64 windows per read, including empty tensors."""
    with window_io._v2_open_new(path) as handle:
        np.lib.format.write_array_header_2_0(
            handle,
            {
                "shape": array.shape,
                "fortran_order": False,
                "descr": np.lib.format.dtype_to_descr(array.dtype),
            },
        )
        for start in range(0, len(array), 64):
            stop = min(start + 64, len(array))
            batch = array.read_batch(np.arange(start, stop, dtype=np.int64))
            if (
                not isinstance(batch, np.ndarray)
                or batch.shape != (stop - start, *array.shape[1:])
                or batch.dtype != array.dtype
            ):
                raise ValueError("resolver batch shape or dtype changed")
            handle.write(batch.tobytes(order="C"))


def publish_resolved_scoped_leaf(
    output_directory,
    *,
    staging_directory,
    schema,
    expected_schema_sha256,
    causal_arrays,
    evidence,
    allowed_splits,
):
    """Stage bounded tensors, use the unrelaxed physical publisher, then reload.

    Both directories must be new, disjoint, unlinked paths. Staging and failures
    are retained. The acceptance record lives outside the leaf's exact file
    graph. This verifies storage, NOT provider extraction or preparation/fit
    recomputation: those require separately pinned preparation acceptance.
    """
    contract._sha(expected_schema_sha256, "expected_schema_sha256")
    schema = copy.deepcopy(schema)
    if contract.canonical_json_sha256(schema) != expected_schema_sha256:
        raise ValueError("external schema pin mismatch")
    allowed = store._allowed_splits(allowed_splits)
    schema = store._validate_publication_metadata(schema, evidence)
    contract._keys(causal_arrays, contract.ARRAY_ROLES, "causal_arrays")
    references = {name: {"path": f"arrays/{name}.npy"} for name in contract.ARRAY_ROLES}
    roles = store._derived_leaf_roles(schema, references)
    scopes = [
        {"path": path, "role": role, "splits": splits} for path, (role, splits) in roles.items()
    ]
    store._authorize(scopes, allowed)
    output = store._reject_link_components(output_directory)
    stage = store._reject_link_components(staging_directory)
    if output == stage or output in stage.parents or stage in output.parents:
        raise ValueError("staging and output must be disjoint directories")
    window_io._v2_check_destination(output)
    window_io._v2_check_destination(stage)
    # Types and geometry before resolver reads; logical and semantic identities
    # are checked against the frozen schema after the staged bytes are loaded.
    for name, array in causal_arrays.items():
        resolver = type(array) is contract.ResolverModelArray
        if not isinstance(array, np.ndarray) and not resolver:
            raise TypeError("unsupported array storage")
        if array.dtype.hasobject or (resolver and name not in contract.MODEL_ARRAYS):
            raise TypeError("resolver allowed only for non-object model tensors")
        specification = schema["arrays"][name]
        if (
            list(array.shape) != specification["shape"]
            or str(array.dtype) != specification["dtype"]
        ):
            raise ValueError("array geometry differs from schema")
    attempt = window_io._v2_attempt(stage, directory=True)
    owned_maps = []
    try:
        physical = dict(causal_arrays)
        staged_files = []
        for name in contract.MODEL_ARRAYS:
            array = physical[name]
            if type(array) is not contract.ResolverModelArray:
                continue
            path = stage / f"{name}.npy"
            _write_resolver(path, array)
            identity = store._identity_for_path(path, stage)
            staged_files.append(identity)
            physical[name] = store._load_npy(identity, path, schema["arrays"][name])
            if isinstance(physical[name], np.memmap):
                owned_maps.append(physical[name])
        result = store.publish_scoped_leaf(
            output, schema=schema, causal_arrays=physical, evidence=evidence
        )
        loaded = store.load_scoped_store(
            result["descriptor_path"],
            expected_descriptor_sha256=result["descriptor_sha256"],
            expected_read_scope_sha256=result["read_scope_sha256"],
            allowed_splits=allowed,
        )
        owned_maps.extend(
            array
            for arrays in loaded["causal_arrays"]
            for array in arrays.values()
            if isinstance(array, np.memmap)
        )
        receipt = {
            "kind": "racepv.scoped-storage-acceptance",
            "version": 1,
            "schema_sha256": expected_schema_sha256,
            **result,
            "staged_files": staged_files,
            "allowed_splits": list(allowed),
            "content_reloaded": True,
            "preparation_recomputed": False,
            "source_interpretation_verified": False,
            "extraction_completeness_verified": False,
            "formal_ready": False,
        }
        encoded = window_io._v2_json_bytes(receipt)
        acceptance = stage / "acceptance.json"
        window_io._v2_write_bytes(acceptance, encoded)
        return {
            **result,
            "acceptance_path": str(acceptance),
            "acceptance_sha256": hashlib.sha256(encoded).hexdigest(),
            "formal_ready": False,
        }
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise
    finally:
        for array in owned_maps:
            array._mmap.close()
