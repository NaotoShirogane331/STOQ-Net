import copy
import hashlib
from pathlib import Path

from .prepared_contract import (
    _aggregate_file_identities,
    _graph_nodes,
    _metadata_graph,
    _sha,
    _strict_json_value,
    _verify_bytes,
    contained_artifact_file,
)


def inspect_v2_metadata_json_only(descriptor_path, *, expected_descriptor_sha256):
    """Inspect a metadata snapshot; never qualify or hash raw/array content."""
    _sha(expected_descriptor_sha256, "expected_descriptor_sha256")
    path = Path(descriptor_path)
    path = contained_artifact_file(path.parent, path.name)
    root = path.parent
    snapshots = {}

    def read_json(candidate):
        candidate = Path(candidate)
        if candidate.suffix != ".json":
            raise ValueError("JSON metadata publication required before opening a file")
        relative = candidate.relative_to(root).as_posix()
        candidate = contained_artifact_file(root, relative)
        if candidate not in snapshots:
            data = candidate.read_bytes()
            if candidate == path and hashlib.sha256(data).hexdigest() != expected_descriptor_sha256:
                raise ValueError("root descriptor identity mismatch")
            snapshots[candidate] = (data, _strict_json_value(data.decode("utf-8")))
        return copy.deepcopy(snapshots[candidate][1])

    metadata = _metadata_graph(path, ancestors=(), read_json=read_json)
    for node in _graph_nodes(metadata):
        if (
            _aggregate_file_identities(node["descriptor"]["files"])
            != node["descriptor"]["aggregate_content_sha256"]
        ):
            raise ValueError("aggregate content identity mismatch")
        for item in node["descriptor"]["files"]:
            candidate = node["_resolved"][item["path"]]
            if candidate in snapshots:
                _verify_bytes(item, snapshots[candidate][0])
        _verify_bytes(node["receipt"]["descriptor"], snapshots[Path(node["descriptor_path"])][0])
    identities = [
        {
            "path": candidate.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for candidate, (data, _) in snapshots.items()
    ]
    for node in _graph_nodes(metadata):
        del node["_resolved"]
    return {
        "artifact_kind": "racepv.prepared-json-preflight",
        "schema_version": 1,
        "descriptor_sha256": expected_descriptor_sha256,
        "metadata": metadata,
        "json_file_identities": sorted(identities, key=lambda item: item["path"]),
        "metadata_json_verified": True,
        "content_verified": False,
        "array_semantics_verified": False,
        "source_interpretation_verified": False,
        "registered_evidence_eligible": False,
    }
