"""Fail-closed registry for the supervised post-hoc baseline adapters.

Importing this module never imports third-party repositories.  A model build
first audits the fixed checkout, snapshots the audited commit, and only then
loads the single whitelisted upstream model module.
"""

from __future__ import annotations

import _imp
import copy
import hashlib
import importlib
import io
import json
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from racepv.baselines.modern import ModernCovariateTokenBuilder, ModernQuantilePostprocessor
from scripts.baseline_extension_sources import audit_checkout

if TYPE_CHECKING:
    from racepv.baselines.fact_qr import FACTQuantileAdapter
    from racepv.baselines.itransformer_qr import ITransformerQuantileAdapter
    from racepv.baselines.softs_qr import SOFTSQuantileAdapter


_SCHEMA_VERSION = "racepv.baseline-extension-supervised-model.v1"
_OUTPUT_HEAD = "direct_ordered_quantiles_v1"
_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
_FEATURE_SCHEMA = {
    "history_values": "registered_history_features_with_target_first",
    "history_mask": "observed_boolean_same_shape",
    "future_known": "registered_weather_without_future_target",
    "site_metadata": "registered_static_features",
    "future_target": "forbidden",
}
_FEATURE_SCHEMA_V2 = {
    **_FEATURE_SCHEMA,
    "history_values": "origin_resolved_history_zero_for_missing_or_imputed",
    "history_mask": "origin_validity_mask_with_observed_provenance_flags",
    "future_known": "deterministic_future_known_without_future_target",
}


def _validate_prepared_provenance(value):
    from racepv.data.prepared_contract import (
        SPLITS,
        V2_WINDOW_SEQUENCE_STORE_FORMAT,
        V2_WINDOW_SHARD_INDEX_FORMAT,
        _exact,
        _integer,
        _keys,
        _sha,
        canonical_json_sha256,
    )
    from racepv.io import REGISTERED_SPATIAL_ROLES

    def site(record):
        _keys(record, ("id", "name", "spatial_role"), "prepared identity site")
        _integer(record["id"], "prepared identity site.id")
        if type(record["name"]) is not str or not record["name"].strip():
            raise ValueError("prepared identity site.name must be nonempty")
        if (
            type(record["spatial_role"]) is not str
            or record["spatial_role"] not in REGISTERED_SPATIAL_ROLES
        ):
            raise ValueError("prepared identity site role differs")

    site(value["site"])
    if type(value["source_evidence_kind"]) is not str or value["source_evidence_kind"] not in (
        "provider_document",
        "synthetic_fixture",
    ):
        raise ValueError("prepared identity evidence kind differs")
    has_sites, has_populations = "sites" in value, "populations" in value
    selected_key = "selected_forecast_population_sha256"
    if has_sites != has_populations or (selected_key in value and not has_sites):
        raise ValueError("prepared identity requires paired collection details")
    if not has_sites:
        return
    sites, populations = value["sites"], value["populations"]
    if (
        type(sites) is not list
        or type(populations) is not list
        or not sites
        or len(sites) != len(populations)
    ):
        raise ValueError("prepared identity collection details must be paired nonempty lists")
    for record, population in zip(sites, populations, strict=True):
        site(record)
        _keys(
            population,
            ("site_id", "split", "forecast_population_sha256"),
            "prepared identity population",
        )
        _integer(population["site_id"], "prepared identity population.site_id")
        _exact(population["site_id"], record["id"], "prepared identity population site")
        if type(population["split"]) is not str or population["split"] not in SPLITS:
            raise ValueError("prepared identity population split differs")
        _sha(population["forecast_population_sha256"], "prepared identity population hash")
    _exact(value["site"], sites[0], "prepared identity representative site")
    ids = [record["id"] for record in sites]
    # Canonical sequence identity lists all four views of one site. A shard
    # lists one selected split per distinct site, in increasing site-id order.
    sequence = len(ids) > 1 and len(set(ids)) == 1
    if sequence:
        _exact(
            [population["split"] for population in populations],
            list(SPLITS),
            "prepared identity sequence splits",
        )
        for record in sites:
            _exact(record, value["site"], "prepared identity sequence site")
        if selected_key not in value:
            raise ValueError("prepared identity sequence selection is missing")
        selected = populations[SPLITS.index(value["split"])]
        _exact(
            value[selected_key],
            selected["forecast_population_sha256"],
            "prepared identity selected population",
        )
        fmt, selected_split = V2_WINDOW_SEQUENCE_STORE_FORMAT, None
    else:
        if selected_key in value or ids != sorted(set(ids)):
            raise ValueError("prepared identity shard sites/selection differ")
        if any(population["split"] != value["split"] for population in populations):
            raise ValueError("prepared identity shard population split differs")
        fmt, selected_split = V2_WINDOW_SHARD_INDEX_FORMAT, value["split"]
    population_hash = canonical_json_sha256(
        {
            "causality_profile": value["causality_profile"],
            "format": fmt,
            "selected_split": selected_split,
            "leaves": populations,
        }
    )
    _exact(
        value["forecast_population_sha256"],
        population_hash,
        "prepared identity collection population hash",
    )


def validate_prepared_identity(value):
    """Validate serialized v2 identity, without asserting scientific qualification."""
    required = {
        "schema_kind",
        "schema_version",
        "causality_profile",
        "schema_identity_sha256",
        "forecast_population_sha256",
        "descriptor_sha256",
        "aggregate_content_sha256",
        "split",
        "site",
        "content_verified",
        "array_semantics_verified",
        "registered_evidence_eligible",
        "source_evidence_kind",
        "source_interpretation_verified",
        "clock",
        "mask_semantics",
    }
    if (
        type(value) is not dict
        or not required <= set(value)
        or set(value) - required - {"sites", "populations", "selected_forecast_population_sha256"}
    ):
        raise ValueError("prepared identity schema differs")
    if (
        value["schema_kind"] != "racepv.prepared-window-contract"
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["causality_profile"] != "as-of-origin-v1"
    ):
        raise ValueError("prepared identity version/profile differs")
    for field in (
        "schema_identity_sha256",
        "forecast_population_sha256",
        "descriptor_sha256",
        "aggregate_content_sha256",
    ):
        if type(value[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            raise ValueError(f"prepared identity {field} differs")
    if "selected_forecast_population_sha256" in value and (
        type(value["selected_forecast_population_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["selected_forecast_population_sha256"]) is None
    ):
        raise ValueError("prepared selected population identity differs")
    for field in (
        "content_verified",
        "array_semantics_verified",
        "registered_evidence_eligible",
        "source_interpretation_verified",
    ):
        if type(value[field]) is not bool:
            raise ValueError(f"prepared identity {field} must be boolean")
    if value["content_verified"] is not True:
        raise ValueError("prepared identity requires complete content verification")
    if value["split"] not in ("train", "validation", "calibration", "test"):
        raise ValueError("prepared identity split differs")
    try:
        _validate_prepared_provenance(value)
    except TypeError as error:
        raise ValueError(f"prepared identity provenance type differs: {error}") from error
    from racepv.data.prepared_contract import deterministic_sampling_points

    clock = {
        "sampling_interval_ns": 900_000_000_000,
        "stored_timezone": "UTC",
        "timestamps": "target_interval_start_ns",
        "issue": "last_history_interval_end",
        "lead": "target_interval_end_minus_issue",
        "day_ids": "utc_days_since_epoch",
        "deterministic_sampling_points": deterministic_sampling_points(),
    }
    mask = {
        "missing_values": "zero_value_zero_mask",
        "imputed_values": "never_exposed",
        "provenance_flag_mask": 1,
    }
    if json.dumps(value["clock"], sort_keys=True) != json.dumps(clock, sort_keys=True):
        raise ValueError("prepared identity clock differs")
    if json.dumps(value["mask_semantics"], sort_keys=True) != json.dumps(mask, sort_keys=True):
        raise ValueError("prepared identity mask semantics differ")
    return copy.deepcopy(value)


def validate_prepared_protocol(value):
    """Versioned recovery identities; legacy five-pin records remain exact."""
    keys = {
        f"{role}_sha256"
        for role in (
            "train_descriptor",
            "train_content",
            "validation_descriptor",
            "validation_content",
            "common_config",
        )
    }
    if type(value) is not dict or set(value) not in (
        keys,
        keys | {"prepared_schema_version", "prepared_contracts"},
    ):
        raise ValueError("binding protocol schema differs")
    for field in keys:
        if type(value[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            raise ValueError("binding protocol SHA-256 differs")
    if "prepared_contracts" in value:
        if (
            type(value["prepared_schema_version"]) is not int
            or value["prepared_schema_version"] != 2
        ):
            raise ValueError("prepared binding version differs")
        contracts = value["prepared_contracts"]
        if type(contracts) is not dict or set(contracts) != {"train", "validation"}:
            raise ValueError("prepared binding splits differ")
        for split, identity in contracts.items():
            identity = validate_prepared_identity(identity)
            if (
                identity["split"] != split
                or identity["descriptor_sha256"] != value[f"{split}_descriptor_sha256"]
                or identity["aggregate_content_sha256"] != value[f"{split}_content_sha256"]
            ):
                raise ValueError("prepared binding source identity differs")
    return copy.deepcopy(value)


def prepared_protocol_from_identities(identities, common_config_sha256):
    protocol = {"common_config_sha256": common_config_sha256}
    for split in ("train", "validation"):
        protocol[f"{split}_descriptor_sha256"] = identities["descriptor_hashes"][split]
        protocol[f"{split}_content_sha256"] = identities["content_hashes"][split]
    if "prepared_contracts" in identities:
        protocol["prepared_schema_version"] = 2
        protocol["prepared_contracts"] = identities["prepared_contracts"]
    return validate_prepared_protocol(protocol)


_CONFIG_KEYS = {
    "schema_version",
    "model_id",
    "input_dim",
    "future_dim",
    "site_dim",
    "history_feature_names",
    "future_feature_names",
    "site_feature_names",
    "clear_sky_feature_index",
    "daylight_feature_index",
    "history_length",
    "horizon",
    "quantiles",
    "feature_schema",
    "source_identity",
    "backbone_defaults",
    "output_head",
}
_SOURCE_KEYS = {
    "repository_url",
    "git_commit",
    "source_tree_sha256",
    "primary_model_path",
    "primary_model_sha256",
}

SOURCE_IDENTITIES: dict[str, dict[str, str]] = {
    "fact_qr": {
        "repository_url": "https://github.com/wanghq21/FACT.git",
        "git_commit": "aa825721d1a0a6032b2f8bcccc6e0f7b14884ae4",
        "source_tree_sha256": ("76a108d667b2d799f2ca8d3ac56dadba3b705539e8d9b1155f06aea5bfc13426"),
        "primary_model_path": "models/FACT.py",
        "primary_model_sha256": (
            "43e433220a204461b6bb579ef6a4d802be3ac0b8c4eea15c0880cdc393dba831"
        ),
    },
    "itransformer_qr": {
        "repository_url": "https://github.com/thuml/iTransformer.git",
        "git_commit": "c2426e68ca13f74aaec08045c5c724d8ad328124",
        "source_tree_sha256": ("d858afc6b28aa75dc04499ac0ef750258cc92eb6286d09d904185418810896df"),
        "primary_model_path": "model/iTransformer.py",
        "primary_model_sha256": (
            "6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db"
        ),
    },
    "softs_qr": {
        "repository_url": "https://github.com/Secilia-Cxy/SOFTS.git",
        "git_commit": "f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44",
        "source_tree_sha256": ("cb829511d34a1451dd72b43ad8691e28488a607120294fb17c55e0703204084c"),
        "primary_model_path": "models/SOFTS.py",
        "primary_model_sha256": (
            "71cd0b1d72055ff71a313139bf1e9801351f7798c7b392f679a3befd7c2f16e1"
        ),
    },
}

BACKBONE_DEFAULTS: dict[str, dict[str, object]] = {
    "fact_qr": {
        "d_model": 512,
        "d_ff": 1024,
        "dropout": 0.1,
        "use_norm": True,
        "freq": "n",
        "num_kernels": 4,
        "dilation": [1, 2, 3, 2, 1],
        "core": 0.5,
    },
    "itransformer_qr": {
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 2,
        "d_ff": 2048,
        "factor": 1,
        "dropout": 0.1,
        "embed": "timeF",
        "freq": "h",
        "activation": "gelu",
        "output_attention": False,
        "use_norm": True,
        "class_strategy": "projection",
    },
    "softs_qr": {
        "d_model": 512,
        "d_core": 512,
        "e_layers": 2,
        "d_ff": 2048,
        "dropout": 0.0,
        "activation": "gelu",
        "use_norm": True,
    },
}

_CHECKOUT_NAMES = {
    "fact_qr": "fact_qr",
    "itransformer_qr": "itransformer_qr",
    "softs_qr": "softs_qr",
}
_MODULE_NAMES = {
    "fact_qr": "models.FACT",
    "itransformer_qr": "model.iTransformer",
    "softs_qr": "models.SOFTS",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_feature_names(name: str, value: object, expected_length: int) -> list[str]:
    if (
        type(value) is not list
        or len(value) != expected_length
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be an ordered unique list matching its dimension")
    return value


def _validate_config(model_id: str, value: Mapping[str, object]) -> dict[str, object]:
    if model_id not in SOURCE_IDENTITIES:
        raise ValueError(f"unknown supervised extension model: {model_id}")
    v2 = value.get("schema_version") == "racepv.baseline-extension-supervised-model.v2"
    if type(value) is not dict or set(value) != (
        _CONFIG_KEYS | {"prepared_contract"} if v2 else _CONFIG_KEYS
    ):
        raise ValueError("resolved config schema differs")
    if v2:
        validate_prepared_identity(value["prepared_contract"])
    if (
        value["schema_version"]
        != ("racepv.baseline-extension-supervised-model.v2" if v2 else _SCHEMA_VERSION)
        or value["model_id"] != model_id
    ):
        raise ValueError("resolved config identity differs")
    for field in ("input_dim", "future_dim", "history_length", "horizon"):
        _validate_positive_integer(field, value[field])
    if type(value["site_dim"]) is not int or value["site_dim"] < 0:
        raise ValueError("site_dim must be a non-negative integer")
    history_names = _validate_feature_names(
        "history feature names", value["history_feature_names"], value["input_dim"]
    )
    future_names = _validate_feature_names(
        "future feature names", value["future_feature_names"], value["future_dim"]
    )
    _validate_feature_names("site feature names", value["site_feature_names"], value["site_dim"])
    if history_names[0] != "power_norm":
        raise ValueError("history feature names must place power_norm first")
    forbidden_future_roles = {"power_norm", "target", "targets", "future_target"}
    if any(name.casefold() in forbidden_future_roles for name in future_names):
        raise ValueError("future feature names contain a target role")
    clear_sky_index = value["clear_sky_feature_index"]
    if (
        type(clear_sky_index) is not int
        or not 0 <= clear_sky_index < len(future_names)
        or future_names[clear_sky_index] != "clear_sky_power_norm"
    ):
        raise ValueError("clear-sky feature identity differs")
    daylight_index = value["daylight_feature_index"]
    if (
        type(daylight_index) is not int
        or not 0 <= daylight_index < len(future_names)
        or future_names[daylight_index] != "solar_elevation"
    ):
        raise ValueError("daylight feature identity differs")
    if value["quantiles"] != _QUANTILES:
        raise ValueError("quantiles differ from the registered seven-level contract")
    if type(value["feature_schema"]) is not dict or value["feature_schema"] != (
        _FEATURE_SCHEMA_V2 if v2 else _FEATURE_SCHEMA
    ):
        raise ValueError("feature schema differs from the leakage-safe registered contract")
    if type(value["source_identity"]) is not dict or set(value["source_identity"]) != _SOURCE_KEYS:
        raise ValueError("source identity differs")
    if value["source_identity"] != SOURCE_IDENTITIES[model_id]:
        raise ValueError("source identity differs")
    if (
        type(value["backbone_defaults"]) is not dict
        or value["backbone_defaults"] != BACKBONE_DEFAULTS[model_id]
    ):
        raise ValueError("backbone defaults differ from the fixed official entry point")
    if value["output_head"] != _OUTPUT_HEAD:
        raise ValueError("output head differs from the direct seven-quantile contract")
    return copy.deepcopy(value)


def _audit_fixed_checkout(model_id: str) -> Path:
    root = _project_root() / "third-party" / "baseline-extension" / _CHECKOUT_NAMES[model_id]
    audit = audit_checkout(model_id, root)
    expected = SOURCE_IDENTITIES[model_id]
    for field in ("repository_url", "git_commit", "source_tree_sha256"):
        if audit[field] != expected[field]:
            raise ValueError(f"audited source {field} differs: {model_id}")
    primary = root.joinpath(*PurePosixPath(expected["primary_model_path"]).parts)
    if primary.is_symlink() or not primary.is_file():
        raise ValueError(f"primary model source is not a regular file: {model_id}")
    digest = hashlib.sha256(primary.read_bytes()).hexdigest()
    if digest != expected["primary_model_sha256"]:
        raise ValueError(f"primary model source hash differs: {model_id}")
    return root


def _git_archive(root: Path, commit: str) -> bytes:
    completed = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    aliases: dict[str, bool] = {}
    for member in members:
        relative = PurePosixPath(member.name)
        normalized_parts = tuple(unicodedata.normalize("NFC", part) for part in relative.parts)
        if (
            relative.is_absolute()
            or not normalized_parts
            or any(part in {"", ".", ".."} for part in normalized_parts)
            or "\\" in member.name
        ):
            raise ValueError("unsafe path in fixed source archive")
        if not (member.isdir() or member.isfile()):
            raise ValueError("non-regular member in fixed source archive")
        alias = "/".join(part.casefold() for part in normalized_parts)
        if alias in aliases:
            raise ValueError("casefold or Unicode path collision in fixed source archive")
        aliases[alias] = member.isfile()

    for alias, is_file in aliases.items():
        parts = alias.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            if aliases.get(parent) is True:
                raise ValueError("file/directory prefix collision in fixed source archive")
        if is_file and any(other.startswith(f"{alias}/") for other in aliases):
            raise ValueError("file/directory prefix collision in fixed source archive")
    return members


def _extract_regular_archive(archive_bytes: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in _safe_archive_members(archive):
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("unreadable member in fixed source archive")
            with target.open("xb") as output:
                output.write(source.read())


def _snapshot_tree_sha256(snapshot: Path) -> str:
    inventory: list[dict[str, str]] = []
    for path in sorted(snapshot.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("snapshot contains linked or non-regular content")
        before = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        content = path.read_bytes()
        after_info = path.lstat()
        after = (
            after_info.st_dev,
            after_info.st_ino,
            after_info.st_size,
            after_info.st_mtime_ns,
            after_info.st_ctime_ns,
        )
        if before != after:
            raise ValueError("snapshot changed while hashing")
        inventory.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@contextmanager
def _fixed_source_snapshot(model_id: str, root: Path) -> Iterator[Path]:
    identity = SOURCE_IDENTITIES[model_id]
    archive = _git_archive(root, identity["git_commit"])
    with tempfile.TemporaryDirectory(prefix=f"racepv-{model_id}-") as directory:
        snapshot = Path(directory)
        _extract_regular_archive(archive, snapshot)
        if _snapshot_tree_sha256(snapshot) != identity["source_tree_sha256"]:
            raise ValueError(f"snapshot source tree hash differs: {model_id}")
        primary = snapshot.joinpath(*PurePosixPath(identity["primary_model_path"]).parts)
        digest = hashlib.sha256(primary.read_bytes()).hexdigest()
        if digest != identity["primary_model_sha256"]:
            raise ValueError(f"snapshot model source hash differs: {model_id}")
        yield snapshot


def _module_family(name: str) -> tuple[str, ...]:
    top = name.split(".", maxsplit=1)[0]
    return (top, "layers")


def _belongs_to_family(name: str, families: tuple[str, ...]) -> bool:
    return any(name == family or name.startswith(f"{family}.") for family in families)


def _missing_reformer_module() -> types.ModuleType:
    module = types.ModuleType("reformer_pytorch")

    class LSHSelfAttention(nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            raise RuntimeError("Reformer is outside the fixed iTransformer adapter")

    module.LSHSelfAttention = LSHSelfAttention
    return module


def _load_upstream_model_class(model_id: str, snapshot: Path) -> type[nn.Module]:
    module_name = _MODULE_NAMES[model_id]
    families = _module_family(module_name)
    _imp.acquire_lock()
    original_path = list(sys.path)
    saved = {
        name: module
        for name, module in tuple(sys.modules.items())
        if _belongs_to_family(name, families)
    }
    reformer_was_present = "reformer_pytorch" in sys.modules
    reformer_module = sys.modules.get("reformer_pytorch")
    try:
        for name in tuple(sys.modules):
            if _belongs_to_family(name, families):
                del sys.modules[name]
        if model_id == "itransformer_qr" and importlib.util.find_spec("reformer_pytorch") is None:
            sys.modules["reformer_pytorch"] = _missing_reformer_module()
        sys.path.insert(0, str(snapshot))
        module = importlib.import_module(module_name)
        model_class: Any = getattr(module, "Model", None)
        if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
            raise TypeError(f"official module does not expose an nn.Module Model: {model_id}")
        return model_class
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if _belongs_to_family(name, families):
                del sys.modules[name]
        sys.modules.update(saved)
        if model_id == "itransformer_qr" and not reformer_was_present:
            sys.modules.pop("reformer_pytorch", None)
        elif reformer_module is not None:
            sys.modules["reformer_pytorch"] = reformer_module
        _imp.release_lock()


def _load_fixed_model_class(model_id: str) -> type[nn.Module]:
    root = _audit_fixed_checkout(model_id)
    with _fixed_source_snapshot(model_id, root) as snapshot:
        model_class = _load_upstream_model_class(model_id, snapshot)
    if _audit_fixed_checkout(model_id) != root:
        raise ValueError(f"source checkout root changed during model load: {model_id}")
    return model_class


class SupervisedQRAdapterBase(nn.Module):
    """Shared leakage-safe input and ordered-output boundary for adapters."""

    quantile_head: nn.Module
    backbone: nn.Module

    def __init__(self, resolved_config: dict[str, object]) -> None:
        super().__init__()
        self._resolved_config = copy.deepcopy(resolved_config)
        self.history_length = int(resolved_config["history_length"])
        self.horizon = int(resolved_config["horizon"])
        self.quantiles = tuple(float(item) for item in resolved_config["quantiles"])
        self.use_norm = bool(resolved_config["backbone_defaults"]["use_norm"])
        self.token_builder = ModernCovariateTokenBuilder(
            int(resolved_config["input_dim"]),
            int(resolved_config["future_dim"]),
            int(resolved_config["site_dim"]),
            horizon=self.horizon,
        )
        self.postprocess = ModernQuantilePostprocessor(
            self.quantiles,
            horizon=self.horizon,
            clear_sky_feature_index=int(resolved_config["clear_sky_feature_index"]),
            daylight_feature_index=int(resolved_config["daylight_feature_index"]),
        )

    def resolved_identity(self) -> dict[str, object]:
        identity = copy.deepcopy(self._resolved_config)
        identity.update(
            {
                "official_code_used": True,
                "implementation_kind": "paper_derived_protocol_adaptation",
                "point_projection_retained": False,
                "input_mapping": "leakage_safe_history_plus_known_future_tokens_v1",
                "parity_contract": {
                    "boundary": "official_final_projection_input",
                    "rtol": 1e-6,
                    "atol": 1e-6,
                },
            }
        )
        return identity

    def backbone_features(self, tokens: Tensor) -> Tensor:
        raise NotImplementedError

    def raw_quantiles(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def normalization_stats(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        means = tokens.mean(1, keepdim=True).detach()
        centered = tokens - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return means, stdev

    def normalize_tokens(self, tokens: Tensor) -> Tensor:
        if not self.use_norm:
            return tokens
        means, stdev = self.normalization_stats(tokens)
        return (tokens - means) / stdev

    def postprocess_normalized(
        self,
        raw: Tensor,
        tokens: Tensor,
        future_known: Tensor,
    ) -> Tensor:
        expected = (future_known.shape[0], self.horizon, len(self.quantiles))
        if raw.ndim != 3 or raw.shape != expected:
            raise ValueError(
                "raw quantiles must have shape "
                f"[batch, horizon={self.horizon}, quantiles={len(self.quantiles)}]"
            )
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("raw quantiles contain non-finite values")
        ordered = self.postprocess.ordered(raw)
        if self.use_norm:
            means, stdev = self.normalization_stats(tokens)
            target_mean = means[:, :, 0, None]
            target_stdev = stdev[:, :, 0, None]
            ordered = ordered * target_stdev + target_mean

        daylight = torch.ones(
            future_known.shape[:2],
            dtype=torch.bool,
            device=future_known.device,
        )
        for name, index in (
            ("clear_sky_feature_index", self.postprocess.clear_sky_feature_index),
            ("daylight_feature_index", self.postprocess.daylight_feature_index),
        ):
            if index is None or not 0 <= index < future_known.shape[-1]:
                raise ValueError(f"{name} is outside the future_known feature dimension")
            signal = future_known[..., index]
            daylight &= torch.isfinite(signal) & (signal > 0)
        bounded = ordered.clamp(0.0, self.postprocess.maximum_power)
        return torch.where(daylight[..., None], bounded, torch.zeros_like(bounded))

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> Tensor:
        if history_values.ndim != 3 or history_values.shape[1] != self.history_length:
            raise ValueError(f"history_values must have history_length={self.history_length}")
        tokens = self.token_builder(
            history_values,
            history_mask,
            future_known,
            site_metadata,
        )
        raw = self.raw_quantiles(self.backbone_features(tokens))
        return self.postprocess_normalized(raw, tokens, future_known)


def build_extension_model(
    model_id: str,
    resolved_config: Mapping[str, object],
) -> nn.Module:
    """Build one source-bound supervised QR adapter without implicit defaults."""

    config = _validate_config(model_id, resolved_config)
    model_class = _load_fixed_model_class(model_id)
    if model_id == "fact_qr":
        from racepv.baselines.fact_qr import FACTQuantileAdapter

        adapter: type[FACTQuantileAdapter] = FACTQuantileAdapter
    elif model_id == "itransformer_qr":
        from racepv.baselines.itransformer_qr import ITransformerQuantileAdapter

        adapter: type[ITransformerQuantileAdapter] = ITransformerQuantileAdapter
    else:
        from racepv.baselines.softs_qr import SOFTSQuantileAdapter

        adapter: type[SOFTSQuantileAdapter] = SOFTSQuantileAdapter
    return adapter(config, model_class)
