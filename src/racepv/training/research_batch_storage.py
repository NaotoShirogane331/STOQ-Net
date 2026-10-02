"""Stream-hashed own artifacts and immutable non-epoch unit publication."""

import hashlib
import os
import shutil
import stat
import uuid
from contextlib import contextmanager

from racepv.training.exploratory_contract import digest, equal, safe_read, strict_json
from racepv.training.recoverable_run import (
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _write_exclusive,
)
from racepv.training.recovery import (
    RecoveryIntegrityError,
    _open_absolute_directory,
    validate_relative_path,
)


def require_space(path, requested, reserve=0):
    free = shutil.disk_usage(path).free
    need = int(requested) + int(reserve)
    if free < need:
        raise RuntimeError(f"RESOURCE_FAILED: required={need} bytes (payload={requested}, reserve={reserve}); available={free} at {path}")
    return dict(required_bytes=need, available_bytes=free)


@contextmanager
def verified_file(path, *, limit, expected=None):
    """Hash and consume the SAME no-link file handle, including artifacts above 2 GiB."""
    directory = _open_absolute_directory(path.absolute().parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise RecoveryIntegrityError("unsafe or oversized batch artifact")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            hasher = hashlib.sha256()
            consumed = 0
            while block := handle.read(8 * 1024**2):
                consumed += len(block)
                if consumed > limit:
                    raise RecoveryIntegrityError("batch artifact grew beyond the file limit")
                hasher.update(block)
            if consumed != before.st_size:
                raise RecoveryIntegrityError("batch artifact size changed while hashing")
            record = dict(path=path.name, sha256=hasher.hexdigest(), size_bytes=before.st_size)
            if expected is not None:
                equal(record, expected, "artifact SHA/size")
            handle.seek(0)
            yield handle, record
        after = os.fstat(fd)
        entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        def identity(info):
            return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink
        if identity(before) != identity(after) or identity(after) != identity(entry):
            raise RecoveryIntegrityError("batch artifact changed while being verified")
    except OSError as error:
        raise RecoveryIntegrityError(f"unsafe/missing artifact: {path}") from error
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def file_record(path, *, limit):
    with verified_file(path, limit=limit) as (_, record):
        return record


def stage_directory(binding, label):
    stage = binding.run_root / "staging" / (label + "-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    return stage


def write_array(path, value):
    import numpy as np
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def read_array(path, record, limit):
    import numpy as np
    with verified_file(path, limit=limit, expected=record) as (handle, _):
        return np.load(handle, allow_pickle=False)


def commit_unit(binding, stage, unit, *, metadata, previous):
    validate_relative_path(unit)
    if "/" in unit:
        raise ValueError("unit must be one component")
    limits = binding.raw["shared"]["resources"]
    records = [file_record(path, limit=limits["model_file_limit_bytes"]) for path in sorted(stage.iterdir())]
    if not records:
        raise RecoveryIntegrityError("empty unit")
    require_space(stage, 64 * 1024, limits["scratch_reserve_bytes"])
    receipt = dict(schema_version=1, binding_sha256=binding.sha256, unit=unit,
        previous_commit_sha256=previous, files=records, metadata=metadata)
    raw = _canonical_json(receipt)
    _write_exclusive(stage / "commit.json", raw)
    _fsync_directory(stage)
    _rename_noreplace(stage, binding.run_root / "units" / unit)
    _fsync_directory(binding.run_root / "units")
    return dict(receipt, commit_sha256=digest(raw))


def read_units(binding, expected_units):
    root = binding.run_root / "units"
    names = sorted(path.name for path in root.iterdir())
    if names != expected_units[:len(names)]:
        raise RecoveryIntegrityError("unit sequence is not an expected contiguous prefix")
    previous, receipts = None, []
    limit = binding.raw["shared"]["resources"]["model_file_limit_bytes"]
    for name in names:
        raw = safe_read(root / name / "commit.json")
        receipt = strict_json(raw, "batch unit")
        if set(receipt) != {"schema_version", "binding_sha256", "unit", "previous_commit_sha256", "files", "metadata"} or receipt["schema_version"] != 1:
            raise RecoveryIntegrityError("unit commit schema differs")
        equal([receipt["binding_sha256"], receipt["unit"], receipt["previous_commit_sha256"]],
              [binding.sha256, name, previous], "unit predecessor/binding")
        files = receipt["files"]
        expected_files = ["model.joblib", "prediction.npy"] if name.startswith("l") else ["prediction.npy", "reference.json"]
        if [item["path"] for item in files] != expected_files:
            raise RecoveryIntegrityError("unit file set differs")
        for item in files:
            validate_relative_path(item["path"])
            if "/" in item["path"]:
                raise RecoveryIntegrityError("unit artifact path differs")
            with verified_file(root / name / item["path"], limit=limit, expected=item):
                pass
        previous = digest(raw)
        receipts.append(dict(receipt, commit_sha256=previous))
    return receipts
