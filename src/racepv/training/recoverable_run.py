"""Read-only state inference for recoverable single-seed runs."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
import re
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import torch

from .engine import EarlyStoppingTrainer, TrainerProgress
from .recovery import (
    CommittedGeneration,
    RecoveryBusyError,
    RecoveryIntegrityError,
    RecoveryUnsupportedError,
    RunBinding,
    _manifest_pins,
    decode_progress,
    encode_progress,
    load_binding,
    load_tensor_mapping,
    parse_json_bytes,
    read_pinned_binding_inputs,
    read_regular_file_nofollow,
    restore_rng_state,
    validate_commit,
    validate_relative_path,
)

_GENERATION = re.compile(r"g([0-9]{4})")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_REFERENCE_FILES = (
    "best.ckpt",
    "model.pt",
    "resolved_run_config.json",
    "run.json",
    "validation_predictions.json",
    "validation_predictions.npz",
)
_STAGE_ROLES = {
    "model": (
        "best.ckpt",
        "model.pt",
        "resolved_config.yaml",
        "resolved_run_config.json",
    ),
    "validation": ("validation_predictions.json", "validation_predictions.npz"),
    "summary": ("run.json",),
}
_STAGE_PREDECESSORS = {
    "model": (),
    "validation": ("model",),
    "summary": ("model", "validation"),
}
_STAGE_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "binding_sha256",
        "terminal_commit_sha256",
        "stage",
        "dependencies",
        "files",
    }
)
_EXPORT_COMMIT_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "binding_sha256",
        "terminal_commit_sha256",
        "stages",
        "files",
    }
)
_ACCEPTANCE_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "binding_sha256",
        "terminal_commit_sha256",
        "export_commit_sha256",
        "session_receipt",
        "status",
    }
)
_SESSION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "session_id",
        "action",
        "binding_sha256",
        "started_at",
        "finished_at",
        "worker_returncode",
        "prior_session_outcome",
        "log_files",
    }
)


class RunStatus(str, Enum):
    INITIAL_COMMITTED = "initial_committed"
    TRAINING_COMMITTED = "training_committed"
    FIT_COMPLETE = "fit_complete"
    EXPORTS_READY = "exports_ready"
    COMPLETED = "completed"


@dataclass(frozen=True)
class RunInspection:
    status: RunStatus
    binding: RunBinding
    generations: tuple[CommittedGeneration, ...]
    terminal: CommittedGeneration | None
    export_commit_sha256: str | None
    acceptance_sha256: str | None


@dataclass(frozen=True)
class RunTransaction:
    run_root: Path
    binding_sha256: str


@dataclass(frozen=True)
class SeedRunRequest:
    mode: str
    run_root: Path
    binding: RunBinding


@dataclass(frozen=True)
class SessionResult:
    action: str
    status: RunStatus
    worker_returncode: int


def _loader_contract(train_loader: object, validation_loader: object) -> dict[str, object]:
    def dataset_length(loader: object) -> int:
        dataset = getattr(loader, "dataset", loader)
        return len(dataset)  # type: ignore[arg-type]

    return {
        "num_workers": int(getattr(train_loader, "num_workers", 0)),
        "shuffle": True,
        "drop_last": bool(getattr(train_loader, "drop_last", False)),
        "batch_size": getattr(train_loader, "batch_size", None),
        "train_length": dataset_length(train_loader),
        "validation_length": dataset_length(validation_loader),
    }


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name != "posix":
        raise RecoveryUnsupportedError("no-replace publication requires Linux")
    library = ctypes.CDLL(None, use_errno=True)
    operation = getattr(library, "renameat2", None)
    if operation is None:
        raise RecoveryUnsupportedError("renameat2(RENAME_NOREPLACE) is unavailable")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RecoveryIntegrityError(f"destination already exists: {destination}")
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        raise RecoveryUnsupportedError("filesystem lacks no-replace directory publication")
    raise OSError(error, os.strerror(error), str(destination))


def _state_bytes(payload: Mapping[str, object]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def _write_generation_directory(
    directory: Path,
    state: Mapping[str, object],
    *,
    binding_sha256: str,
    generation: int,
    previous_commit_sha256: str | None,
    fail_at: str | None = None,
) -> None:
    directory.mkdir(mode=0o700)
    state_bytes = _state_bytes(state)
    _write_exclusive(directory / "state.pt", state_bytes)
    if fail_at == "after_state_fsync":
        raise RuntimeError("injected failure after state fsync")
    commit = {
        "schema_version": 1,
        "purpose": "racepv_epoch_commit",
        "generation": generation,
        "binding_sha256": binding_sha256,
        "previous_commit_sha256": previous_commit_sha256,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    }
    _write_exclusive(directory / "commit.json", _canonical_json(commit))
    if fail_at == "after_commit_fsync":
        raise RuntimeError("injected failure after commit fsync")
    _fsync_directory(directory)


def create_run(
    parent: Path,
    run_name: str,
    *,
    binding_bytes: bytes,
    binding_sha256: str,
    initial_state: Mapping[str, object],
    input_files: Mapping[str, bytes],
) -> RunTransaction:
    if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise RecoveryIntegrityError("run name must be one safe path component")
    destination = parent / run_name
    if destination.exists() or destination.is_symlink():
        raise RecoveryIntegrityError(f"destination already exists: {destination}")
    if hashlib.sha256(binding_bytes).hexdigest() != binding_sha256:
        raise RecoveryIntegrityError("binding bytes do not match external SHA-256")
    staging = parent / f".{run_name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        for name in (
            "inputs",
            "epochs",
            "staging",
            "export-stages",
            "exports",
            "receipts",
            "sessions",
        ):
            (staging / name).mkdir(mode=0o700)
        _write_exclusive(staging / "binding.json", binding_bytes)
        binding = load_binding(staging, Path("binding.json"), binding_sha256)
        expected_input_paths = {
            str(binding.raw[role]["path"])
            for role in (
                "reference_policy",
                "source_inventory",
                "resolved_training_contract",
                "feature_contract",
            )
        }
        reference_path = str(binding.raw["reference_policy"]["path"])
        if reference_path not in input_files:
            raise RecoveryIntegrityError("reference policy snapshot is missing")
        expected_input_paths.update(
            str(pin["path"]) for pin in _manifest_pins(input_files[reference_path]).values()
        )
        if set(input_files) != expected_input_paths:
            raise RecoveryIntegrityError("pinned input snapshot paths differ from binding")
        for relative, data in sorted(input_files.items()):
            parts = validate_relative_path(relative)
            if parts[0] != "inputs":
                raise RecoveryIntegrityError("pinned inputs must be contained by inputs/")
            destination_file = staging.joinpath(*parts)
            destination_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_exclusive(destination_file, data)
        read_pinned_binding_inputs(load_binding(staging, Path("binding.json"), binding_sha256))
        if initial_state.get("generation") != 0 or initial_state.get("next_epoch") != 0:
            raise RecoveryIntegrityError("initial state must be generation zero")
        if initial_state.get("binding_sha256") != binding.raw_sha256:
            raise RecoveryIntegrityError("initial state binding mismatch")
        _write_generation_directory(
            staging / "epochs" / "g0000",
            initial_state,
            binding_sha256=binding_sha256,
            generation=0,
            previous_commit_sha256=None,
        )
        _fsync_directory(staging / "epochs")
        _fsync_directory(staging)
        _rename_noreplace(staging, destination)
        _fsync_directory(parent)
    except Exception:
        # Preserve staging for diagnosis; it is intentionally never accepted as a run.
        raise
    return RunTransaction(destination, binding_sha256)


def publish_generation(
    run_root: Path,
    state: Mapping[str, object],
    *,
    binding_sha256: str,
    expected_previous_commit_sha256: str | None,
    fail_at: str | None = None,
) -> CommittedGeneration:
    inspection = inspect_run(run_root, binding_sha256=binding_sha256)
    previous = inspection.generations[-1]
    if previous.commit_sha256 != expected_previous_commit_sha256:
        raise RecoveryIntegrityError("previous generation SHA-256 changed")
    generation = previous.generation + 1
    if state.get("generation") != generation or state.get("next_epoch") != generation:
        raise RecoveryIntegrityError("new state generation is not the next generation")
    staging = run_root / "staging" / f"epoch-{generation:04d}-{uuid.uuid4().hex}"
    _write_generation_directory(
        staging,
        state,
        binding_sha256=binding_sha256,
        generation=generation,
        previous_commit_sha256=previous.commit_sha256,
        fail_at=fail_at,
    )
    if fail_at == "before_publish":
        raise RuntimeError("injected failure before publish")
    destination = run_root / "epochs" / f"g{generation:04d}"
    _rename_noreplace(staging, destination)
    _fsync_directory(run_root / "epochs")
    if fail_at == "after_publish":
        raise RuntimeError("injected failure after publish")
    return inspect_run(run_root, binding_sha256=binding_sha256).generations[-1]


@contextmanager
def writer_lock(run_root: Path) -> Iterator[None]:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    descriptor = os.open(run_root / "writer.lock", flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RecoveryIntegrityError("writer lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryBusyError("another writer owns this run") from exc
        yield
    finally:
        os.close(descriptor)


@contextmanager
def seed_request_lock(run_root: Path) -> Iterator[None]:
    """Serialize start/continue requests before the run directory exists."""

    if not run_root.name or run_root.name in {".", ".."}:
        raise RecoveryIntegrityError("run path must end in one safe component")
    parent_info = os.lstat(run_root.parent)
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise RecoveryIntegrityError("run parent must be a real directory")
    lock_path = run_root.parent / f".{run_root.name}.writer.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RecoveryIntegrityError("seed request lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryBusyError("another writer owns this seed request") from exc
        yield
    finally:
        os.close(descriptor)


class RecoverableRunController:
    """Connect trainer epoch commits to one immutable per-seed run."""

    def __init__(self, request: SeedRunRequest) -> None:
        self.request = request
        self.run_root = request.run_root
        self.binding_sha256 = request.binding.raw_sha256
        self.previous_commit_sha256: str | None = None
        self.loader_contract: dict[str, object] | None = None
        self.session_id = uuid.uuid4().hex

    def prepare(
        self,
        trainer: EarlyStoppingTrainer,
        train_loader: object,
        validation_loader: object,
    ) -> tuple[Path, TrainerProgress | None, Callable[[TrainerProgress], None]]:
        contract = _loader_contract(train_loader, validation_loader)
        self.loader_contract = contract
        initial_progress: TrainerProgress | None
        if self.request.mode == "start":
            pinned_inputs = read_pinned_binding_inputs(self.request.binding)
            progress = trainer.capture_progress(
                next_epoch=0,
                best_epoch=-1,
                best_loss=float("inf"),
                stale_epochs=0,
                history=[],
                terminal_reason=None,
                best_checkpoint=None,
            )
            state = encode_progress(progress, self.binding_sha256, contract)
            binding_bytes = self.request.binding.raw_bytes
            if binding_bytes is None:
                raise RecoveryIntegrityError("binding source bytes are unavailable")
            if hashlib.sha256(binding_bytes).hexdigest() != self.binding_sha256:
                raise RecoveryIntegrityError("binding is not in canonical byte form")
            create_run(
                self.run_root.parent,
                self.run_root.name,
                binding_bytes=binding_bytes,
                binding_sha256=self.binding_sha256,
                initial_state=state,
                input_files={path: item.data for path, item in pinned_inputs.items()},
            )
            inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
            initial_progress = None
        elif self.request.mode == "continue":
            inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
            if inspection.status in {RunStatus.EXPORTS_READY, RunStatus.COMPLETED}:
                raise RecoveryIntegrityError("completed exports do not require trainer preparation")
            latest = inspection.generations[-1]
            initial_progress = decode_progress(
                latest.state,
                binding_sha256=self.binding_sha256,
                loader_contract=contract,
                trainer=trainer,
            )
            restore_rng_state(latest.state["rng_state"])
        else:
            raise RecoveryIntegrityError("controller mode must be start or continue")
        self.previous_commit_sha256 = inspection.generations[-1].commit_sha256
        output = self.run_root / "staging" / f"export-{self.session_id}"
        output.mkdir(mode=0o700)

        def commit(progress: TrainerProgress) -> None:
            if self.loader_contract is None or self.previous_commit_sha256 is None:
                raise RecoveryIntegrityError("controller was not prepared")
            state = encode_progress(progress, self.binding_sha256, self.loader_contract)
            generation = publish_generation(
                self.run_root,
                state,
                binding_sha256=self.binding_sha256,
                expected_previous_commit_sha256=self.previous_commit_sha256,
            )
            self.previous_commit_sha256 = generation.commit_sha256

        return output, initial_progress, commit

    def materialize_best_checkpoint(self, output: Path) -> None:
        inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        if inspection.status is not RunStatus.FIT_COMPLETE or inspection.terminal is None:
            raise RecoveryIntegrityError("best checkpoint requires a terminal fit generation")
        if output.parent != self.run_root / "staging" or output.is_symlink():
            raise RecoveryIntegrityError("checkpoint staging path is outside the run")
        best_checkpoint = inspection.terminal.state.get("best_checkpoint")
        if not isinstance(best_checkpoint, dict):
            raise RecoveryIntegrityError("terminal state has no selected best checkpoint")
        _write_exclusive(output / "best.ckpt", _state_bytes(best_checkpoint))

    def _require_export_output(self, output: Path) -> Path:
        if output.parent != self.run_root / "staging" or output.is_symlink():
            raise RecoveryIntegrityError("export staging path is outside the run")
        return Path("staging") / output.name

    def _validated_stage_pin(
        self,
        stage: str,
        terminal_commit_sha256: str,
    ) -> dict[str, str] | None:
        if stage not in _STAGE_ROLES:
            raise RecoveryIntegrityError("unknown export stage")
        relative_directory = Path("export-stages") / stage / "e0001"
        directory = self.run_root / relative_directory
        if not directory.exists() and not directory.is_symlink():
            return None
        record_relative = relative_directory / "record.json"
        record_raw = read_regular_file_nofollow(self.run_root, record_relative)
        record = parse_json_bytes(record_raw.data, f"{stage} export stage")
        if frozenset(record) != _STAGE_RECORD_KEYS:
            raise RecoveryIntegrityError("export stage record keys differ")
        expected_dependencies: dict[str, dict[str, str]] = {}
        for predecessor in _STAGE_PREDECESSORS[stage]:
            pin = self._validated_stage_pin(predecessor, terminal_commit_sha256)
            if pin is None:
                raise RecoveryIntegrityError("export stage predecessor is missing")
            expected_dependencies[predecessor] = pin
        if (
            record["schema_version"] != 1
            or isinstance(record["schema_version"], bool)
            or record["purpose"] != "racepv_export_stage"
            or record["binding_sha256"] != self.binding_sha256
            or record["terminal_commit_sha256"] != terminal_commit_sha256
            or record["stage"] != stage
            or record["dependencies"] != expected_dependencies
        ):
            raise RecoveryIntegrityError("export stage identity differs")
        files = record["files"]
        expected_names = list(_STAGE_ROLES[stage])
        if (
            not isinstance(files, list)
            or not all(isinstance(item, dict) for item in files)
            or [item.get("path") for item in files] != expected_names
        ):
            raise RecoveryIntegrityError("export stage file list differs")
        for item in files:
            if not isinstance(item, dict) or frozenset(item) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise RecoveryIntegrityError("export stage file record is invalid")
            observed = read_regular_file_nofollow(
                self.run_root,
                relative_directory / str(item["path"]),
            )
            if observed.sha256 != item["sha256"] or observed.size_bytes != item["size_bytes"]:
                raise RecoveryIntegrityError("export stage file drifted")
        return {"path": record_relative.as_posix(), "sha256": record_raw.sha256}

    def publish_export_stage(self, stage: str, output: Path) -> dict[str, str]:
        inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        if inspection.terminal is None:
            raise RecoveryIntegrityError("export stage requires a terminal fit generation")
        output_relative = self._require_export_output(output)
        existing = self._validated_stage_pin(stage, inspection.terminal.commit_sha256)
        if existing is not None:
            return existing
        dependencies: dict[str, dict[str, str]] = {}
        for predecessor in _STAGE_PREDECESSORS.get(stage, ()):
            pin = self._validated_stage_pin(predecessor, inspection.terminal.commit_sha256)
            if pin is None:
                raise RecoveryIntegrityError("export stage predecessor is missing")
            dependencies[predecessor] = pin
        if stage not in _STAGE_ROLES:
            raise RecoveryIntegrityError("unknown export stage")
        staging = self.run_root / "staging" / f"stage-{stage}-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        files: list[dict[str, object]] = []
        for name in _STAGE_ROLES[stage]:
            observed = read_regular_file_nofollow(self.run_root, output_relative / name)
            _write_exclusive(staging / name, observed.data)
            files.append(
                {"path": name, "sha256": observed.sha256, "size_bytes": observed.size_bytes}
            )
        record = {
            "schema_version": 1,
            "purpose": "racepv_export_stage",
            "binding_sha256": self.binding_sha256,
            "terminal_commit_sha256": inspection.terminal.commit_sha256,
            "stage": stage,
            "dependencies": dependencies,
            "files": files,
        }
        _write_exclusive(staging / "record.json", _canonical_json(record))
        _fsync_directory(staging)
        destination = self.run_root / "export-stages" / stage / "e0001"
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
        pin = self._validated_stage_pin(stage, inspection.terminal.commit_sha256)
        if pin is None:  # pragma: no cover - publication and validation are contiguous
            raise RecoveryIntegrityError("published export stage disappeared")
        return pin

    def restore_export_stage(self, stage: str, output: Path) -> bool:
        inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        if inspection.terminal is None:
            raise RecoveryIntegrityError("export stage restore requires terminal fit")
        output_relative = self._require_export_output(output)
        pin = self._validated_stage_pin(stage, inspection.terminal.commit_sha256)
        if pin is None:
            return False
        stage_directory = Path(pin["path"]).parent
        for name in _STAGE_ROLES[stage]:
            source = read_regular_file_nofollow(self.run_root, stage_directory / name)
            destination = output / name
            if destination.exists() or destination.is_symlink():
                current = read_regular_file_nofollow(self.run_root, output_relative / name)
                if current.sha256 != source.sha256 or current.size_bytes != source.size_bytes:
                    raise RecoveryIntegrityError("restored export destination differs")
            else:
                _write_exclusive(destination, source.data)
        _fsync_directory(output)
        return True

    def _publish_acceptance(self, inspection: RunInspection) -> RunInspection:
        if inspection.status is not RunStatus.EXPORTS_READY or inspection.terminal is None:
            raise RecoveryIntegrityError("acceptance requires verified ready exports")
        if inspection.export_commit_sha256 is None:
            raise RecoveryIntegrityError("ready exports have no commit identity")
        session_directory = self.run_root / "sessions" / self.session_id
        session_directory.mkdir(mode=0o700)
        now = datetime.now(UTC).isoformat()
        exit_record = {
            "schema_version": 1,
            "purpose": "racepv_recovery_session_exit",
            "session_id": self.session_id,
            "action": "verify_exports",
            "binding_sha256": self.binding_sha256,
            "started_at": now,
            "finished_at": now,
            "worker_returncode": 0,
            "prior_session_outcome": "unknown_after_interruption",
            "log_files": [],
        }
        exit_path = session_directory / "exit.json"
        _write_exclusive(exit_path, _canonical_json(exit_record))
        _fsync_directory(session_directory)
        acceptance = {
            "schema_version": 1,
            "purpose": "racepv_recoverable_run_acceptance",
            "binding_sha256": self.binding_sha256,
            "terminal_commit_sha256": inspection.terminal.commit_sha256,
            "export_commit_sha256": inspection.export_commit_sha256,
            "session_receipt": {
                "path": exit_path.relative_to(self.run_root).as_posix(),
                "sha256": hashlib.sha256(exit_path.read_bytes()).hexdigest(),
            },
            "status": "completed",
        }
        _write_exclusive(
            self.run_root / "receipts" / "accepted.json",
            _canonical_json(acceptance),
        )
        _fsync_directory(self.run_root / "receipts")
        return inspect_run(self.run_root, binding_sha256=self.binding_sha256)

    def accept_existing_exports(self) -> RunInspection:
        inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        return self._publish_acceptance(inspection)

    def publish_completed_exports(self, output: Path) -> RunInspection:
        inspection = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        if inspection.status is not RunStatus.FIT_COMPLETE or inspection.terminal is None:
            raise RecoveryIntegrityError("exports require a terminal fit generation")
        self._require_export_output(output)
        run_raw = read_regular_file_nofollow(
            self.run_root, Path("staging") / output.name / "run.json"
        )
        run = parse_json_bytes(run_raw.data, "run manifest")
        artifacts = run.get("artifacts")
        protocol = self.request.binding.protocol_identity
        expected_artifacts = {
            "train_npz_sha256": protocol["train_descriptor_sha256"],
            "train_window_content_sha256": protocol["train_content_sha256"],
            "validation_npz_sha256": protocol["validation_descriptor_sha256"],
            "validation_window_content_sha256": protocol["validation_content_sha256"],
            "config_sha256": protocol["common_config_sha256"],
            "calibration_npz_sha256": None,
            "calibration_window_content_sha256": None,
            "test_npz_sha256": None,
            "test_window_content_sha256": None,
        }
        if (
            run.get("seed") != self.request.binding.seed
            or isinstance(run.get("seed"), bool)
            or run.get("selection_boundary") != "validation_only"
            or run.get("diagnostic_only") is not False
            or not isinstance(artifacts, dict)
            or any(artifacts.get(key) != value for key, value in expected_artifacts.items())
        ):
            raise RecoveryIntegrityError("run protocol identity differs from binding")
        stage_pins: dict[str, dict[str, str]] = {}
        for stage in ("model", "validation", "summary"):
            pin = self._validated_stage_pin(stage, inspection.terminal.commit_sha256)
            if pin is None:
                pin = self.publish_export_stage(stage, output)
            stage_pins[stage] = pin
        files: list[dict[str, object]] = []
        for name in _REFERENCE_FILES:
            source_stage = (
                "model"
                if name in _STAGE_ROLES["model"]
                else "validation"
                if name in _STAGE_ROLES["validation"]
                else "summary"
            )
            stage_directory = Path(stage_pins[source_stage]["path"]).parent
            observed = read_regular_file_nofollow(self.run_root, stage_directory / name)
            files.append(
                {"path": name, "sha256": observed.sha256, "size_bytes": observed.size_bytes}
            )
            destination_file = output / name
            if destination_file.exists() or destination_file.is_symlink():
                current = read_regular_file_nofollow(
                    self.run_root, Path("staging") / output.name / name
                )
                if current.sha256 != observed.sha256:
                    raise RecoveryIntegrityError("final export staging differs from stage")
            else:
                _write_exclusive(destination_file, observed.data)
        export = {
            "schema_version": 1,
            "purpose": "racepv_export_commit",
            "binding_sha256": self.binding_sha256,
            "terminal_commit_sha256": inspection.terminal.commit_sha256,
            "stages": stage_pins,
            "files": files,
        }
        _write_exclusive(output / "export-commit.json", _canonical_json(export))
        _fsync_directory(output)
        destination = self.run_root / "exports" / "e0001"
        _rename_noreplace(output, destination)
        _fsync_directory(destination.parent)
        ready = inspect_run(self.run_root, binding_sha256=self.binding_sha256)
        return self._publish_acceptance(ready)


def execute_seed_request(
    request: SeedRunRequest,
    *,
    worker: Callable[[SeedRunRequest], SessionResult],
) -> SessionResult:
    if request.mode not in {"continue", "start"}:
        raise RecoveryIntegrityError("worker mode must be start or continue")
    with seed_request_lock(request.run_root):
        if request.mode == "start":
            if request.run_root.exists() or request.run_root.is_symlink():
                raise RecoveryIntegrityError("run directory already exists")
            return worker(request)
        inspection = inspect_run(
            request.run_root,
            binding_sha256=request.binding.raw_sha256,
        )
        if inspection.status is RunStatus.COMPLETED:
            return SessionResult("skipped_completed", RunStatus.COMPLETED, 0)
        with writer_lock(request.run_root):
            inspection = inspect_run(
                request.run_root,
                binding_sha256=request.binding.raw_sha256,
            )
            if inspection.status is RunStatus.COMPLETED:
                return SessionResult("skipped_completed", RunStatus.COMPLETED, 0)
            return worker(request)


def read_generation_chain(
    run_root: Path,
    *,
    binding_sha256: str,
) -> tuple[CommittedGeneration, ...]:
    epochs = run_root / "epochs"
    if not epochs.is_dir() or (epochs / "g0000").is_symlink():
        raise RecoveryUnsupportedError("run has no committed g0000 generation")
    names: list[tuple[int, str]] = []
    for entry in epochs.iterdir():
        match = _GENERATION.fullmatch(entry.name)
        if match is None:
            raise RecoveryIntegrityError(f"unexpected epoch entry: {entry.name}")
        if entry.is_symlink() or not entry.is_dir():
            raise RecoveryIntegrityError("epoch generation must be a real directory")
        names.append((int(match.group(1)), entry.name))
    names.sort()
    if [index for index, _ in names] != list(range(len(names))):
        raise RecoveryIntegrityError("epoch generation chain has a gap")
    generations: list[CommittedGeneration] = []
    previous: str | None = None
    for index, name in names:
        commit_raw = read_regular_file_nofollow(
            run_root,
            Path("epochs") / name / "commit.json",
        )
        commit = parse_json_bytes(commit_raw.data, "epoch commit")
        validate_commit(commit, index, binding_sha256)
        if commit["previous_commit_sha256"] != previous:
            raise RecoveryIntegrityError("epoch commit predecessor mismatch")
        state_raw = read_regular_file_nofollow(
            run_root,
            Path("epochs") / name / "state.pt",
        )
        if state_raw.sha256 != commit["state_sha256"]:
            raise RecoveryIntegrityError("epoch state SHA-256 mismatch")
        state = load_tensor_mapping(run_root, Path("epochs") / name / "state.pt")
        if state.get("generation") != index or state.get("binding_sha256") != binding_sha256:
            raise RecoveryIntegrityError("epoch state identity mismatch")
        loader_contract = state.get("loader_contract")
        if not isinstance(loader_contract, dict):
            raise RecoveryIntegrityError("epoch loader contract is invalid")
        decode_progress(
            state,
            binding_sha256=binding_sha256,
            loader_contract=loader_contract,
        )
        generations.append(CommittedGeneration(index, commit_raw.sha256, commit, state))
        previous = commit_raw.sha256
    return tuple(generations)


def infer_run_inspection(
    binding: RunBinding,
    generations: tuple[CommittedGeneration, ...],
) -> RunInspection:
    if not generations:
        raise RecoveryUnsupportedError("run has no committed g0000 generation")
    latest = generations[-1]
    terminal_reason = latest.state.get("terminal_reason")
    terminal = latest if terminal_reason in {"early_stopping", "max_epochs"} else None
    if terminal is not None:
        status = RunStatus.FIT_COMPLETE
    elif latest.generation == 0:
        status = RunStatus.INITIAL_COMMITTED
    else:
        status = RunStatus.TRAINING_COMMITTED
    return RunInspection(status, binding, generations, terminal, None, None)


def inspect_run(run_root: Path, *, binding_sha256: str) -> RunInspection:
    if not (run_root / "epochs" / "g0000").is_dir():
        raise RecoveryUnsupportedError("run has no committed g0000 generation")
    binding = load_binding(run_root, Path("binding.json"), binding_sha256)
    read_pinned_binding_inputs(binding)
    generations = read_generation_chain(run_root, binding_sha256=binding_sha256)
    inspection = infer_run_inspection(binding, generations)
    if inspection.terminal is None:
        return inspection
    export_path = run_root / "exports" / "e0001" / "export-commit.json"
    if not os.path.lexists(export_path):
        return inspection
    export_raw = read_regular_file_nofollow(run_root, Path("exports/e0001/export-commit.json"))
    export = parse_json_bytes(export_raw.data, "export commit")
    if frozenset(export) != _EXPORT_COMMIT_KEYS:
        raise RecoveryIntegrityError("export commit keys differ")
    if export["schema_version"] != 1 or export["purpose"] != "racepv_export_commit":
        raise RecoveryIntegrityError("export commit schema is invalid")
    if export["binding_sha256"] != binding_sha256:
        raise RecoveryIntegrityError("export commit binding mismatch")
    if export["terminal_commit_sha256"] != inspection.terminal.commit_sha256:
        raise RecoveryIntegrityError("export commit terminal mismatch")
    files = export["files"]
    if (
        not isinstance(files, list)
        or not all(isinstance(item, dict) for item in files)
        or [item.get("path") for item in files] != list(_REFERENCE_FILES)
    ):
        raise RecoveryIntegrityError("export file list differs")
    for item in files:
        if not isinstance(item, dict) or frozenset(item) != {"path", "sha256", "size_bytes"}:
            raise RecoveryIntegrityError("export file record is invalid")
        relative = Path("exports/e0001") / str(item["path"])
        observed = read_regular_file_nofollow(run_root, relative)
        if observed.sha256 != item["sha256"] or observed.size_bytes != item["size_bytes"]:
            raise RecoveryIntegrityError("exported file identity mismatch")
    stages = export["stages"]
    if not isinstance(stages, dict) or frozenset(stages) != {"model", "validation", "summary"}:
        raise RecoveryIntegrityError("export stage pins differ")
    for stage, pin in stages.items():
        if not isinstance(pin, dict) or frozenset(pin) != {"path", "sha256"}:
            raise RecoveryIntegrityError("export stage pin is invalid")
        observed = read_regular_file_nofollow(run_root, Path(str(pin["path"])))
        if observed.sha256 != pin["sha256"]:
            raise RecoveryIntegrityError(f"{stage} export stage drifted")
    stage_validator = RecoverableRunController(SeedRunRequest("continue", run_root, binding))
    for stage in ("model", "validation", "summary"):
        validated = stage_validator._validated_stage_pin(stage, inspection.terminal.commit_sha256)
        if validated != stages[stage]:
            raise RecoveryIntegrityError(f"{stage} export stage identity differs")
    ready = RunInspection(
        RunStatus.EXPORTS_READY,
        binding,
        generations,
        inspection.terminal,
        export_raw.sha256,
        None,
    )
    accepted_path = run_root / "receipts" / "accepted.json"
    if not os.path.lexists(accepted_path):
        return ready
    accepted_raw = read_regular_file_nofollow(run_root, Path("receipts/accepted.json"))
    accepted = parse_json_bytes(accepted_raw.data, "acceptance")
    if frozenset(accepted) != _ACCEPTANCE_KEYS:
        raise RecoveryIntegrityError("acceptance keys differ")
    if (
        accepted["schema_version"] != 1
        or accepted["purpose"] != "racepv_recoverable_run_acceptance"
        or accepted["status"] != "completed"
        or accepted["binding_sha256"] != binding_sha256
        or accepted["terminal_commit_sha256"] != inspection.terminal.commit_sha256
        or accepted["export_commit_sha256"] != export_raw.sha256
    ):
        raise RecoveryIntegrityError("acceptance identity mismatch")
    receipt = accepted["session_receipt"]
    if not isinstance(receipt, dict) or frozenset(receipt) != {"path", "sha256"}:
        raise RecoveryIntegrityError("session receipt pin is invalid")
    receipt_path = str(receipt["path"])
    receipt_parts = validate_relative_path(receipt_path)
    if len(receipt_parts) != 3 or receipt_parts[0] != "sessions" or receipt_parts[2] != "exit.json":
        raise RecoveryIntegrityError("session receipt path is invalid")
    receipt_raw = read_regular_file_nofollow(run_root, Path(receipt_path))
    if receipt_raw.sha256 != receipt["sha256"]:
        raise RecoveryIntegrityError("session receipt drifted")
    receipt_payload = parse_json_bytes(receipt_raw.data, "session receipt")
    if frozenset(receipt_payload) != _SESSION_RECEIPT_KEYS:
        raise RecoveryIntegrityError("session receipt keys differ")
    if (
        receipt_payload["schema_version"] != 1
        or isinstance(receipt_payload["schema_version"], bool)
        or receipt_payload["purpose"] != "racepv_recovery_session_exit"
        or receipt_payload["binding_sha256"] != binding_sha256
        or receipt_payload["worker_returncode"] != 0
        or isinstance(receipt_payload["worker_returncode"], bool)
        or receipt_payload["action"] not in {"start", "continue", "export_only", "verify_exports"}
        or receipt_payload["prior_session_outcome"]
        not in {"observed_exit", "unknown_after_interruption", "not_applicable"}
        or not isinstance(receipt_payload["started_at"], str)
        or not receipt_payload["started_at"]
        or not isinstance(receipt_payload["finished_at"], str)
        or not receipt_payload["finished_at"]
        or not isinstance(receipt_payload["log_files"], list)
    ):
        raise RecoveryIntegrityError("session receipt semantics are invalid")
    return RunInspection(
        RunStatus.COMPLETED,
        binding,
        generations,
        inspection.terminal,
        export_raw.sha256,
        accepted_raw.sha256,
    )
