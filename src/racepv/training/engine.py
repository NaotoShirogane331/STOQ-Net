"""A small auditable trainer for RaCE-PV and neural baselines."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

Batch = Mapping[str, Tensor]
Criterion = Callable[[Mapping[str, Tensor], Tensor, Tensor, Tensor | None], Any]


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    seconds: float
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    validation_selection_metric: float = float("nan")


@dataclass
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    best_validation_metric_name: str = "loss"
    history: list[EpochRecord] = field(default_factory=list)
    stopped_early: bool = False


@dataclass
class TrainerProgress:
    next_epoch: int
    best_epoch: int
    best_validation_loss: float
    stale_epochs: int
    history: list[EpochRecord]
    terminal_reason: str | None
    current_model_state: dict[str, Tensor]
    best_checkpoint: dict[str, object] | None
    optimizer_state: dict[str, object]
    scaler_state: dict[str, object]


def _model_inputs(batch: Batch) -> dict[str, Tensor]:
    required = ("history_values", "history_mask", "future_known", "site_metadata")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"batch is missing model inputs: {missing}")
    inputs = {key: batch[key] for key in required}
    weather_keys = ("future_weather", "future_weather_mask")
    present_weather = [key for key in weather_keys if key in batch]
    if present_weather and len(present_weather) != len(weather_keys):
        raise KeyError("future weather values and mask must be supplied as an atomic pair")
    inputs.update({key: batch[key] for key in weather_keys if key in batch})
    return inputs


def _loss_value(result: Any) -> Tensor:
    if isinstance(result, Tensor):
        return result
    if isinstance(result, Mapping):
        for key in ("loss", "total", "total_loss"):
            if key in result:
                return result[key]
    if hasattr(result, "total"):
        return result.total
    if hasattr(result, "loss"):
        return result.loss
    raise TypeError(
        "criterion must return a tensor, mapping with loss, or object with .total/.loss"
    )


def _state_classification_metrics(
    probabilities: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    """Compute exact multiclass Brier, NLL, and macro one-vs-rest AUPRC."""

    probabilities = probabilities.detach().float().cpu()
    labels = labels.detach().long().cpu()
    if probabilities.ndim != 2 or labels.shape != probabilities.shape[:1]:
        raise ValueError("probabilities must have shape [N,C] and labels shape [N]")
    if probabilities.shape[0] == 0:
        raise ValueError("state metrics require at least one labelled sample")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("state probabilities must be finite")
    if not bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()) or not torch.allclose(
        probabilities.sum(dim=-1), torch.ones(probabilities.shape[0]), atol=1e-5
    ):
        raise ValueError("state probabilities must lie on the simplex")
    num_classes = probabilities.shape[-1]
    if not bool(((labels >= 0) & (labels < num_classes)).all()):
        raise ValueError("state labels are outside the probability dimension")

    expected = F.one_hot(labels, num_classes=num_classes).to(probabilities.dtype)
    brier = (probabilities - expected).square().sum(dim=-1).mean()
    selected = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    nll = -selected.clamp_min(torch.finfo(probabilities.dtype).tiny).log().mean()

    average_precisions: list[Tensor] = []
    ranks = torch.arange(1, probabilities.shape[0] + 1, dtype=probabilities.dtype)
    for state in range(num_classes):
        positives = labels == state
        if not bool(positives.any()):
            continue
        order = torch.argsort(probabilities[:, state], descending=True, stable=True)
        ordered_positive = positives[order]
        precision_at_rank = ordered_positive.cumsum(dim=0).to(probabilities.dtype) / ranks
        average_precisions.append(precision_at_rank[ordered_positive].mean())
    auprc = torch.stack(average_precisions).mean()
    return {
        "state_brier": float(brier),
        "state_nll": float(nll),
        "state_auprc": float(auprc),
    }


class EarlyStoppingTrainer:
    """Train with deterministic early stopping and restore the best state."""

    def __init__(
        self,
        model: nn.Module,
        criterion: Criterion,
        *,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 80,
        patience: int = 10,
        min_epochs: int = 0,
        gradient_clip_norm: float = 1.0,
        mixed_precision: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        selection_metric: str = "loss",
        separate_state_optimizer: bool = False,
        single_optimizer_schema: str = "single_optimizer_v1",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.criterion = (
            criterion.to(self.device) if isinstance(criterion, nn.Module) else criterion
        )
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        if type(min_epochs) is not int or not 0 <= min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must be an integer within the epoch budget")
        self.min_epochs = min_epochs
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        if not selection_metric:
            raise ValueError("selection_metric must be a non-empty metric name")
        self.selection_metric = str(selection_metric)
        self.separate_state_optimizer = bool(separate_state_optimizer)
        if not single_optimizer_schema:
            raise ValueError("single_optimizer_schema must be a non-empty string")
        self.single_optimizer_schema = str(single_optimizer_schema)
        self.optimizer: torch.optim.Optimizer | None = None
        self.forecast_optimizer: torch.optim.Optimizer | None = None
        self.state_optimizer: torch.optim.Optimizer | None = None
        self.forecast_parameters: tuple[nn.Parameter, ...] = ()
        self.state_parameters: tuple[nn.Parameter, ...] = ()
        if self.separate_state_optimizer:
            state_head = getattr(self.model, "state_head", None)
            if not isinstance(state_head, nn.Module):
                raise ValueError("separate_state_optimizer requires model.state_head")
            self.state_parameters = tuple(
                parameter for parameter in state_head.parameters() if parameter.requires_grad
            )
            state_ids = {id(parameter) for parameter in self.state_parameters}
            self.forecast_parameters = tuple(
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad and id(parameter) not in state_ids
            )
            all_trainable = tuple(
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            )
            if not self.state_parameters or not self.forecast_parameters:
                raise ValueError("separated optimization requires non-empty parameter groups")
            if len(state_ids) != len(self.state_parameters) or len(all_trainable) != (
                len(self.forecast_parameters) + len(self.state_parameters)
            ):
                raise RuntimeError(
                    "forecast and state parameter groups are not disjoint/exhaustive"
                )
            self.forecast_optimizer = torch.optim.AdamW(
                self.forecast_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.state_optimizer = torch.optim.AdamW(
                self.state_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.forecast_parameters = tuple(
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            )
            if not self.forecast_parameters:
                raise ValueError("single optimization requires at least one trainable parameter")
            self.optimizer = torch.optim.AdamW(
                self.forecast_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        amp_available = self.device.type == "cuda"
        self.use_amp = bool(mixed_precision and amp_available)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _assert_finite_model_parameters(self, *, stage: str) -> None:
        for name, parameter in self.model.named_parameters():
            if parameter.is_floating_point() and not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"non-finite model parameter {stage}: {name}")

    def _optimizers(self) -> tuple[torch.optim.Optimizer, ...]:
        if self.separate_state_optimizer:
            assert self.forecast_optimizer is not None
            assert self.state_optimizer is not None
            return (self.forecast_optimizer, self.state_optimizer)
        assert self.optimizer is not None
        return (self.optimizer,)

    def _assert_finite_optimizer_state(self, *, stage: str) -> None:
        for optimizer_index, optimizer in enumerate(self._optimizers()):
            for parameter, state in optimizer.state.items():
                for name, value in state.items():
                    if (
                        isinstance(value, Tensor)
                        and (value.is_floating_point() or value.is_complex())
                        and not bool(torch.isfinite(value).all())
                    ):
                        raise FloatingPointError(
                            "non-finite optimizer state "
                            f"{stage}: optimizer={optimizer_index}, "
                            f"parameter_id={id(parameter)}, state={name}"
                        )

    def _assert_finite_gradients(self, *, stage: str) -> None:
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if (
                gradient is not None
                and (gradient.is_floating_point() or gradient.is_complex())
                and not bool(torch.isfinite(gradient).all())
            ):
                raise FloatingPointError(f"non-finite gradient {stage}: {name}")

    def _assert_finite_training_state(self, *, stage: str) -> None:
        self._assert_finite_model_parameters(stage=stage)
        self._assert_finite_optimizer_state(stage=stage)
        self._assert_finite_gradients(stage=stage)

    def _clear_optimizer_gradients(self) -> None:
        for optimizer in self._optimizers():
            optimizer.zero_grad(set_to_none=True)

    def _step_separated_losses(
        self,
        *,
        forecast_loss: Tensor | None,
        state_loss: Tensor | None,
    ) -> None:
        """Update only the parameter group owned by each supplied objective."""

        if not self.separate_state_optimizer:
            raise RuntimeError("separated loss stepping was not enabled")
        if forecast_loss is None and state_loss is None:
            raise ValueError("at least one separated loss is required")
        assert self.forecast_optimizer is not None
        assert self.state_optimizer is not None
        self.forecast_optimizer.zero_grad(set_to_none=True)
        self.state_optimizer.zero_grad(set_to_none=True)
        if forecast_loss is not None:
            self.scaler.scale(forecast_loss).backward(retain_graph=state_loss is not None)
        if state_loss is not None:
            self.scaler.scale(state_loss).backward()
        if forecast_loss is not None:
            self.scaler.unscale_(self.forecast_optimizer)
            torch.nn.utils.clip_grad_norm_(self.forecast_parameters, self.gradient_clip_norm)
        if state_loss is not None:
            self.scaler.unscale_(self.state_optimizer)
            torch.nn.utils.clip_grad_norm_(self.state_parameters, self.gradient_clip_norm)
        if forecast_loss is not None:
            self.scaler.step(self.forecast_optimizer)
        if state_loss is not None:
            self.scaler.step(self.state_optimizer)
        self.scaler.update()
        self._clear_optimizer_gradients()
        self._assert_finite_training_state(stage="after optimizer step")

    def _move(self, batch: Batch) -> dict[str, Tensor]:
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _criterion(
        self, outputs: Mapping[str, Tensor], batch: Batch
    ) -> tuple[Tensor, dict[str, Tensor]]:
        daylight = batch.get("daylight_mask")
        from racepv.reliability_a_loss import ReliabilityAObjective

        candidate_masks = (
            {"target_mask": batch.get("target_mask")}
            if isinstance(self.criterion, ReliabilityAObjective)
            else {}
        )
        result = self.criterion(
            outputs,
            batch["targets"],
            batch["state_labels"],
            daylight_mask=daylight,
            **candidate_masks,
        )
        loss = _loss_value(result)
        metrics = {"loss": loss}
        if isinstance(result, Mapping):
            metrics.update(
                {
                    str(name): value
                    for name, value in result.items()
                    if isinstance(value, Tensor) and value.ndim == 0
                }
            )
        return loss, metrics

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        metric_totals: dict[str, float] = {}
        metric_weights: dict[str, float] = {}
        state_probabilities: list[Tensor] = []
        state_targets: list[Tensor] = []
        samples = 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for raw_batch in loader:
                batch = self._move(raw_batch)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(**_model_inputs(batch))
                    loss, batch_metrics = self._criterion(outputs, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss: {loss.detach().item()}")
                if training:
                    if self.separate_state_optimizer:
                        if (
                            "forecast_loss" not in batch_metrics
                            or "state_loss" not in batch_metrics
                        ):
                            raise KeyError(
                                "separated optimization requires forecast_loss and state_loss"
                            )
                        self._step_separated_losses(
                            forecast_loss=batch_metrics["forecast_loss"],
                            state_loss=batch_metrics["state_loss"],
                        )
                    else:
                        assert self.optimizer is not None
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.forecast_parameters, self.gradient_clip_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self._clear_optimizer_gradients()
                        self._assert_finite_training_state(stage="after optimizer step")
                batch_size = int(batch["targets"].shape[0])
                samples += batch_size
                finite_targets = torch.isfinite(batch["targets"])
                daylight = batch.get("daylight_mask")
                valid_daylight = finite_targets
                from racepv.reliability_a_loss import ReliabilityAObjective

                if isinstance(self.criterion, ReliabilityAObjective) and "target_mask" in batch:
                    valid_daylight = valid_daylight & batch["target_mask"].bool()
                if daylight is not None:
                    valid_daylight = valid_daylight & daylight.bool()
                daylight_weight = float(valid_daylight.sum())
                valid_states = (
                    valid_daylight & (batch["state_labels"] >= 0) & (batch["state_labels"] < 3)
                )
                state_weight = float(valid_states.sum())
                is_v2 = "daylight_qcrps" in batch_metrics
                for name, value in batch_metrics.items():
                    if name in {"state_loss", "state_ce", "state_brier", "state_nll"}:
                        weight = state_weight
                    elif name in {
                        "forecast_loss",
                        "daylight_qcrps",
                        "specialist_qcrps",
                        "ramp_auxiliary",
                        "tail_ramp_up_mse",
                        "tail_trigger_sparsity",
                    } or (name == "loss" and is_v2):
                        weight = daylight_weight
                    else:
                        weight = float(batch_size)
                    if weight <= 0:
                        continue
                    metric_totals[name] = (
                        metric_totals.get(name, 0.0) + float(value.detach()) * weight
                    )
                    metric_weights[name] = metric_weights.get(name, 0.0) + weight
                if "state_probs" in outputs and bool(valid_states.any()):
                    state_probabilities.append(outputs["state_probs"][valid_states].detach().cpu())
                    state_targets.append(batch["state_labels"][valid_states].detach().cpu())
        if samples == 0:
            raise ValueError("empty data loader")
        metrics = {name: total / metric_weights[name] for name, total in metric_totals.items()}
        if state_probabilities:
            metrics.update(
                _state_classification_metrics(
                    torch.cat(state_probabilities, dim=0),
                    torch.cat(state_targets, dim=0),
                )
            )
        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        validation_metric: float,
        validation_metrics: Mapping[str, float],
    ) -> None:
        if self.checkpoint_path is None:
            return
        self._assert_finite_model_parameters(stage="before checkpoint")
        self._assert_finite_optimizer_state(stage="before checkpoint")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        payload = self._checkpoint_payload(epoch, validation_metric, validation_metrics)
        torch.save(payload, temporary)
        temporary.replace(self.checkpoint_path)

    def _optimizer_payload(self) -> dict[str, Any]:
        parameter_names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        if self.separate_state_optimizer:
            assert self.forecast_optimizer is not None
            assert self.state_optimizer is not None
            return {
                "optimizer_schema": "separated_forecast_state_v1",
                "forecast_optimizer_state_dict": self.forecast_optimizer.state_dict(),
                "state_optimizer_state_dict": self.state_optimizer.state_dict(),
                "forecast_parameter_names": [
                    parameter_names[id(parameter)] for parameter in self.forecast_parameters
                ],
                "state_parameter_names": [
                    parameter_names[id(parameter)] for parameter in self.state_parameters
                ],
            }
        else:
            assert self.optimizer is not None
            return {
                "optimizer_schema": self.single_optimizer_schema,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "parameter_names": [
                    parameter_names[id(parameter)] for parameter in self.forecast_parameters
                ],
            }

    def _checkpoint_payload(
        self,
        epoch: int,
        validation_metric: float,
        validation_metrics: Mapping[str, float],
    ) -> dict[str, object]:
        return {
            "epoch": epoch,
            "validation_loss": validation_metric,
            "selection_metric": self.selection_metric,
            "validation_selection_metric": validation_metric,
            "validation_metrics": dict(validation_metrics),
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            **copy.deepcopy(self._optimizer_payload()),
            "metadata": copy.deepcopy(self.checkpoint_metadata),
        }

    def capture_progress(
        self,
        *,
        next_epoch: int,
        best_epoch: int,
        best_loss: float,
        stale_epochs: int,
        history: list[EpochRecord],
        terminal_reason: str | None,
        best_checkpoint: dict[str, object] | None,
    ) -> TrainerProgress:
        return TrainerProgress(
            next_epoch=next_epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stale_epochs=stale_epochs,
            history=copy.deepcopy(history),
            terminal_reason=terminal_reason,
            current_model_state=copy.deepcopy(self.model.state_dict()),
            best_checkpoint=copy.deepcopy(best_checkpoint),
            optimizer_state=copy.deepcopy(self._optimizer_payload()),
            scaler_state=copy.deepcopy(self.scaler.state_dict()),
        )

    def restore_progress(self, progress: TrainerProgress) -> None:
        self.model.load_state_dict(progress.current_model_state, strict=True)
        schema = progress.optimizer_state.get("optimizer_schema")
        if self.separate_state_optimizer:
            if schema != "separated_forecast_state_v1":
                raise ValueError("resume optimizer schema differs from separated trainer")
            assert self.forecast_optimizer is not None
            assert self.state_optimizer is not None
            self.forecast_optimizer.load_state_dict(
                progress.optimizer_state["forecast_optimizer_state_dict"]
            )
            self.state_optimizer.load_state_dict(
                progress.optimizer_state["state_optimizer_state_dict"]
            )
        else:
            if schema != self.single_optimizer_schema:
                raise ValueError("resume optimizer schema differs from single trainer")
            assert self.optimizer is not None
            self.optimizer.load_state_dict(progress.optimizer_state["optimizer_state_dict"])
        self.scaler.load_state_dict(progress.scaler_state)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        initial_progress: TrainerProgress | None = None,
        on_epoch_committed: Callable[[TrainerProgress], None] | None = None,
    ) -> TrainingResult:
        if initial_progress is None:
            best_loss = float("inf")
            best_epoch = -1
            best_checkpoint: dict[str, object] | None = None
            stale_epochs = 0
            history: list[EpochRecord] = []
            next_epoch = 0
        else:
            self.restore_progress(initial_progress)
            best_loss = float(initial_progress.best_validation_loss)
            best_epoch = int(initial_progress.best_epoch)
            best_checkpoint = copy.deepcopy(initial_progress.best_checkpoint)
            stale_epochs = int(initial_progress.stale_epochs)
            history = copy.deepcopy(initial_progress.history)
            next_epoch = int(initial_progress.next_epoch)
            if initial_progress.terminal_reason is not None:
                if best_checkpoint is None:
                    raise RuntimeError("terminal resume is missing its best checkpoint")
                self.model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
                return TrainingResult(
                    best_epoch=best_epoch,
                    best_validation_loss=best_loss,
                    best_validation_metric_name=self.selection_metric,
                    history=history,
                    stopped_early=len(history) < self.max_epochs,
                )

        for epoch in range(next_epoch, self.max_epochs):
            started = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            if self.selection_metric not in validation_metrics:
                raise KeyError(
                    f"validation metrics do not contain selection metric {self.selection_metric!r}"
                )
            validation_metric = float(validation_metrics[self.selection_metric])
            if not torch.isfinite(torch.tensor(validation_metric)):
                raise FloatingPointError(
                    f"non-finite validation selection metric: {validation_metric}"
                )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=float(train_metrics["loss"]),
                    validation_loss=float(validation_metrics["loss"]),
                    seconds=time.perf_counter() - started,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    validation_selection_metric=validation_metric,
                )
            )
            if validation_metric < best_loss:
                best_loss = validation_metric
                best_epoch = epoch
                best_checkpoint = self._checkpoint_payload(
                    epoch, validation_metric, validation_metrics
                )
                stale_epochs = 0
                self._save_checkpoint(epoch, validation_metric, validation_metrics)
                terminal_reason = None
            else:
                stale_epochs += 1
                terminal_reason = (
                    "early_stopping"
                    if stale_epochs >= self.patience and epoch + 1 >= self.min_epochs
                    else None
                )
            if terminal_reason is None and epoch + 1 >= self.max_epochs:
                terminal_reason = "max_epochs"
            progress = self.capture_progress(
                next_epoch=epoch + 1,
                best_epoch=best_epoch,
                best_loss=best_loss,
                stale_epochs=stale_epochs,
                history=history,
                terminal_reason=terminal_reason,
                best_checkpoint=best_checkpoint,
            )
            if on_epoch_committed is not None:
                on_epoch_committed(progress)
            if terminal_reason is not None:
                break

        if best_checkpoint is None:
            raise RuntimeError("training did not produce a valid checkpoint")
        self.model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        return TrainingResult(
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            best_validation_metric_name=self.selection_metric,
            history=history,
            stopped_early=len(history) < self.max_epochs,
        )

    @torch.inference_mode()
    def predict(self, loader: DataLoader) -> dict[str, Tensor]:
        self.model.eval()
        collected: dict[str, list[Tensor]] = {}
        for raw_batch in loader:
            batch = self._move(raw_batch)
            outputs = self.model(**_model_inputs(batch))
            for key, value in outputs.items():
                if isinstance(value, Tensor):
                    collected.setdefault(key, []).append(value.detach().cpu())
            for key in (
                "targets",
                "state_labels",
                "daylight_mask",
                "target_mask",
                "target_interval_start_ns",
                "target_local_day_ids",
                "issue_ns",
                "timestamps",
                "day_ids",
                "site_ids",
            ):
                if key in batch:
                    collected.setdefault(key, []).append(batch[key].detach().cpu())
        return {key: torch.cat(values, dim=0) for key, values in collected.items()}
