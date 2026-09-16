"""Common prediction cells -> versioned scores -> hourly/calendar statistics.

Pure in-memory adapter. Artifact/checkpoint/source qualification and permission
to load real arrays remain the caller's responsibility; no formal admission here.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from racepv.metrics.calendar_bootstrap import _validate_axes, summarize_calendar_losses
from racepv.metrics.quantile_scoring import quantile_score_cells, quantile_score_summary
from racepv.metrics.temporal_inference import _ns, _times, build_hourly_origin_losses

_STEP = 900_000_000_000
_SEEDS = (13, 29, 47, 71, 101)
_FIELDS = {
    "model_id",
    "seed_id",
    "site_id",
    "issue_ns",
    "target_start_ns",
    "targets",
    "quantiles",
    "eligible_mask",
    "levels",
}


def _record_arrays(record, *, anchor, start, stop):
    issue = _times(record["issue_ns"])
    if not issue or len(set(issue)) != len(issue):
        raise ValueError("nonempty unique issue times required")
    if issue != sorted(issue) or any(not start <= t < stop or (t - anchor) % _STEP for t in issue):
        raise ValueError("issue times must be sorted and on the frozen in-range grid")
    truth = np.asarray(record["targets"])
    predictions = np.asarray(record["quantiles"])
    mask = np.asarray(record["eligible_mask"])
    levels = np.asarray(record["levels"])
    target_time = np.asarray(record["target_start_ns"])
    shape = (len(issue), 24)
    if truth.shape != shape or truth.dtype.kind not in "iuf":
        raise ValueError("targets must have numeric origin x 24 shape")
    if mask.shape != shape or mask.dtype.kind != "b":
        raise ValueError("eligible mask must be boolean with exact cell shape")
    if target_time.shape != shape or target_time.dtype.kind not in "iu":
        raise ValueError("target start times must be integer origin x 24")
    expected = np.array(
        [[_ns(t + lead * _STEP) for lead in range(24)] for t in issue], dtype=np.int64
    )
    for t in issue:
        _ns(t + 24 * _STEP)
    if not np.array_equal(target_time, expected):
        raise ValueError("target interval starts do not match explicit issue and lead")
    if levels.ndim != 1 or not len(levels) or predictions.shape != shape + (len(levels),):
        raise ValueError("quantile axes do not match target cells")
    return {
        "issue_ns": np.array(issue, dtype=np.int64),
        "target_start_ns": expected,
        "targets": truth,
        "quantiles": predictions,
        "eligible_mask": mask,
        "levels": levels,
    }


def build_common_cell_statistics(
    records,
    *,
    model_kinds,
    site_ids,
    calendar_days,
    calendar_timezone,
    hour_anchor_ns,
    range_start_ns,
    range_stop_ns,
    score_id,
    mask_id,
):
    """Use exactly the same eligible cells for all frozen models and seeds.

    No intersection, implicit daylight restriction, zero-filled predictions or
    ensemble-before-scoring. Calendar axes are supplied, never inferred by taking
    the dates where one model happens to have results.
    """
    if score_id != "qcrps_trapezoid_zero_endpoints_v1" or mask_id != "overall_eligible_v1":
        raise ValueError("explicit registered overall qCRPS contract required")
    sites, days = _validate_axes(site_ids, calendar_days)
    anchor, start, stop = map(_ns, (hour_anchor_ns, range_start_ns, range_stop_ns))
    if start >= stop:
        raise ValueError("nonempty origin range required")
    if type(calendar_timezone) is not str:
        raise ValueError("explicit calendar timezone required")
    try:
        zone = ZoneInfo(calendar_timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("unknown calendar timezone") from exc
    if type(model_kinds) is not dict or not model_kinds:
        raise ValueError("explicit nonempty model kinds required")
    seeds = {}
    for model, kind in model_kinds.items():
        if type(model) is not str or not model or kind not in ("neural", "deterministic"):
            raise ValueError("invalid model identity")
        seeds[model] = _SEEDS if kind == "neural" else ("deterministic",)
    required = {(model, seed, site) for model in seeds for seed in seeds[model] for site in sites}
    if type(records) not in (list, tuple) or not records:
        raise ValueError("prediction records required")
    indexed = {}
    for record in records:
        if type(record) is not dict or set(record) != _FIELDS:
            raise ValueError("exact prediction record fields required")
        model, seed, site = record["model_id"], record["seed_id"], record["site_id"]
        if type(model) is not str or type(site) is not str or type(seed) not in (str, int):
            raise ValueError("invalid model/seed/site types")
        key = model, seed, site
        if key not in required or key in indexed:
            raise ValueError("unexpected or duplicate model/seed/site")
        indexed[key] = _record_arrays(record, anchor=anchor, start=start, stop=stop)
    if set(indexed) != required:
        raise ValueError("missing model, seed or site; cannot form a result intersection")
    common = {}
    for (_, _, site), arrays in indexed.items():
        if site not in common:
            common[site] = arrays
        else:
            for field in ("issue_ns", "target_start_ns", "targets", "eligible_mask", "levels"):
                if not np.array_equal(arrays[field], common[site][field], equal_nan=True):
                    raise ValueError("common prediction population mismatch: " + field)
    # One explicit quantile grid across sites, not just within each site.
    if any(not np.array_equal(common[s]["levels"], common[sites[0]]["levels"]) for s in sites):
        raise ValueError("cross-site quantile levels differ")
    day_lookup = {day: i for i, day in enumerate(days)}
    day_indices = {}
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    for site in sites:
        times = common[site]["target_start_ns"]
        indexes = np.empty(times.shape, dtype=np.int64)
        for position in np.ndindex(times.shape):
            day = (
                (epoch + timedelta(seconds=int(times[position]) // 1_000_000_000))
                .astimezone(zone)
                .date()
                .isoformat()
            )
            if day not in day_lookup:
                raise ValueError("target date outside frozen full calendar")
            indexes[position] = day_lookup[day]
        day_indices[site] = indexes
    calendar_models, hourly = [], {}
    for model, kind in model_kinds.items():
        nseeds = len(seeds[model])
        sums = np.zeros((nseeds, len(sites), len(days)), dtype=np.float64)
        counts = np.zeros(sums.shape, dtype=np.int64)
        hourly[model] = {}
        for site_index, site in enumerate(sites):
            n = len(common[site]["issue_ns"])
            origin_sums = np.zeros((nseeds, n))
            origin_counts = np.zeros((nseeds, n), dtype=np.int64)
            for seed_index, seed in enumerate(seeds[model]):
                a = indexed[model, seed, site]
                cells = quantile_score_cells(
                    a["targets"],
                    a["quantiles"],
                    a["levels"],
                    eligible_mask=a["eligible_mask"],
                    score_id=score_id,
                )
                for day_index in range(len(days)):
                    selected = a["eligible_mask"] & (day_indices[site] == day_index)
                    summary = quantile_score_summary(cells, eligible_mask=selected)
                    sums[seed_index, site_index, day_index] = summary["loss_sum"]
                    counts[seed_index, site_index, day_index] = summary["count"]
                for origin in range(n):
                    summary = quantile_score_summary(
                        cells[origin], eligible_mask=a["eligible_mask"][origin]
                    )
                    origin_sums[seed_index, origin] = summary["loss_sum"]
                    origin_counts[seed_index, origin] = summary["count"]
            hourly[model][site] = build_hourly_origin_losses(
                origin_sums,
                origin_counts,
                common[site]["issue_ns"],
                hour_anchor_ns=anchor,
                range_start_ns=start,
                range_stop_ns=stop,
            )
        calendar_models.append(
            dict(
                model_id=model, kind=kind, seed_ids=seeds[model], loss_sum=sums, valid_count=counts
            )
        )
    summary = summarize_calendar_losses(calendar_models, site_ids=sites, calendar_days=days)
    return dict(
        kind="racepv.common-cell-statistics",
        version=1,
        score_id=score_id,
        mask_id=mask_id,
        calendar_timezone=calendar_timezone,
        calendar_models=calendar_models,
        hourly=hourly,
        summary=summary,
        formal_ready=False,
    )
