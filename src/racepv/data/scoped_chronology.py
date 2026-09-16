"""Actual selected-window chronology, independent of targets and qualification.

Accepts an already pinned scientific view and the selected store's resident
audit arrays. It does not load files, establish source authority, or replace
the caller's full store/content/binding verification.
"""

from collections.abc import Mapping

import numpy as np

from racepv.data import prepared_contract as contract
from racepv.data.origin_clock import GRID_STEP_NS, origin_times
from racepv.data.scoped_scientific_view import (
    _revalidate,
    require_train_validation_compatibility,
)

_AUDITS = ("labels_ns", "window_starts", "issue_ns", "timestamps")
_I64 = np.iinfo(np.int64)


def require_half_open_separation(earlier_end_exclusive_ns, later_history_start_ns):
    """Allow equality at a half-open boundary, never overlapping intervals."""
    for value in (earlier_end_exclusive_ns, later_history_start_ns):
        if type(value) is not int:
            raise TypeError("chronology boundaries require integer nanoseconds")
        if not _I64.min < value <= _I64.max:
            raise ValueError("chronology boundary is outside int64 or NaT")
    if earlier_end_exclusive_ns > later_history_start_ns:
        raise ValueError("actual forecast target/history intervals overlap")


def _audit_identities(schema, arrays):
    if not isinstance(arrays, Mapping) or any(role not in arrays for role in _AUDITS):
        raise ValueError("selected site is missing audit arrays")
    selected = {}
    for role in _AUDITS:
        array = arrays[role]
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype("int64"):
            raise TypeError("audit arrays must be int64 numpy arrays")
        if contract._bounded_array_identity(array) != schema["arrays"][role]:
            raise ValueError(f"audit array identity drift: {role}")
        selected[role] = array
    return selected


def _leaf_bounds(schema, arrays):
    selected = _audit_identities(schema, arrays)
    labels, starts, issues, targets = (selected[role] for role in _AUDITS)
    scope = schema["row_scope"]
    # All labels must match the admitted grid, not merely the first/last row.
    for offset in range(0, len(labels), 8192):
        block = labels[offset : offset + 8192]
        start = scope["stored_start_ns"] + offset * GRID_STEP_NS
        expected = start + np.arange(len(block), dtype=np.int64) * GRID_STEP_NS
        if not np.array_equal(block, expected):
            raise ValueError("audit labels disagree with the complete selected grid")
    geometry = schema["geometry"]
    lookback, horizon = geometry["lookback"], geometry["horizon"]
    count = len(starts)
    if count == 0:
        raise ValueError("selected site has no actual forecast windows")
    first_history = first_issue = last_issue = last_end = None
    previous = -1
    for offset in range(0, count, 64):
        batch = starts[offset : offset + 64]
        if (
            int(batch[0]) <= previous
            or np.any(batch[1:] <= batch[:-1])
            or np.any(batch < 0)
            or np.any(batch > len(labels) - lookback - horizon)
            or np.any(batch % geometry["stride"] != 0)
        ):
            raise ValueError("audit window starts are unordered or outside scoped geometry")
        previous = int(batch[-1])
        lo, hi = int(batch[0]), int(batch[-1]) + lookback + horizon
        times = origin_times(labels[lo:hi], batch - lo, lookback=lookback, horizon=horizon)
        stop = offset + len(batch)
        if not np.array_equal(issues[offset:stop], times.issue_ns):
            raise ValueError("audit issue must be last history interval end")
        if not np.array_equal(targets[offset:stop], times.target_start_ns):
            raise ValueError("audit targets disagree with actual window target intervals")
        if first_history is None:
            first_history = int(labels[int(batch[0])])
            first_issue = int(times.issue_ns[0])
        last_issue = int(times.issue_ns[-1])
        last_end = int(times.target_end_ns[-1, -1])
    # Recheck resident audit bytes after deriving bounds, not cached true flags.
    _audit_identities(schema, arrays)
    return {
        "window_count": count,
        "history_start_ns": first_history,
        "first_issue_ns": first_issue,
        "last_issue_ns": last_issue,
        "target_start_ns": first_issue,
        "target_end_exclusive_ns": last_end,
        "schema_sha256": contract.canonical_json_sha256(schema),
        "forecast_population_sha256": schema["forecast_population_sha256"],
        "audit_identities": {role: dict(schema["arrays"][role]) for role in _AUDITS},
    }


def scoped_temporal_bounds(view, arrays_by_site):
    """Verify every site's four audit arrays; never access model/target values."""
    projection = _revalidate(view)
    schemas = projection["schemas"]
    if not isinstance(arrays_by_site, Mapping) or set(arrays_by_site) != set(schemas):
        raise ValueError("audit site collection must match the entire scientific view")
    sites = {site: _leaf_bounds(schema, arrays_by_site[site]) for site, schema in schemas.items()}
    result = {
        "artifact_kind": "racepv.scoped-actual-window-chronology",
        "schema_version": 1,
        "selected_split": projection["selected_split"],
        "view_identity_sha256": view.identity_sha256,
        "endpoint_convention": "target_end_exclusive_lte_later_history_start",
        "sites": sites,
        "formal_ready": False,
    }
    for field in ("history_start_ns", "first_issue_ns", "target_start_ns"):
        result[field] = min(item[field] for item in sites.values())
    for field in ("last_issue_ns", "target_end_exclusive_ns"):
        result[field] = max(item[field] for item in sites.values())
    return result


def require_scoped_train_validation_chronology(
    train_view, train_arrays, validation_view, validation_arrays
):
    """Require global separation after exact cross-split scientific matching."""
    require_train_validation_compatibility(train_view, validation_view)
    train = scoped_temporal_bounds(train_view, train_arrays)
    validation = scoped_temporal_bounds(validation_view, validation_arrays)
    end, start = train["target_end_exclusive_ns"], validation["history_start_ns"]
    require_half_open_separation(end, start)
    return {
        "train": train,
        "validation": validation,
        "gap_ns": start - end,
        "passed": True,
        "formal_ready": False,
    }
