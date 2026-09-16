"""Dataset eligibility checks fixed before model training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .adapters import ManifestEntry


@dataclass(frozen=True)
class SiteQualityReport:
    site_id: str
    continuous_months: int
    daylight_completeness: float
    observed_daylight_points: int
    expected_daylight_points: int
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DKASCManifestSelection:
    """Selected manifest entries plus auditable reports for every candidate."""

    entries: tuple[ManifestEntry, ...]
    reports: tuple[SiteQualityReport, ...]

    @property
    def selected_site_ids(self) -> tuple[str, ...]:
        return tuple(entry.site_id for entry in self.entries)

    @property
    def rejected_reports(self) -> tuple[SiteQualityReport, ...]:
        selected = set(self.selected_site_ids)
        return tuple(report for report in self.reports if report.site_id not in selected)


def _longest_consecutive_months(month_ids: np.ndarray) -> int:
    if not len(month_ids):
        return 0
    unique = np.unique(month_ids)
    longest = current = 1
    for previous, value in zip(unique[:-1], unique[1:], strict=True):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def audit_site_quality(
    site_id: str,
    index: pd.DatetimeIndex,
    *,
    observed_power_mask: pd.Series | np.ndarray,
    daylight_mask: pd.Series | np.ndarray,
    minimum_continuous_months: int = 24,
    minimum_daylight_completeness: float = 0.90,
) -> SiteQualityReport:
    """Evaluate the pre-registered DKASC inclusion rules for one device."""

    if len(index) == 0 or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Site index must be non-empty, sorted, and unique")
    observed = np.asarray(observed_power_mask, dtype=bool)
    daylight = np.asarray(daylight_mask, dtype=bool)
    if observed.shape != (len(index),) or daylight.shape != (len(index),):
        raise ValueError("Observed and daylight masks must have one value per timestamp")
    expected_count = int(daylight.sum())
    observed_count = int((observed & daylight).sum())
    completeness = observed_count / expected_count if expected_count else 0.0

    # Count a month as present only when it contains a measured daylight power
    # point; a fully missing month therefore breaks the continuous run.
    local_index = index.tz_localize(None) if index.tz is not None else index
    month_ids = local_index.year.to_numpy() * 12 + local_index.month.to_numpy()
    present_months = np.unique(month_ids[observed & daylight])
    continuous_months = _longest_consecutive_months(present_months)
    reasons: list[str] = []
    if continuous_months < minimum_continuous_months:
        reasons.append(
            f"continuous_months={continuous_months} < {minimum_continuous_months}"
        )
    if completeness < minimum_daylight_completeness:
        reasons.append(
            f"daylight_completeness={completeness:.4f} < "
            f"{minimum_daylight_completeness:.4f}"
        )
    return SiteQualityReport(
        site_id=str(site_id),
        continuous_months=continuous_months,
        daylight_completeness=float(completeness),
        observed_daylight_points=observed_count,
        expected_daylight_points=expected_count,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def select_eligible_sites(
    reports: Sequence[SiteQualityReport],
    *,
    maximum_sites: int = 12,
) -> tuple[SiteQualityReport, ...]:
    """Rank eligible devices by completeness with deterministic tie breaking."""

    if maximum_sites <= 0:
        raise ValueError("maximum_sites must be positive")
    eligible = (report for report in reports if report.eligible)
    ranked = sorted(
        eligible,
        key=lambda report: (
            -report.daylight_completeness,
            -report.continuous_months,
            report.site_id,
        ),
    )
    return tuple(ranked[:maximum_sites])


def select_dkasc_manifest_entries(
    entries: Sequence[ManifestEntry],
    data_root: str | Path,
    *,
    minimum_continuous_months: int = 24,
    minimum_daylight_completeness: float = 0.90,
    maximum_sites: int = 12,
    allow_approximate_solar: bool = False,
) -> DKASCManifestSelection:
    """Apply the pre-registered DKASC device inclusion rules before preparation.

    Daylight is derived only from site coordinates, timestamps, and clear-sky
    irradiance, so eligibility does not require fitting a power mapping or using
    held-out power values. The completeness numerator uses the preserved raw
    observation mask, not interpolated values.
    """

    # Local imports avoid making the lightweight quality dataclasses depend on
    # adapter/physics imports during package initialization.
    from racepv.physics import ClearSkyFeatureBuilder, SiteMetadata

    from .adapters import adapter_for

    candidates = tuple(entries)
    if not candidates:
        raise ValueError("No DKASC manifest entries were provided")
    if any(entry.dataset != "dkasc" for entry in candidates):
        wrong = sorted({entry.dataset for entry in candidates if entry.dataset != "dkasc"})
        raise ValueError(f"DKASC prefilter received other dataset kinds: {wrong}")
    site_ids = [entry.site_id for entry in candidates]
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("DKASC manifest site_id values must be unique")

    reports: list[SiteQualityReport] = []
    by_site = {entry.site_id: entry for entry in candidates}
    for entry in candidates:
        resampled = adapter_for(entry, data_root).load_resampled(entry)
        solar = ClearSkyFeatureBuilder(
            SiteMetadata(
                latitude=entry.latitude,
                longitude=entry.longitude,
                timezone=entry.timezone,
                altitude_m=0.0,
                name=entry.site_id,
            ),
            allow_approximate_fallback=allow_approximate_solar,
        ).build_solar_irradiance(resampled.values.index)
        report = audit_site_quality(
            entry.site_id,
            resampled.values.index,
            observed_power_mask=resampled.observed_mask["power"],
            daylight_mask=solar["is_solar_daylight"],
            minimum_continuous_months=minimum_continuous_months,
            minimum_daylight_completeness=minimum_daylight_completeness,
        )
        if entry.capacity_kw is None:
            report = SiteQualityReport(
                site_id=report.site_id,
                continuous_months=report.continuous_months,
                daylight_completeness=report.daylight_completeness,
                observed_daylight_points=report.observed_daylight_points,
                expected_daylight_points=report.expected_daylight_points,
                eligible=False,
                reasons=(*report.reasons, "rated capacity metadata is missing"),
            )
        reports.append(report)

    selected_reports = select_eligible_sites(reports, maximum_sites=maximum_sites)
    selected_entries = tuple(by_site[report.site_id] for report in selected_reports)
    return DKASCManifestSelection(entries=selected_entries, reports=tuple(reports))
