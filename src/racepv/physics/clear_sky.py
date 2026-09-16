"""Solar geometry, clear-sky irradiance, and leakage-safe PV power mapping."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd


class MissingPVLibError(ImportError):
    """Raised when production clear-sky features are requested without pvlib."""


class UnfittedClearSkyMapperError(RuntimeError):
    """Raised when system geometry is absent and no training mapping was fitted."""


@dataclass(frozen=True)
class SiteMetadata:
    latitude: float
    longitude: float
    timezone: str = "UTC"
    altitude_m: float = 0.0
    capacity_kw: float | None = None
    surface_tilt: float | None = None
    surface_azimuth: float | None = None
    gamma_pdc: float = -0.0035
    losses_fraction: float = 0.14
    name: str = "pv_site"

    def with_capacity(self, capacity_kw: float) -> SiteMetadata:
        if not np.isfinite(capacity_kw) or capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
        return replace(self, capacity_kw=float(capacity_kw))


@dataclass
class ClearSkyPowerMapper:
    """Non-negative site-scale mapping fitted on training data only.

    This is used when panel tilt/azimuth are unavailable.  It maps horizontal
    clear-sky GHI to power using an upper conditional ratio from training data,
    rather than averaging cloudy observations into the nominal clear-sky curve.
    Predictions are clipped to the same [0, 1.05] per-unit range as observed power.
    """

    scale_kw_per_wm2: float | None = field(default=None, init=False)
    mapping_quantile: float = 0.95
    capacity_kw: float | None = field(default=None, init=False)
    training_start: str | None = field(default=None, init=False)
    training_end: str | None = field(default=None, init=False)
    n_training_points: int = field(default=0, init=False)

    def fit(
        self,
        clear_sky_ghi: pd.Series,
        training_power_kw: pd.Series,
        *,
        capacity_kw: float,
        daylight_mask: pd.Series | np.ndarray | None = None,
        partition: str = "train",
    ) -> ClearSkyPowerMapper:
        if partition != "train":
            raise ValueError("Clear-sky mapping may only be fitted with partition='train'")
        if not clear_sky_ghi.index.equals(training_power_kw.index):
            raise ValueError("Irradiance and power indexes must match exactly")
        if not np.isfinite(capacity_kw) or capacity_kw <= 0:
            raise ValueError("capacity_kw must be positive")
        ghi = pd.to_numeric(clear_sky_ghi, errors="coerce")
        power = pd.to_numeric(training_power_kw, errors="coerce")
        if not 0.5 <= self.mapping_quantile < 1.0:
            raise ValueError("mapping_quantile must lie in [0.5, 1.0)")
        valid = np.isfinite(ghi) & np.isfinite(power) & (ghi > 50) & (power >= 0)
        if daylight_mask is not None:
            valid &= np.asarray(daylight_mask, dtype=bool)
        x = ghi.loc[valid].to_numpy(dtype=float)
        y = power.loc[valid].clip(lower=0, upper=1.05 * capacity_kw).to_numpy(dtype=float)
        if len(x) < 20:
            raise ValueError("At least 20 valid daylight training points are required")
        ratios = y / x
        scale = max(float(np.quantile(ratios, self.mapping_quantile)), 0.0)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Fitted clear-sky power scale is not positive")
        self.scale_kw_per_wm2 = scale
        self.capacity_kw = float(capacity_kw)
        self.n_training_points = int(len(x))
        if isinstance(ghi.index, pd.DatetimeIndex) and len(ghi):
            self.training_start = ghi.index.min().isoformat()
            self.training_end = ghi.index.max().isoformat()
        return self

    def predict(self, clear_sky_ghi: pd.Series) -> pd.Series:
        if self.scale_kw_per_wm2 is None or self.capacity_kw is None:
            raise UnfittedClearSkyMapperError("ClearSkyPowerMapper has not been fitted")
        power = pd.to_numeric(clear_sky_ghi, errors="coerce") * self.scale_kw_per_wm2
        return power.clip(lower=0, upper=1.05 * self.capacity_kw).rename(
            "clear_sky_power"
        )


def _load_pvlib() -> Any:
    try:
        import pvlib  # type: ignore[import-not-found]
    except ImportError as error:
        raise MissingPVLibError(
            "pvlib is required for production Ineichen/PVWatts features. Install "
            "the project's physics extra, or set allow_approximate_fallback=True "
            "only for synthetic tests; the fallback is not labeled Ineichen."
        ) from error
    return pvlib


def _localized_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return index.tz_convert(timezone)


def _approximate_solar_and_clear_sky(
    index: pd.DatetimeIndex, site: SiteMetadata
) -> pd.DataFrame:
    """NOAA-style solar position plus Haurwitz GHI for tests without pvlib.

    This fallback is intentionally marked in frame attributes and never described
    as Ineichen.  Production experiments should install pvlib.
    """

    local = _localized_index(index, site.timezone)
    day = local.dayofyear.to_numpy(dtype=float)
    hour = (
        local.hour.to_numpy(dtype=float)
        + local.minute.to_numpy(dtype=float) / 60.0
        + local.second.to_numpy(dtype=float) / 3600.0
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1 + (hour - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    utc_offsets = np.asarray(
        [timestamp.utcoffset().total_seconds() / 3600.0 for timestamp in local],
        dtype=float,
    )
    true_solar_minutes = (
        hour * 60.0 + equation_of_time + 4.0 * site.longitude - 60.0 * utc_offsets
    ) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude = np.deg2rad(site.latitude)
    cos_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.rad2deg(np.arccos(cos_zenith))
    elevation = 90.0 - zenith
    azimuth = np.rad2deg(
        np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(latitude)
            - np.tan(declination) * np.cos(latitude),
        )
    )
    azimuth = (azimuth + 180.0) % 360.0
    positive_cosine = np.clip(cos_zenith, 0.0, None)
    ghi = np.where(
        positive_cosine > 0,
        1098.0 * positive_cosine * np.exp(-0.059 / np.maximum(positive_cosine, 1e-6)),
        0.0,
    )
    # DNI/DHI are approximate placeholders derived from Haurwitz GHI; callers can
    # identify the model through attrs and should not use them for publication runs.
    dni = np.where(positive_cosine > 0.05, 0.85 * ghi / positive_cosine, 0.0)
    dhi = np.clip(ghi - dni * positive_cosine, 0.0, None)
    frame = pd.DataFrame(
        {
            "solar_elevation": elevation,
            "solar_zenith": zenith,
            "solar_azimuth": azimuth,
            "clear_sky_ghi": ghi,
            "clear_sky_dni": dni,
            "clear_sky_dhi": dhi,
        },
        index=index,
    )
    frame.attrs.update(
        solar_engine="noaa_approximation", clear_sky_model="haurwitz_approximation"
    )
    return frame


def _pvlib_solar_and_clear_sky(
    index: pd.DatetimeIndex, site: SiteMetadata
) -> pd.DataFrame:
    pvlib = _load_pvlib()
    localized = _localized_index(index, site.timezone)
    location = pvlib.location.Location(
        site.latitude,
        site.longitude,
        tz=site.timezone,
        altitude=site.altitude_m,
        name=site.name,
    )
    solar = location.get_solarposition(localized)
    clear = location.get_clearsky(localized, model="ineichen")
    result = pd.DataFrame(
        {
            "solar_elevation": solar["apparent_elevation"].to_numpy(dtype=float),
            "solar_zenith": solar["apparent_zenith"].to_numpy(dtype=float),
            "solar_azimuth": solar["azimuth"].to_numpy(dtype=float),
            "clear_sky_ghi": clear["ghi"].to_numpy(dtype=float),
            "clear_sky_dni": clear["dni"].to_numpy(dtype=float),
            "clear_sky_dhi": clear["dhi"].to_numpy(dtype=float),
        },
        index=index,
    )
    result.attrs.update(solar_engine="pvlib", clear_sky_model="ineichen")
    return result


def add_known_time_features(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    result = frame.copy()
    local = _localized_index(result.index, timezone)
    hour = local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0
    day = local.dayofyear.to_numpy(dtype=float)
    azimuth_radians = np.deg2rad(result["solar_azimuth"].to_numpy(dtype=float))
    result["solar_azimuth_sin"] = np.sin(azimuth_radians)
    result["solar_azimuth_cos"] = np.cos(azimuth_radians)
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["day_of_year_sin"] = np.sin(2 * np.pi * (day - 1) / 365.2425)
    result["day_of_year_cos"] = np.cos(2 * np.pi * (day - 1) / 365.2425)
    return result


class ClearSkyFeatureBuilder:
    """Build known-future physical features without future measured weather."""

    def __init__(
        self,
        site: SiteMetadata,
        *,
        mapper: ClearSkyPowerMapper | None = None,
        allow_approximate_fallback: bool = False,
    ) -> None:
        if not -90 <= site.latitude <= 90 or not -180 <= site.longitude <= 180:
            raise ValueError("Invalid site latitude/longitude")
        self.site = site
        self.mapper = mapper or ClearSkyPowerMapper()
        self.allow_approximate_fallback = allow_approximate_fallback

    def _irradiance(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        try:
            return _pvlib_solar_and_clear_sky(index, self.site)
        except MissingPVLibError:
            if not self.allow_approximate_fallback:
                raise
            return _approximate_solar_and_clear_sky(index, self.site)

    def fit_empirical_mapping(
        self,
        training_index: pd.DatetimeIndex,
        training_power_kw: pd.Series,
        *,
        daylight_mask: pd.Series | np.ndarray | None = None,
        partition: str = "train",
    ) -> ClearSkyFeatureBuilder:
        if self.site.capacity_kw is None:
            raise ValueError(
                "Set SiteMetadata.capacity_kw to the rated or training-q99.5 fallback capacity"
            )
        irradiance = self._irradiance(training_index)
        aligned_power = training_power_kw.reindex(training_index)
        if daylight_mask is None:
            daylight_mask = irradiance["clear_sky_ghi"] > 10.0
        self.mapper.fit(
            irradiance["clear_sky_ghi"],
            aligned_power,
            capacity_kw=self.site.capacity_kw,
            daylight_mask=daylight_mask,
            partition=partition,
        )
        return self

    def build_solar_irradiance(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Build solar geometry/Ineichen features without a PV power mapping.

        This is sufficient for daylight-completeness audits before a site is
        admitted to training. ``is_solar_daylight`` uses clear-sky GHI >= 10
        W/m², a mapping-free proxy for the later 1% clear-sky-power rule.
        """

        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("ClearSkyFeatureBuilder requires a DatetimeIndex")
        if index.hasnans:
            raise ValueError("Timestamps may not contain NaT")
        irradiance = self._irradiance(index)
        model_attrs = dict(irradiance.attrs)
        result = add_known_time_features(irradiance, self.site.timezone)
        result["is_solar_daylight"] = result["clear_sky_ghi"] >= 10.0
        result.attrs.update(model_attrs)
        return result

    def _pvwatts_power(self, irradiance: pd.DataFrame) -> pd.Series:
        if (
            self.site.capacity_kw is None
            or self.site.surface_tilt is None
            or self.site.surface_azimuth is None
        ):
            raise ValueError("PVWatts mapping requires capacity, surface tilt, and azimuth")
        pvlib = _load_pvlib()
        dni_extra = pvlib.irradiance.get_extra_radiation(irradiance.index)
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.site.surface_tilt,
            surface_azimuth=self.site.surface_azimuth,
            solar_zenith=irradiance["solar_zenith"],
            solar_azimuth=irradiance["solar_azimuth"],
            dni=irradiance["clear_sky_dni"],
            ghi=irradiance["clear_sky_ghi"],
            dhi=irradiance["clear_sky_dhi"],
            dni_extra=dni_extra,
            model="isotropic",
        )
        capacity_w = self.site.capacity_kw * 1000.0
        # Standard PVWatts reference conditions.  Future observed temperature is
        # intentionally not used; 25 C is a deterministic clear-sky reference.
        pdc = pvlib.pvsystem.pvwatts_dc(
            poa["poa_global"],
            temp_cell=25.0,
            pdc0=capacity_w,
            gamma_pdc=self.site.gamma_pdc,
        )
        pac = pvlib.inverter.pvwatts(pdc, pdc0=capacity_w)
        loss_multiplier = 1.0 - float(np.clip(self.site.losses_fraction, 0.0, 0.99))
        return pd.Series(
            np.asarray(pac, dtype=float) * loss_multiplier / 1000.0,
            index=irradiance.index,
            name="clear_sky_power",
        ).clip(lower=0.0, upper=1.05 * self.site.capacity_kw)

    def build(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("ClearSkyFeatureBuilder requires a DatetimeIndex")
        if index.hasnans:
            raise ValueError("Timestamps may not contain NaT")
        irradiance = self._irradiance(index)
        model_attrs = dict(irradiance.attrs)
        has_system_geometry = (
            self.site.capacity_kw is not None
            and self.site.surface_tilt is not None
            and self.site.surface_azimuth is not None
        )
        if has_system_geometry:
            try:
                power = self._pvwatts_power(irradiance)
                power_model = "pvwatts"
            except MissingPVLibError:
                # The analytic fallback never impersonates PVWatts. Prefer a
                # training-fitted mapping; for synthetic smoke tests only, a
                # plainly labeled capacity/GHI approximation keeps pvlib optional.
                if self.mapper.scale_kw_per_wm2 is not None:
                    power = self.mapper.predict(irradiance["clear_sky_ghi"])
                    power_model = "training_scale_mapping"
                else:
                    power = (
                        irradiance["clear_sky_ghi"] / 1000.0 * self.site.capacity_kw
                    ).clip(lower=0.0, upper=1.05 * self.site.capacity_kw)
                    power_model = "approximate_capacity_ghi_mapping"
        else:
            power = self.mapper.predict(irradiance["clear_sky_ghi"])
            power_model = "training_scale_mapping"

        if self.site.capacity_kw is None:
            raise ValueError("capacity_kw is required to normalize clear-sky power")
        result = irradiance.copy()
        result["clear_sky_power"] = power
        result["clear_sky_power_norm"] = (power / self.site.capacity_kw).clip(0.0, 1.05)
        result["is_daylight"] = result["clear_sky_power_norm"] >= 0.01
        result["is_night"] = ~result["is_daylight"]
        result = add_known_time_features(result, self.site.timezone)
        result.attrs.update(model_attrs)
        result.attrs["power_model"] = power_model
        return result


def night_mask(
    clear_sky_power_normalized: pd.Series | np.ndarray,
    threshold: float = 0.01,
) -> pd.Series | np.ndarray:
    if threshold < 0:
        raise ValueError("Night threshold must be non-negative")
    return clear_sky_power_normalized < threshold


def force_night_zero(
    power: pd.Series | pd.DataFrame | np.ndarray,
    clear_sky_power_normalized: pd.Series | np.ndarray,
    threshold: float = 0.01,
) -> pd.Series | pd.DataFrame | np.ndarray:
    """Return a copy with predictions/targets forced to zero at night."""

    mask = np.asarray(night_mask(clear_sky_power_normalized, threshold), dtype=bool)
    if isinstance(power, pd.Series):
        result = power.copy()
        result.loc[mask] = 0.0
        return result
    if isinstance(power, pd.DataFrame):
        result = power.copy()
        result.loc[mask, :] = 0.0
        return result
    result = np.array(power, copy=True)
    if result.shape[0] != len(mask):
        raise ValueError("Night mask length must match the first power dimension")
    result[mask] = 0.0
    return result
