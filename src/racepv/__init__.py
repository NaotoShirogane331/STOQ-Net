"""RaCE-PV: ramp-aware probabilistic photovoltaic forecasting."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("race-pv")
except PackageNotFoundError:  # pragma: no cover - source tree import
    __version__ = "0.1.0"

__all__ = ["__version__"]

