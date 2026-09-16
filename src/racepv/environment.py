"""Runtime capability report embedded in every experiment artifact."""

from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import metadata
from typing import Any


def environment_report() -> dict[str, Any]:
    distributions = (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "pvlib",
        "lightgbm",
        "neuralforecast",
        "xlstm",
        "pytorch-forecasting",
        "mamba-ssm",
        "causal-conv1d",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "mamba_ssm_installed": importlib.util.find_spec("mamba_ssm") is not None,
        "pvlib_installed": importlib.util.find_spec("pvlib") is not None,
        "lightgbm_installed": importlib.util.find_spec("lightgbm") is not None,
        "package_versions": versions,
    }
    try:
        import torch
    except ImportError:
        report.update(torch_installed=False, cuda_available=False, mps_available=False)
        return report
    report.update(
        torch_installed=True,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        mps_available=bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    )
    return report
