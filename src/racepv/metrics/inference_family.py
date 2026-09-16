import math
from numbers import Real

import numpy as np

YULARA_QCRPS_FAMILY_V1 = tuple(
    (model, site)
    for model in ("timemixerpp_qr", "timer_xl_qr", "rqr_tft")
    for site in (
        "yulara_01_desert_gardens",
        "yulara_02_service_station",
        "yulara_03_sails_combined",
        "yulara_04_laundry",
        "yulara_05_connellan_airport",
    )
)


def _validated_p_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("invalid p_value")

    if isinstance(value, np.generic):
        if not isinstance(value, (np.integer, np.floating)):
            raise ValueError("invalid p_value")
        if value < 0 or value > 1 or not np.isfinite(value):
            raise ValueError("invalid p_value")
    else:
        if not isinstance(value, Real):
            raise ValueError("invalid p_value")
        if value < 0 or value > 1 or not math.isfinite(value):
            raise ValueError("invalid p_value")

    return float(value)


def holm_adjust_complete_family(rows, *, expected_keys):
    try:
        keys = tuple(expected_keys)
    except TypeError as error:
        raise ValueError("invalid expected family keys") from error
    if not keys or any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(value, str) or not value for value in key)
        for key in keys
    ):
        raise ValueError("invalid expected family keys")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate expected family keys")

    observed = {}
    try:
        iterator = iter(rows)
    except TypeError as error:
        raise ValueError("invalid family row schema") from error
    for row in iterator:
        if not isinstance(row, dict) or set(row) != {
            "comparison",
            "site_id",
            "p_value",
        }:
            raise ValueError("invalid family row schema")
        if any(
            not isinstance(row[field], str) or not row[field] for field in ("comparison", "site_id")
        ):
            raise ValueError("invalid family row key")
        key = (row["comparison"], row["site_id"])
        if key in observed:
            raise ValueError("duplicate family row")
        observed[key] = _validated_p_value(row["p_value"])

    if set(observed) != set(keys):
        raise ValueError("family is incomplete or contains unregistered rows")

    probabilities = [1.0 if observed[key] is None else observed[key] for key in keys]
    order = sorted(range(len(keys)), key=lambda index: (probabilities[index], index))
    adjusted = [None] * len(keys)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(
            running,
            min(1.0, (len(keys) - rank) * probabilities[index]),
        )
        if observed[keys[index]] is not None:
            adjusted[index] = running

    return [
        {
            "comparison": key[0],
            "site_id": key[1],
            "p_value": observed[key],
            "adjusted_p_value": adjusted[index],
            "family_size": len(keys),
        }
        for index, key in enumerate(keys)
    ]
