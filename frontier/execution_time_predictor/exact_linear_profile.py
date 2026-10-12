"""Exact measured GEMM points, without interpolation or estimator/cache mutation.

The caller must first filter the CSV by model, TP and measurement family using
the ordinary loader. Summary statistics screen repeatability, not measurement
accuracy or fidelity to a complete model. Values retain the CSV's ms units.
"""
from collections import Counter
from statistics import median

import numpy as np
import pandas as pd


LINEAR_OPERATIONS = (
    "attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj",
)
MIN_SAMPLES = 20
MAX_RELATIVE_STD = 0.10
MAX_REPEAT_SPREAD = 0.10


def measured_linear_points(frame, operation):
    """Return accepted exact points and a complete rejection/quality audit.

    All records for a token shape must pass, so selecting just the quietest
    duplicate cannot hide conflicting or unstable measurements. Missing target
    values represent other op-specific rows and are not samples for this op.
    Duplicate *identical* records are not additional independent observations.
    Aggregation is the median of row medians, not a claimed pooled-sample median.
    """
    if operation not in LINEAR_OPERATIONS:
        raise ValueError(f"Unsupported exact-profile operation: {operation}")
    prefix = f"time_stats.{operation}."
    columns = [prefix + key for key in ("median", "mean", "std", "count")]
    audit = {"operation": operation, "min_samples": MIN_SAMPLES,
             "max_relative_std": MAX_RELATIVE_STD,
             "max_repeat_spread": MAX_REPEAT_SPREAD,
             "accepted_shapes": 0, "rejected": [], "points": []}
    missing = sorted(set(["num_tokens", *columns]) - set(frame.columns))
    if missing:
        audit["missing_columns"] = missing
        return {}, audit
    rows = frame.drop_duplicates()
    rows = rows.loc[rows[columns[0]].notna()].copy()
    for column in ["num_tokens", *columns]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    points = {}
    for tokens, group in rows.groupby("num_tokens", dropna=False, sort=True):
        key = int(tokens) if np.isfinite(tokens) and tokens == int(tokens) else None
        values = group[columns].to_numpy(dtype=float)
        reason = None
        if key is None or key <= 0:
            reason = "invalid_shape"
        elif not np.isfinite(values).all():
            reason = "nonfinite_statistics"
        elif (values[:, :2] <= 0).any() or (values[:, 2] < 0).any():
            reason = "invalid_statistics"
        elif (values[:, 3] < MIN_SAMPLES).any() or (values[:, 3] % 1 != 0).any():
            reason = "insufficient_samples"
        elif (values[:, 2] / values[:, 1] > MAX_RELATIVE_STD).any():
            reason = "unstable_samples"
        else:
            center = float(median(values[:, 0]))
            spread = float(np.ptp(values[:, 0]) / center)
            if spread > MAX_REPEAT_SPREAD:
                reason = "conflicting_repeats"
        if reason:
            audit["rejected"].append({"num_tokens": key, "reason": reason})
            continue
        points[(key,)] = center
        audit["points"].append({"num_tokens": key, "median_ms": center,
                                "summary_rows": len(group),
                                "min_count": int(values[:, 3].min()),
                                "max_relative_std": float((values[:, 2] / values[:, 1]).max()),
                                "repeat_median_spread": spread})
    audit["accepted_shapes"] = len(points)
    audit["rejected_by_reason"] = dict(Counter(row["reason"] for row in audit["rejected"]))
    return points, audit


def overlay_exact_points(predictions, points):
    """Copy a lookup table; replace original keys only, never clip OOD keys."""
    result = predictions.copy()
    changes = []
    for key, value in points.items():
        if key not in result:
            continue
        changes.append({"num_tokens": key[0], "rf_ms": float(result[key]),
                        "profile_ms": value, "delta_ms": value - float(result[key])})
        result[key] = value
    return result, changes
