from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FeatureBounds = Dict[str, Tuple[float, float]]


def compute_feature_bounds(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    context: str,
) -> FeatureBounds:
    """Return the finite per-feature envelope represented by profiling rows."""
    bounds: FeatureBounds = {}
    for feature_name in feature_cols:
        if feature_name not in df.columns:
            raise ValueError(
                f"Cannot determine profiling domain for {context}: "
                f"missing feature column {feature_name!r}."
            )

        try:
            values = np.asarray(df[feature_name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot determine profiling domain for {context}: feature "
                f"{feature_name!r} is not numeric."
            ) from exc

        if values.size == 0:
            raise ValueError(
                f"Cannot determine profiling domain for {context}: feature "
                f"{feature_name!r} has no rows."
            )
        if not bool(np.isfinite(values).all()):
            raise ValueError(
                f"Cannot determine profiling domain for {context}: feature "
                f"{feature_name!r} contains non-finite values."
            )

        bounds[feature_name] = (float(values.min()), float(values.max()))

    return bounds
