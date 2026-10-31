"""SHAP-based reason codes for the LightGBM direction model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _positive_class_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    array = np.asarray(values)
    if array.ndim == 3:
        # (rows, features, classes)
        array = array[0, :, 1] if array.shape[-1] > 1 else array[0, :, 0]
    elif array.ndim == 2:
        array = array[0]
    return np.asarray(array, dtype=float).reshape(-1)


def top_reasons(
    pipeline: Any,
    row: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Largest absolute SHAP contributions toward an up session."""
    import shap

    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = _positive_class_values(explainer.shap_values(row[feature_names]))
    order = np.argsort(np.abs(values))[::-1][:top_k]
    reasons = []
    for index in order:
        name = feature_names[index]
        contribution = float(values[index])
        reasons.append(
            {
                "feature": name,
                "value": round(float(row[name].iloc[0]), 6),
                "shap": round(contribution, 6),
                "effect": "raises_p_up" if contribution >= 0 else "lowers_p_up",
            }
        )
    return reasons
