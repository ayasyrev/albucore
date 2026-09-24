"""Scalar normalization of caller-validated quantitative float32 arrays."""

from typing import Literal

import numpy as np

from albucore.stats import mean

__all__ = ["intensity_statistic", "normalize_by_scalar"]


def intensity_statistic(
    image: np.ndarray,
    statistic: Literal["mean", "rms"],
    selection: np.ndarray | None = None,
) -> float:
    """Reduce one float32 HWC/DHWC array globally with float64 accumulation.

    The caller validates layout, nonempty selection and finite selected values.
    Selection is a spatial boolean array; all channels at each selected position
    participate. Zero RMS and nonpositive mean are valid mathematical results.
    No epsilon, centering, clipping or automatic positive selection is applied.
    """
    values = image if selection is None else image[selection]
    if statistic == "mean":
        return float(mean(values, axis="global"))
    return float(np.sqrt(np.square(values, dtype=np.float64).mean(dtype=np.float64)))


def normalize_by_scalar(
    image: np.ndarray,
    coefficients: tuple[float, float],
    *,
    clamp_min: float | None = None,
    clip_range: tuple[float, float] | None = None,
    support: np.ndarray | None = None,
) -> np.ndarray:
    """Apply scalar normalization to one prevalidated float32 HWC/DHWC array.

    Coefficients are ``(offset, divisor)``. The caller supplies finite input and float32-representable controls, a
    positive divisor, and optional broadcastable boolean support. Arithmetic
    order is clamp, subtract, direct divide, clip, then support zeroing.
    Overflow is rejected before clipping or masking can conceal it. Output owns
    independent float32 storage, including identity operations and strided input.
    """
    offset, divisor = coefficients
    result = image.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if clamp_min is not None:
            np.maximum(result, np.float32(clamp_min), out=result)
        if offset != 0:
            np.subtract(result, np.float32(offset), out=result)
        np.divide(result, np.float32(divisor), out=result)
    if clip_range is not None:
        np.clip(result, *clip_range, out=result)
    if support is not None:
        np.copyto(result, np.float32(0), where=~support)
    return result
