"""Scalar normalization of caller-validated quantitative float32 arrays."""

from typing import Literal

import numpy as np

from albucore.stats import mean

__all__ = ["intensity_statistic", "normalize_by_scalar"]

_RMS_EINSUM_MIN_SIZE = 4_194_304


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
    # Avoid the image-sized float64 square only in the measured large/C-contiguous
    # region. The reshape is a view; strided inputs keep the square/mean path.
    if values.size >= _RMS_EINSUM_MIN_SIZE and values.flags.c_contiguous:
        flat = values.reshape(-1)
        return float(np.sqrt(np.einsum("i,i->", flat, flat, dtype=np.float64) / flat.size))
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
    result: np.ndarray
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if clamp_min is not None:
            result = np.maximum(image, np.float32(clamp_min))
            if offset != 0:
                np.subtract(result, np.float32(offset), out=result)
            np.divide(result, np.float32(divisor), out=result)
        elif offset != 0:
            result = np.subtract(image, np.float32(offset))
            np.divide(result, np.float32(divisor), out=result)
        else:
            result = np.divide(image, np.float32(divisor))
    if clip_range is not None:
        np.clip(result, *clip_range, out=result)
    if support is not None:
        np.copyto(result, np.float32(0), where=~support)
    return result
