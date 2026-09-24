"""Prevalidated scalar kernels, checked against independent float64 expressions."""

import numpy as np
import pytest

from albucore import intensity_statistic, normalize_by_scalar


@pytest.mark.parametrize("shape", [(2, 7, 1), (2, 7, 3), (2, 7, 5), (2, 3, 7, 1), (2, 3, 7, 5)])
def test_reduction_and_owned_application(shape: tuple[int, ...]) -> None:
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)[:, ::-1]
    before = source.copy()
    selection = np.zeros(shape[:-1], bool)
    selection[0] = True
    values = [float(value) for value in source[selection].ravel()]
    np.testing.assert_allclose(intensity_statistic(source, "mean", selection), sum(values) / len(values), rtol=1e-14)
    np.testing.assert_allclose(
        intensity_statistic(source, "rms", selection), (sum(v * v for v in values) / len(values)) ** 0.5, rtol=1e-14,
    )
    result = normalize_by_scalar(source, (4, 2), clamp_min=1, clip_range=(-1, 3), support=selection[..., None])
    expected = np.clip((np.maximum(source.astype(np.float64), 1) - 4) / 2, -1, 3)
    expected[~selection] = 0
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_equal(result.dtype, np.dtype("float32"))
    np.testing.assert_equal(np.shares_memory(result, source), False)


def test_identity_owns_storage_and_degenerate_statistics() -> None:
    source = np.zeros((1, 2, 1), np.float32)
    np.testing.assert_equal(intensity_statistic(source, "rms"), 0)
    np.testing.assert_equal(intensity_statistic(source - 1, "mean"), -1)
    np.testing.assert_equal(np.shares_memory(normalize_by_scalar(source, (0, 1)), source), False)


def test_overflow_is_detected_before_clip_and_support() -> None:
    source = np.full((1, 1, 1), np.finfo(np.float32).max, np.float32)
    with pytest.raises(FloatingPointError):
        normalize_by_scalar(source, (0, 0.5), clip_range=(0, 1), support=np.zeros_like(source, bool))
