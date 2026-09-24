"""Prevalidated scalar kernels, checked against independent float64 expressions."""

import math

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


@pytest.mark.parametrize("shape", [(1, 5, 1), (1, 1, 5, 5)])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "fortran"])
@pytest.mark.parametrize(
    ("clamp", "offset", "expected_values"),
    [
        (None, 0, [-2, -0.125, 0, 0.5, 3.5]),
        (None, 1, [-2.5, -0.625, -0.5, 0, 3]),
        (0, 0, [0, 0, 0, 0.5, 3.5]),
        (0, 1, [-0.5, -0.5, -0.5, 0, 3]),
    ],
)
def test_first_operation_owns_readonly_output(
    shape: tuple[int, ...], layout: str, clamp: float | None, offset: float, expected_values: list[float],
) -> None:
    source = np.broadcast_to(np.array([-4, -0.25, 0, 1, 7], np.float32)[:, None], shape).copy()
    expected = np.broadcast_to(np.array(expected_values, np.float32)[:, None], shape).copy()
    if layout == "reversed":
        source = source[..., ::-1, :]
        expected = expected[..., ::-1, :]
    elif layout == "fortran":
        source = np.asfortranarray(source)
    before = source.copy()
    source.flags.writeable = False
    result = normalize_by_scalar(source, (offset, 2), clamp_min=clamp)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_equal(result.shape, source.shape)
    np.testing.assert_equal(result.dtype, np.dtype("float32"))
    np.testing.assert_equal(np.shares_memory(result, source), False)
    support = np.ones(shape[:-1] + (1,), bool)
    support[..., 1::2, :] = False
    clipped = normalize_by_scalar(source, (offset, 2), clamp_min=clamp, clip_range=(-0.25, 2), support=support)
    expected = np.clip(expected, -0.25, 2)
    expected[..., 1::2, :] = 0
    np.testing.assert_array_equal(clipped, expected)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("clamp", [None, 0])
@pytest.mark.parametrize("coefficients", [(0, 0.5), (-float(np.finfo(np.float32).max), 1)])
def test_first_operation_overflow_does_not_mutate_source(
    clamp: float | None, coefficients: tuple[float, float],
) -> None:
    source = np.full((1, 2, 1), np.finfo(np.float32).max, np.float32)[:, ::-1]
    before = source.copy()
    with pytest.raises(FloatingPointError):
        normalize_by_scalar(source, coefficients, clamp_min=clamp, clip_range=(0, 1), support=np.zeros_like(source, bool))
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("size", [2**22 - 1, 2**22, 2**22 + 1])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "selected"])
def test_large_rms_boundary_and_selection(size: int, layout: str) -> None:
    source = np.resize(np.array([-3, 4], np.float32), size).reshape(1, size, 1)
    selection = None
    if layout == "reversed":
        source = source[:, ::-1]
    elif layout == "selected":
        padded = np.full((1, 2 * size, 1), np.nan, np.float32)
        padded[:, ::2] = source
        source = padded
        selection = np.zeros(source.shape[:-1], bool)
        selection[:, ::2] = True
    source.flags.writeable = False
    expected = math.sqrt((9 * ((size + 1) // 2) + 16 * (size // 2)) / size)
    np.testing.assert_allclose(intensity_statistic(source, "rms", selection), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("value", [0, np.finfo(np.float32).smallest_subnormal, np.finfo(np.float32).max])
def test_large_rms_float64_range(value: float) -> None:
    source = np.full((1024, 1025, 5), value, np.float32)
    np.testing.assert_allclose(intensity_statistic(source, "rms"), float(value), rtol=1e-14, atol=0)
