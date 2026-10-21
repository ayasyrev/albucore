"""Regression coverage for uint8 moments overflow (Albucore #196)."""

import numpy as np
import pytest
import torch

import albucore as ac
import albucore.stats as stats_mod
from albucore.normalize import normalize_per_image_lut, normalize_per_image_opencv
from albucore.utils import get_opencv_max_channels


@pytest.fixture
def forbid_uint8_moments(monkeypatch):
    original = stats_mod.nk.moments

    def checked_moments(array, *args, **kwargs):
        if array.dtype == np.uint8:
            raise AssertionError("uint8 statistics must not call the overflowing NumKong moments kernel")
        return original(array, *args, **kwargs)

    monkeypatch.setattr(stats_mod.nk, "moments", checked_moments)


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 9])
@pytest.mark.parametrize("pattern", ["bright", "constant"])
def test_large_uint8_statistics(channels, pattern):
    image = np.full((600, 900, channels), 255, dtype=np.uint8)
    if pattern == "bright":
        image[:, ::2] = 240
    before = image.copy()
    expected_mean = 247.5 if pattern == "bright" else 255.0
    expected_std = 7.5 if pattern == "bright" else 0.0

    for axis in (None, "per_channel"):
        result_mean, result_std = ac.mean_std(image, axis, eps=0)
        np.testing.assert_allclose(result_mean, expected_mean, rtol=0, atol=1e-10)
        np.testing.assert_allclose(result_std, expected_std, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ac.std(image, axis, eps=0), expected_std, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 23, 3), (17, 23, 5), (2, 17, 23, 9), (2, 3, 17, 23, 4), (391, 5)])
@pytest.mark.parametrize("storage", ["c", "fortran", "planar", "sliced", "reversed", "readonly"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_uint8_statistics_layout_and_axes(shape, storage, keepdims, forbid_uint8_moments):
    image = np.random.default_rng(196).integers(0, 256, shape, dtype=np.uint8)
    if storage == "fortran":
        image = np.asfortranarray(image)
    elif storage == "planar":
        image = np.moveaxis(np.ascontiguousarray(np.moveaxis(image, -1, 0)), 0, -1)
    elif storage == "sliced":
        backing = np.empty((*shape[:-1], shape[-1] * 2), dtype=np.uint8)
        backing[..., ::2] = image
        image = backing[..., ::2]
    elif storage == "reversed":
        image = image[::-1, ..., ::-1]
    elif storage == "readonly":
        image.flags.writeable = False
    before = image.copy()
    spatial = tuple(range(image.ndim - 1))
    negative_spatial = tuple(range(-image.ndim, -1))
    for axis in (None, "global", "per_channel", spatial, negative_spatial, tuple(range(image.ndim)), -1):
        axes = None if axis is None or axis == "global" else spatial if axis == "per_channel" else axis
        expected_mean = image.mean(axis=axes, dtype=np.float64, keepdims=keepdims)
        expected_std = image.std(axis=axes, dtype=np.float64, keepdims=keepdims) + 0.25
        result_mean, result_std = ac.mean_std(image, axis, keepdims=keepdims, eps=0.25)
        np.testing.assert_allclose(result_mean, expected_mean, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(result_std, expected_std, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(ac.std(image, axis, keepdims=keepdims, eps=0.25), expected_std, rtol=1e-12)
        assert np.shape(result_mean) == np.shape(expected_mean)
        assert np.shape(result_std) == np.shape(expected_std)
        if isinstance(result_std, np.ndarray):
            assert result_std.dtype == np.float64
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("channels", [get_opencv_max_channels(), get_opencv_max_channels() + 1])
def test_uint8_statistics_opencv_channel_boundary(channels, forbid_uint8_moments):
    image = np.random.default_rng(196).integers(240, 256, (7, 11, channels), dtype=np.uint8)
    result_mean, result_std = ac.mean_std(image, "per_channel", eps=0)
    np.testing.assert_allclose(result_mean, image.mean(axis=(0, 1), dtype=np.float64), rtol=1e-12)
    np.testing.assert_allclose(result_std, image.std(axis=(0, 1), dtype=np.float64), rtol=1e-10)
    np.testing.assert_allclose(ac.std(image, "per_channel", eps=0), result_std, rtol=1e-12)


@pytest.mark.parametrize("normalization", ["image", "image_per_channel"])
@pytest.mark.parametrize("function", [ac.normalize_per_image, normalize_per_image_lut, normalize_per_image_opencv])
def test_large_bright_uint8_normalization(function, normalization):
    image = np.random.default_rng(137).integers(240, 256, (600, 500, 4), dtype=np.uint8)
    before = image.copy()
    axis = None if normalization == "image" else (0, 1)
    expected = (image.astype(np.float64) - image.mean(axis=axis, dtype=np.float64)) / (
        image.std(axis=axis, dtype=np.float64) + stats_mod.DEFAULT_EPS
    )
    result = function(image, normalization)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected.clip(-20, 20), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize(
    ("layout", "shape", "channel_axis"),
    [
        ("CHW", (4, 600, 500), 0),
        ("NCHW", (2, 4, 300, 500), 1),
        ("CDHW", (4, 2, 300, 500), 0),
        ("NCDHW", (2, 4, 2, 150, 500), 1),
    ],
)
@pytest.mark.parametrize("planar", [False, True])
def test_uint8_tensor_statistics(layout, shape, channel_axis, planar, forbid_uint8_moments):
    array = np.random.default_rng(196).integers(240, 256, shape, dtype=np.uint8)
    if not planar:
        array = np.moveaxis(np.ascontiguousarray(np.moveaxis(array, channel_axis, -1)), -1, channel_axis)
    image = torch.from_numpy(array)
    before = image.clone()
    for axis in (None, "per_channel"):
        axes = None if axis is None else tuple(i for i in range(array.ndim) if i != channel_axis)
        for keepdims in (False, True):
            expected_mean = array.mean(axis=axes, dtype=np.float64, keepdims=keepdims)
            expected_std = array.std(axis=axes, dtype=np.float64, keepdims=keepdims)
            result_mean, result_std = ac.mean_std(image, axis, keepdims=keepdims, eps=0, layout=layout)
            assert result_mean.dtype == result_std.dtype == torch.float64
            np.testing.assert_allclose(result_mean.numpy(), expected_mean, rtol=1e-12)
            np.testing.assert_allclose(result_std.numpy(), expected_std, rtol=1e-9)
            result = ac.std(image, axis, keepdims=keepdims, eps=0, layout=layout)
            np.testing.assert_allclose(result.numpy(), expected_std, rtol=1e-9)
    for normalization in ("image", "image_per_channel"):
        axes = tuple(
            i
            for i in range(array.ndim)
            if not (layout.startswith("N") and i == 0)
            and not (normalization == "image_per_channel" and i == channel_axis)
        )
        expected = (array.astype(np.float64) - array.mean(axis=axes, dtype=np.float64, keepdims=True)) / (
            array.std(axis=axes, dtype=np.float64, keepdims=True) + stats_mod.DEFAULT_EPS
        )
        result = ac.normalize_per_image(image, normalization, layout=layout)
        assert result.shape == image.shape
        assert result.dtype == torch.float32
        np.testing.assert_allclose(result.numpy(), expected.clip(-20, 20), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(image, before, rtol=0, atol=0)
