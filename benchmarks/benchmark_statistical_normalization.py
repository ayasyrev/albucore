"""Compare scalar arithmetic/reduction candidates and optional linked AX calls."""

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

# Thread environment must be set before loading numerical runtimes.
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from albucore import intensity_statistic, normalize_by_scalar  # noqa: E402


def measure(function: Callable[[], Any]) -> dict[str, Any]:
    """Return repeated per-call timings after warmup, including all allocations."""
    for _ in range(3):
        function()
    start = time.perf_counter()
    function()
    elapsed = time.perf_counter() - start
    count = 100 if elapsed < 0.005 else max(5, int(0.5 / elapsed))
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(count):
            function()
        samples.append((time.perf_counter() - start) * 1000 / count)
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "iterations": count}


def torch_apply(array: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(array).clone()
    value.clamp_min_(0).sub_(0.25)
    if not torch.isfinite(value).all():
        raise FloatingPointError
    value.div_(3)
    if not torch.isfinite(value).all():
        raise FloatingPointError
    return np.asarray(value.numpy())


def cv_apply(array: np.ndarray) -> np.ndarray:
    value = np.maximum(array.reshape(-1), np.float32(0)).reshape(-1, 1)
    cv2.subtract(value, 0.25, dst=value)
    if not np.isfinite(value).all():
        raise FloatingPointError
    cv2.divide(value, 3.0, dst=value)
    if not np.isfinite(value).all():
        raise FloatingPointError
    return value.reshape(array.shape)


def compose_timings(array: np.ndarray) -> dict[str, Any]:
    ax = importlib.import_module("albumentations")
    target = "image" if array.ndim == 3 else "volume"
    tensor = torch.from_numpy(array).movedim(-1, 0)
    timings = {}
    for mode, kwargs in (("supplied", {"divisor": 3}), ("self-rms", {"statistic": "rms"})):
        compose = ax.Compose([ax.NormalizeByStatistic(**kwargs, clamp_min=0)])
        for backend, value in (
            ("numpy", array), ("tensor-view", tensor), ("tensor-contiguous", tensor.contiguous()),
        ):
            timings[f"compose/{mode}/{backend}"] = measure(
                lambda compose=compose, value=value: compose(**{target: value}),
            )
    return timings


def benchmark_shape(array: np.ndarray, with_ax: bool) -> dict[str, Any]:
    flat = array.reshape(-1)
    candidates = {
        "apply_numpy": lambda: normalize_by_scalar(array, (0.25, 3), clamp_min=0),
        "apply_torch_bridge": lambda: torch_apply(array),
        "apply_opencv_flat": lambda: cv_apply(array),
        "rms_numpy_square": lambda: intensity_statistic(array, "rms"),
        "rms_numpy_einsum": lambda: math.sqrt(float(np.einsum("i,i->", flat, flat, dtype=np.float64)) / flat.size),
        "rms_opencv_norm": lambda: cv2.norm(flat, cv2.NORM_L2) / math.sqrt(flat.size),
        "rms_torch_bridge": lambda: float(torch.from_numpy(array).double().square().mean().sqrt()),
    }
    reference = candidates["apply_numpy"]()
    scalar = candidates["rms_numpy_square"]()
    row: dict[str, Any] = {
        "shape": array.shape, "dtype": "float32", "timings": {}, "max_abs": {}, "eligible": {},
    }
    for name, function in candidates.items():
        actual = function()
        expected = reference if name.startswith("apply") else scalar
        row["max_abs"][name] = float(np.max(np.abs(actual - expected)))
        row["eligible"][name] = bool(
            np.array_equal(actual, expected) if name.startswith("apply")
            else np.allclose(actual, expected, rtol=1e-14, atol=0),
        )
        row["timings"][name] = measure(function)
    if with_ax:
        row["timings"].update(compose_timings(array))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-ax", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    shapes = [(size, size, channels) for size in (256, 512, 1024) for channels in (1, 3, 5)]
    shapes += [
        (height, width, channels)
        for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)) for channels in (1, 3, 9)
    ]
    shapes += [
        (16, 128, 160, 1), (16, 128, 160, 3), (32, 128, 160, 1), (32, 128, 160, 3),
        (64, 128, 160, 3), (96, 128, 160, 1), (48, 240, 320, 3), (138, 138, 138, 1),
    ]
    rng = np.random.default_rng(137)
    rows = []
    for shape in shapes:
        rows.append(benchmark_shape(rng.uniform(-1, 10, shape).astype(np.float32), args.with_ax))
        sys.stdout.write(f"completed {shape}\n")
        sys.stdout.flush()
    args.output.write_text(json.dumps({
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "opencv": cv2.__version__},
        "threads": {"torch": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                    "opencv": cv2.getNumThreads()},
        "comparison": "new API: equivalent arithmetic candidates, not a legacy Normalize regression claim",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
