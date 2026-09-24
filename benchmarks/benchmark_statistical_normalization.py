"""Compare scalar arithmetic/reduction candidates and optional linked AX calls."""

import argparse
import importlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
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
import numkong as nk  # noqa: E402
import torch  # noqa: E402

import albucore  # noqa: E402
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


def torch_apply(
    array: np.ndarray, coefficients: tuple[float, float] = (0.25, 3), *, clamp_min: float | None = 0, **kwargs: Any,
) -> np.ndarray:
    # Materialize exactly once, including negative-stride repair, inside timing.
    value = torch.from_numpy(array.copy())
    if clamp_min is not None:
        value.clamp_min_(clamp_min)
    offset, divisor = coefficients
    if offset != 0:
        value.sub_(offset)
    if not torch.isfinite(value).all():
        raise FloatingPointError
    value.div_(divisor)
    if not torch.isfinite(value).all():
        raise FloatingPointError
    return finish_application(np.asarray(value.numpy()), **kwargs)


def cv_apply(
    array: np.ndarray, coefficients: tuple[float, float] = (0.25, 3), *, clamp_min: float | None = 0, **kwargs: Any,
) -> np.ndarray:
    value = array.copy() if clamp_min is None else np.maximum(array, np.float32(clamp_min))
    value = value.reshape(-1, 1)
    offset, divisor = coefficients
    if offset != 0:
        cv2.subtract(value, offset, dst=value)
    if not np.isfinite(value).all():
        raise FloatingPointError
    cv2.divide(value, float(divisor), dst=value)
    if not np.isfinite(value).all():
        raise FloatingPointError
    return finish_application(value.reshape(array.shape), **kwargs)


def finish_application(
    value: np.ndarray, *, clip_range: tuple[float, float] | None = None, support: np.ndarray | None = None,
) -> np.ndarray:
    if clip_range is not None:
        np.clip(value, *clip_range, out=value)
    if support is not None:
        np.copyto(value, np.float32(0), where=~support)
    return value


def rms_candidate(array: np.ndarray, selection: np.ndarray | None, backend: str) -> float:
    """Include selection, flattening, dtype conversion and bridges in each call."""
    values = array if selection is None else array[selection]
    if backend == "square":
        return float(np.sqrt(np.square(values, dtype=np.float64).mean(dtype=np.float64)))
    if backend == "einsum":
        flat = values.reshape(-1)
        return math.sqrt(float(np.einsum("i,i->", flat, flat, dtype=np.float64)) / flat.size)
    if backend == "opencv":
        return cv2.norm(values.reshape(-1), cv2.NORM_L2) / math.sqrt(values.size)
    if backend == "torch":
        return float(torch.from_numpy(np.ascontiguousarray(values)).double().square().mean().sqrt())
    if backend == "numkong_norm":
        return float(nk.norm(values)) / math.sqrt(values.size)
    if backend == "numkong_moments":
        return math.sqrt(nk.moments(values)[1] / values.size)
    flat = values.reshape(-1)
    if backend == "numkong_dot64":
        flat = flat.astype(np.float64)
    return math.sqrt(float(nk.dot(flat, flat)) / flat.size)


def measure_candidates(candidates: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """Alternate candidate order, using identical warmups and fixed sample counts."""
    for function in candidates.values():
        for _ in range(3):
            function()
    names = list(candidates)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(8):
        for name in names if repeat % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            for _ in range(12):
                candidates[name]()
            samples[name].append((time.perf_counter() - start) * 1000 / 12)
    return {
        name: {"median_ms": statistics.median(times), "samples_ms": times, "iterations": 12}
        for name, times in samples.items()
    }


def allocation_peak(function: Callable[[], Any]) -> int:
    """Trace NumPy/Python peak allocation separately from timing; this is not RSS."""
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


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


def benchmark_shape(
    array: np.ndarray, with_ax: bool, *, profile: str = "plain", baseline: Any = None, public_only: bool = False,
) -> dict[str, Any]:
    selection = None
    coefficients = (0.25, 3)
    controls: dict[str, Any] = {"clamp_min": 0}
    if profile in {"unclamped", "divide-only", "identity"}:
        controls["clamp_min"] = None
        if profile != "unclamped":
            coefficients = (0, 1 if profile == "identity" else 3)
    elif profile in {"selected-sparse", "selected-dense"}:
        # Selection construction is caller work; gathering remains inside timing.
        selection = np.zeros(array.shape[:-1], dtype=bool)
        selection[..., ::8 if profile == "selected-sparse" else 2] = True
        controls.update(clip_range=(-1, 2), support=selection[..., None])
    candidates = {
        "apply_numpy": lambda: normalize_by_scalar(array, coefficients, **controls),
        "apply_torch_bridge": lambda: torch_apply(array, coefficients, **controls),
        "apply_opencv_flat": lambda: cv_apply(array, coefficients, **controls),
        "rms_public": lambda: intensity_statistic(array, "rms", selection),
        "mean_public": lambda: intensity_statistic(array, "mean", selection),
    }
    for backend in ("square", "einsum", "opencv", "torch", "numkong_norm", "numkong_moments", "numkong_dot", "numkong_dot64"):
        candidates[f"rms_{backend}"] = lambda backend=backend: rms_candidate(array, selection, backend)
    if baseline is not None:
        candidates["apply_baseline"] = lambda: baseline.normalize_by_scalar(array, coefficients, **controls)
        candidates["rms_baseline"] = lambda: baseline.intensity_statistic(array, "rms", selection)
        candidates["mean_baseline"] = lambda: baseline.intensity_statistic(array, "mean", selection)
    reference = candidates["apply_numpy"]()
    scalar = candidates["rms_square"]()
    selected = array if selection is None else array[selection]
    expected_mean = float(selected.mean(dtype=np.float64))
    if public_only:
        candidates = {
            name: function for name, function in candidates.items()
            if name in {"apply_numpy", "rms_public", "mean_public", "apply_baseline", "rms_baseline", "mean_baseline"}
        }
    row: dict[str, Any] = {
        "shape": array.shape, "dtype": "float32", "strides": array.strides,
        "contiguous": bool(array.flags.c_contiguous), "profile": profile,
        "selection_count": None if selection is None else int(selection.sum()),
        "coefficients": coefficients, "clamp_min": controls["clamp_min"],
        "clip_range": controls.get("clip_range"), "support": selection is not None,
        "timings": {}, "max_abs": {}, "eligible": {},
    }
    for name, function in candidates.items():
        actual = function()
        expected = reference if name.startswith("apply") else expected_mean if name.startswith("mean") else scalar
        row["max_abs"][name] = float(np.max(np.abs(actual - expected)))
        row["eligible"][name] = bool(
            np.array_equal(actual, expected) if name.startswith("apply")
            else np.allclose(actual, expected, rtol=1e-14, atol=0),
        )
    row["timings"] = measure_candidates(candidates)
    row["numpy_peak_bytes"] = {
        name: allocation_peak(function) for name, function in candidates.items()
        if name in {"apply_numpy", "apply_baseline", "rms_public", "rms_baseline", "rms_square", "rms_einsum"}
    }
    if with_ax:
        row["timings"].update(compose_timings(array))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-ax", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--public-only", action="store_true", help="Pair current/baseline after the backend audit")
    parser.add_argument("--baseline-source", type=Path, help="Pre-change statistical_normalization.py exported from Git")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    baseline = None
    if args.baseline_source is not None:
        spec = importlib.util.spec_from_file_location("normalization_baseline", args.baseline_source)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    shapes = [
        (height, width, channels)
        for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)) for channels in (1, 3, 9)
    ]
    shapes += [
        (16, 128, 160, 1), (16, 128, 160, 3), (32, 128, 160, 1), (32, 128, 160, 3),
        (64, 128, 160, 3), (96, 128, 160, 1), (48, 240, 320, 3),
    ]
    # Size sweep around the cache-sensitive square/einsum crossover.
    shapes += [(1, size, 1) for size in (2**21, 3 * 2**20, 2**22 - 1, 2**22, 2**22 + 1, 5 * 2**20)]
    # Retain the original image grid and raw-volume shape for historical comparisons.
    shapes += [(size, size, channels) for size in (256, 512, 1024) for channels in (1, 3, 5)]
    shapes.append((138, 138, 138, 1))
    extras = {(240, 320, 3), (768, 1024, 9), (48, 240, 320, 3)}
    if args.quick:
        shapes = [(240, 320, 3), (768, 1024, 9), (48, 240, 320, 3)]
    rng = np.random.default_rng(137)
    rows = []
    for shape in shapes:
        source = rng.uniform(-1, 10, shape).astype(np.float32)
        options = {"baseline": baseline, "public_only": args.public_only}
        rows.append(benchmark_shape(source, args.with_ax, **options))
        if shape in extras:
            for profile in ("unclamped", "divide-only", "identity"):
                rows.append(benchmark_shape(source, False, profile=profile, **options))
            for array in (source, source[..., ::-1, :]):
                for profile in ("selected-sparse", "selected-dense"):
                    rows.append(benchmark_shape(array, False, profile=profile, **options))
            rows.append(benchmark_shape(source[..., ::-1, :], False, **options))
        sys.stdout.write(f"completed {shape}\n")
        sys.stdout.flush()
    args.output.write_text(json.dumps({
        "versions": {"numpy": np.__version__, "torch": torch.__version__, "opencv": cv2.__version__, "numkong": nk.__version__},
        "import": albucore.__file__, "baseline_source": None if baseline is None else baseline.__file__,
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "public_only": args.public_only,
        "timing": {"warmups": 3, "rounds": 8, "calls_per_round": 12, "order": "alternating"},
        "precision": {"application": "exact", "reduction_rtol": 1e-14, "reduction_atol": 0},
        "threads": {"torch": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                    "opencv": cv2.getNumThreads()},
        "comparison": "new API: equivalent arithmetic candidates, not a legacy Normalize regression claim",
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
