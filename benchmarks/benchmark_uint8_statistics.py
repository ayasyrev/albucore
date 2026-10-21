"""Compare the #196 fix with an installed base and alternative uint8 reducers.

Run with one-thread environment variables and an isolated baseline wheel::

    uv run python benchmarks/benchmark_uint8_statistics.py \
        --baseline-installed /tmp/albucore-baseline --output results.json

The baseline must be installed outside its source checkout. Both packages are
loaded before timing; retained function objects keep their own module globals.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter_ns

import cv2
import numkong as nk
import numpy as np
import torch


def load_packages(baseline_path):
    sys.path.insert(0, str(baseline_path))
    baseline = importlib.import_module("albucore")
    baseline_stats = importlib.import_module("albucore.stats")
    assert Path(baseline.__file__).is_relative_to(baseline_path)
    for name in list(sys.modules):
        if name == "albucore" or name.startswith("albucore."):
            del sys.modules[name]
    sys.path.pop(0)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    current = importlib.import_module("albucore")
    current_stats = importlib.import_module("albucore.stats")
    assert baseline_stats is not current_stats
    assert baseline.mean_std.__globals__ is baseline_stats.__dict__
    assert current.mean_std.__globals__ is current_stats.__dict__
    return baseline, current


def measure(functions, samples=21):
    for function in functions.values():
        for _ in range(5):
            function()
    timings = {name: [] for name in functions}
    names = list(functions)
    for index in range(samples):
        order = names if index % 2 == 0 else names[::-1]
        for name in order:
            start = perf_counter_ns()
            functions[name]()
            timings[name].append((perf_counter_ns() - start) / 1000)
    return {
        name: {
            "median_us": float(np.median(values)),
            "mad_us": float(np.median(np.abs(values - np.median(values)))),
            "samples_us": values,
        }
        for name, values in timings.items()
    }


def oracle(array, operation):
    axis = tuple(range(array.ndim - 1)) if operation.endswith("per_channel") else None
    mean = array.mean(axis=axis, dtype=np.float64)
    std = array.std(axis=axis, dtype=np.float64) + 1e-4
    if operation.startswith("normalize"):
        return ((array.astype(np.float64) - mean) / std).clip(-20, 20)
    return (mean, std) if operation.startswith("mean_std") else std


def invoke(package, array, operation):
    axis = "per_channel" if operation.endswith("per_channel") else None
    if operation.startswith("normalize"):
        return package.normalize_per_image(array, "image_per_channel" if axis else "image")
    function = package.mean_std if operation.startswith("mean_std") else package.std
    return function(array, axis)


def assert_result(result, expected):
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)


def histogram_stats(array):
    counts = np.bincount(array.ravel(), minlength=256)
    values = np.arange(256, dtype=np.float64)
    mean = counts @ values / array.size
    variance = counts @ np.square(values - mean) / array.size
    return mean, np.sqrt(variance) + 1e-4


def candidates(array, per_channel):
    axes = tuple(range(array.ndim - 1)) if per_channel else None

    def opencv():
        work = (
            np.ascontiguousarray(array).reshape(-1, 1, array.shape[-1])
            if per_channel
            else array.ravel(order="K").reshape(-1, 1)
        )
        mean, std = cv2.meanStdDev(work)
        return (mean[:, 0], std[:, 0] + 1e-4) if per_channel else (float(mean[0, 0]), float(std[0, 0]) + 1e-4)

    def opencv_loop():
        results = [cv2.meanStdDev(array[..., c].ravel(order="K").reshape(-1, 1)) for c in range(array.shape[-1])]
        return np.array([pair[0][0, 0] for pair in results]), np.array([pair[1][0, 0] + 1e-4 for pair in results])

    def opencv_split():
        work = np.ascontiguousarray(array).reshape(-1, 1, array.shape[-1])
        results = [cv2.meanStdDev(channel) for channel in cv2.split(work)]
        return np.array([pair[0][0, 0] for pair in results]), np.array([pair[1][0, 0] + 1e-4 for pair in results])

    def torch_stats():
        std, mean = torch.std_mean(torch.from_numpy(array).to(torch.float64), dim=axes, correction=0)
        return mean.numpy(), std.numpy() + 1e-4

    result = {
        "opencv": opencv,
        "numpy": lambda: (array.mean(axis=axes, dtype=np.float64), array.std(axis=axes, dtype=np.float64) + 1e-4),
        "torch": torch_stats,
    }
    result["opencv_loop" if per_channel else "histogram"] = (
        opencv_loop if per_channel else lambda: histogram_stats(array)
    )
    if per_channel:
        result["opencv_split"] = opencv_split
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-installed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    baseline, current = load_packages(args.baseline_installed.resolve())
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "numkong": nk.__version__,
        "torch": torch.__version__,
        "enabled_capabilities": [name for name, enabled in nk.get_capabilities().items() if enabled],
        "threads": {
            "opencv": cv2.getNumThreads(),
            "torch": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
        },
        "baseline_import": baseline.__file__,
        "current_import": current.__file__,
        "warmups": 5,
        "samples": 21,
        "regression_samples": 51,
        "units": "microseconds",
        "seed": 196,
        "values": "uint8 uniform 240..255",
    }
    shapes = [(h, w, c) for h, w in [(128, 160), (240, 320), (480, 640), (768, 1024)] for c in [1, 3, 9]]
    shapes += [(4, 128, 160, c) for c in [1, 3, 9]] + [(2, 4, 128, 160, 3)]
    cases = [(shape, "c") for shape in shapes]
    cases += [
        (shape, storage)
        for shape in [(480, 640, 3), (480, 640, 9), (4, 128, 160, 3)]
        for storage in ["planar", "sliced"]
    ]
    operations = [
        "mean_std_global",
        "std_global",
        "mean_std_per_channel",
        "std_per_channel",
        "normalize_image",
        "normalize_image_per_channel",
    ]
    rows, candidate_rows = [], []
    for shape, storage in cases:
        array = np.random.default_rng(196).integers(240, 256, shape, dtype=np.uint8)
        if storage == "planar":
            array = np.moveaxis(np.ascontiguousarray(np.moveaxis(array, -1, 0)), 0, -1)
        elif storage == "sliced":
            array = np.repeat(array, 2, axis=-2)[..., ::2, :]
        for operation in operations:
            expected = oracle(array, operation)
            functions = {
                "baseline": lambda: invoke(baseline, array, operation),
                "current": lambda: invoke(current, array, operation),
            }
            assert_result(functions["current"](), expected)
            baseline_correct = True
            try:
                assert_result(functions["baseline"](), expected)
            except AssertionError:
                baseline_correct = False
            timing = measure(functions)
            ratio = timing["current"]["median_us"] / timing["baseline"]["median_us"]
            row = {
                "shape": shape,
                "storage": storage,
                "strides": array.strides,
                "operation": operation,
                "baseline_correct": baseline_correct,
                "timing": timing,
            }
            if ratio > 1.15:
                row["paired_rerun"] = measure(functions, samples=51)
            rows.append(row)
        # Compare alternate implementations on every size/channel/layout in the manifest.
        for per_channel in [False, True]:
            functions = candidates(array, per_channel)
            expected = oracle(array, "mean_std_per_channel" if per_channel else "mean_std_global")
            for function in functions.values():
                assert_result(function(), expected)
            candidate_rows.append(
                {"shape": shape, "storage": storage, "per_channel": per_channel, "timing": measure(functions)}
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "rows": rows, "candidates": candidate_rows}, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "public_cells": len(rows),
                "candidate_cells": len(candidate_rows),
                "baseline_incorrect_cells": sum(not row["baseline_correct"] for row in rows),
                "rerun_cells": sum("paired_rerun" in row for row in rows),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
