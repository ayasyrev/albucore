# Uint8 statistics: NumKong overflow workaround

Albucore [#196](https://github.com/albumentations-team/albucore/issues/196) is a
correctness fix for [NumKong #412](https://github.com/ashvardanian/NumKong/issues/412).
`nk.moments(uint8)` can overflow in Haswell/AVX2, producing zero standard
deviations for nonconstant bright images. Global and per-channel `mean_std`/`std`
now use OpenCV; their LUT and OpenCV normalization consumers inherit the fix.
No process-wide capabilities or dependencies change. Sum-only and float routes
remain unchanged.

Global reductions flatten in physical order. Contiguous per-channel reductions
reshape to `(-1, 1, C)` without copying; strided reductions process scalar planes
to avoid expensive interleaved packing. Above `get_opencv_max_channels()` they
fall back to NumPy float64. OpenCV 5.0.0.93 encodes 128 channels: tests cover 128
and 129, since passing 129 directly silently changes the reduction shape.

## Evidence

21 October 2026, Linux x86-64 / AMD Ryzen 7 5800X, Python 3.13.14,
NumPy 2.5.1, OpenCV 5.0.0.93, NumKong 7.8.5, Torch 2.14.1+cpu.
Base `9cbc0da2130e4dc2196b17b3ce35992b89d43fe3` was built as a wheel and
installed outside its source checkout. The benchmark verifies separate import
paths and retains independent baseline/current function globals in one process.

The [script](../benchmark_uint8_statistics.py) records [raw timings and metadata](uint8_statistics_issue196.json):
132 public operation cells and 44 backend-comparison cells. HWC sizes are
128x160, 240x320, 480x640, 768x1024 with C=1/3/9; additional cases cover rank-4,
rank-5, planar and sliced arrays. Inputs are reproducible uint8 values 240–255.
Five warmups precede 21 alternating samples; 64 initial >15% regressions were
rerun with 51 alternating samples. OpenCV parallelism is disabled, Torch intra-
and inter-op threads are one, and OpenMP/BLAS environment limits are one.

Every current output passes an independent float64 reference. The baseline is
numerically wrong in 27 public cells; its timings remain recorded, not used as
correctness evidence. Ratios below use the paired rerun where present.

| Public operation | Median current/base | Maximum current/base |
| --- | ---: | ---: |
| `mean_std(global)` | 1.228 | 1.524 |
| `std(global)` | 1.237 | 1.558 |
| `mean_std(per_channel)` | 1.287 | 2.610 |
| `std(per_channel)` | 1.234 | 2.592 |
| `normalize_per_image(image)` | 1.019 | 1.057 |
| `normalize_per_image(image_per_channel)` | 1.031 | 1.206 |

This is a measured correctness/performance tradeoff, not a speedup claim.
RGB per-channel normalization at medium/large sizes costs 18.5–20.6% more;
for 480x640x3, 570.64 us becomes 686.49 us. The wide NumPy and Torch candidates
are much slower than OpenCV. No ARM64 or other-host timing claim is made.

## Candidates and scope

- Packing every strided input was rejected: planar RGB statistics regressed up
  to 42x. Scalar-plane reductions remove that avoidable copy.
- Direct NumPy float64, Torch float64, global histogram reduction and per-channel
  OpenCV loops were checked against the same independent oracle and measured.
- `cv2.split` helps some contiguous RGB sizes but adds full-size channel outputs
  and is slower for C=1/9. A new RGB allocation/specialization is deferred from
  this bounded correctness fix; the alternative remains in the raw evidence.
- NumKong uint8 moments is excluded for correctness regardless of speed. Float
  widening and process-wide disabling are not selected as workarounds.
- LUT application is preserved; LUT/StringZilla are not statistics reducers.
  Histogram/`bincount` is slower here. There is no RNG or annotation work.
  Inputs stay immutable, so in-place reductions are inapplicable.
- Existing Albucore statistics already centralize the affected AX consumers;
  no new atom or AX implementation is required. An AC release and AX dependency
  update are still needed for downstream delivery.

## Reproduction

Build/install the base into a separate target directory without its dependencies,
then use the candidate's environment for both implementations:

```bash
git archive 9cbc0da2130e4dc2196b17b3ce35992b89d43fe3 | tar -x -C /tmp/albucore-base-src
uv build --wheel /tmp/albucore-base-src --out-dir /tmp/albucore-base-wheels
uv pip install --no-deps --target /tmp/albucore-base /tmp/albucore-base-wheels/albucore-0.3.1-py3-none-any.whl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python benchmarks/benchmark_uint8_statistics.py \
  --baseline-installed /tmp/albucore-base --output /tmp/uint8-statistics.json
```

Create the temporary source directory first. The environment needs the package's
`headless` and `torch` extras. The JSON includes baseline/current import paths,
input shapes/strides, library versions, effective OpenCV/Torch threads, raw samples,
and whether each baseline result passed the oracle.
