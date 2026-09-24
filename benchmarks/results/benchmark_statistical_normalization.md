# Statistical normalization: allocation and backend audit

This compares the new float32 quantitative APIs with their pre-optimization
implementation at `a30f12301501592a87b84defd92ceaeb1f477bee`. It is not a comparison
with released `Normalize`, which has different arithmetic and clipping semantics.

## Reproduce

```bash
git show a30f123:albucore/statistical_normalization.py > /tmp/statistical-normalization-baseline.py
uv run --no-sync python -m benchmarks.benchmark_statistical_normalization \
  --baseline-source /tmp/statistical-normalization-baseline.py \
  --output /tmp/normalization-candidates.json
uv run --no-sync python -m benchmarks.benchmark_statistical_normalization \
  --baseline-source /tmp/statistical-normalization-baseline.py --public-only \
  --output /tmp/normalization-public.json
```

The baseline source is loaded as a separate module outside the checkout. The
unchanged `stats.mean` implementation is shared. No released baseline contains
these APIs. `--quick` limits the shape grid to three representative sizes;
`--public-only` retains the public mean/RMS/application and pre-change pair after
candidate eligibility has been established. `--with-ax` remains an optional
consumer measurement and requires the separately linked AX implementation.

Each call includes selection gathering, needed flattening/copying, dtype
conversion, NumPy/Torch bridges, allocation, clipping and support zeroing.
Masks are caller inputs; their construction is outside the timer. Selection
uses every eighth or every second width position (12.5%/50% on canonical sizes).
Strided cells reverse the width axis for both HWC and DHWC inputs.

There are three warmups and eight rounds of twelve calls per candidate, with
candidate order reversed on alternate rounds. All five numerical runtime thread
environment variables are set to one before imports; Torch intra/inter-op are
1/1 and OpenCV internal parallelism is disabled. Raw samples are retained.
`tracemalloc` peak bytes are measured separately, without changing timed calls;
they describe NumPy/Python allocations, not process RSS or native Torch memory.

The backend audit covers the 12 canonical non-square HWC cells (C=1/3/9), seven
DHWC cells, a size sweep, and bounded extras on three shapes. These extras cover
selection, selected mean, support, clipping, unclamped application and reversed
width. The final public comparison additionally covers divide-only and identity
application; identity still produces an independent output.

## Candidate decisions

- NumPy application keeps clamp → subtract → direct divide → clip → support.
  Its first applicable ufunc allocates the output; later operations reuse it.
  This removes one full-array copy pass. Both implementations allocate one
  output buffer, so the improvement is memory traffic, not one fewer output.
- Float32 NumKong `norm`, `moments`, and `dot` are measured candidates, including
  their precision failures. A float64 result type alone does not establish
  `rtol=1e-14, atol=0` agreement with float64 square/mean. Casting to float64 before
  NumKong `dot` is also measured, including the full-size conversion allocation.
- OpenCV and eager CPU Torch application/reduction paths include finite-result
  checks where needed, bridges and output repair. Application eligibility is
  exact; reduction eligibility uses the existing `1e-14` tolerance.
- Existing `stats.mean` remains the global mean router. Selection gathering is
  now included in its public timings.
- LUT/StringZilla byte translation is inapplicable to unrestricted float32
  values. There are no grouped integer labels, Python pixel loops or runtime
  random generation to replace. Python/math is used only for scalar finalization.
  These atoms already live in Albucore; no additional extraction or upstream
  primitive is needed. Caller inputs are never eligible for in-place mutation.

## Results

Measured 2026-09-24 on AMD Ryzen 7 5800X, Linux x86_64, Python 3.13.14,
NumPy 2.5.1, Torch 2.13.0+cpu, OpenCV 5.0.0, NumKong 7.8.0. CPU affinity was
not pinned (all 16 logical CPUs allowed); each numerical runtime used one thread.
The host had 113 GiB available memory and no competing intensive workload at
preflight. These are one-host measurements, not a platform-wide guarantee.

Raw evidence:

- [Candidate screening](statistical_normalization_candidates.json): 43 cells,
  16 candidates each; canonical grid, six size boundaries and bounded extras.
  This was collected before RMS routing changed. The original square-image/raw-volume
  grid is retained in the runner and included in the final comparison.
- [Final public comparison](statistical_normalization_public.json): 59 cells,
  six public/baseline calls each, including divide-only and identity. The ten
  historical square/raw-volume cells were appended in a separate process using
  the same RNG stream and measurement settings, without repeating other cells.
- [Focused selected-mean rerun](statistical_normalization_mean_rerun.json): the
  only apparent slowdown above 15% was an unchanged mean path on reversed
  `768x1024x9`, 12.5% selection. Isolating that pair gave 3.103 ms baseline versus
  3.074 ms current (1.010x); the initial 18.9% slowdown was not reproduced.

All 354 final public/baseline eligibility comparisons passed: exact application
output and reduction `rtol=1e-14, atol=0`. In the candidate audit, NumKong float32
`dot` failed 40/43 cells, `norm` and `moments` each failed 36/43, and OpenCV RMS
failed 38/43. NumKong float64-cast/dot, NumPy einsum and Torch RMS passed all
43 cells; the float64-cast/dot route was slower than einsum on the large cells.

The RMS router uses float64 einsum only after selection, when the resulting
values are C-contiguous and contain at least **4,194,304 elements**. Flattening
is then a view. The square/mean fallback remains for smaller or strided values.
The conservative threshold leaves the crossover region below four million
elements on the previous implementation; it does not claim to be optimal for
every CPU. Around the exact boundary, final public times were:

| Elements | Previous RMS, ms | Current RMS, ms |
|---:|---:|---:|
| 2,097,152 | 0.7739 | 0.7706 |
| 3,145,728 | 2.0760 | 2.0148 |
| 4,194,303 | 3.1975 | 3.1408 |
| 4,194,304 | 3.1458 | 2.2721 |
| 4,194,305 | 3.1072 | 2.2607 |
| 5,242,880 | 5.7641 | 2.8941 |

Representative complete calls, all contiguous:

| Shape | Operation / controls | Previous, ms | Current, ms | Speedup |
|---|---|---:|---:|---:|
| 240x320x3 | Apply clamp=0, offset=.25, divisor=3 | 0.1298 | 0.1101 | 1.18x |
| 768x1024x9 | Same application | 10.6440 | 4.7272 | 2.25x |
| 48x240x320x3 | Same application | 13.1432 | 9.3530 | 1.41x |
| 48x240x320x3 | Apply without clamp, offset=.25, divisor=3 | 9.6520 | 6.3092 | 1.53x |
| 48x240x320x3 | Divide-only, divisor=3 | 7.3647 | 4.6627 | 1.58x |
| 768x1024x9 | RMS, no selection | 7.7270 | 3.9888 | 1.94x |
| 48x240x320x3 | RMS, no selection | 12.9940 | 6.4488 | 2.01x |
| 48x240x320x3 | RMS with 50% selection | 35.7676 | 34.6521 | 1.03x |
| 48x240x320x3 | Apply clamp/offset/divide/clip/support | 39.9053 | 30.3841 | 1.31x |

Across the final matrix, no application or RMS cell slowed by more than 2%.
The single apparent mean regression was investigated as described above.
Gathering dominates selected reductions, so the unselected RMS speedup must not
be generalized to masked calls.

For unselected `48x240x320x3`, traced RMS peak allocation fell from **88,540,136**
to **132,654 bytes** (84.44 to 0.13 MiB). Application still allocates its required
44,236,800-byte output; deleting the copy removes a pass rather than that buffer.
For 50% selection, gathering itself dominates peak allocation: 66,421,752 versus
66,359,312 bytes. This is why reduced square storage is not claimed as the same
peak-memory saving on every public path.

Correctness verification: 207 tests passed across statistical normalization,
existing normalization and statistics. Added cases cover all first-operation
branches, read-only/reversed/Fortran inputs, independent storage, support after
offset, overflow before concealment, the RMS size boundary, selection excluding
NaNs, zero, float32 subnormal and maximum finite values. No AX/DaT pipeline or
training result is inferred from these core measurements.
