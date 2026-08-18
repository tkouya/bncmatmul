# GPU AXPY / GEMV / GEMM benchmark — cross-machine merge

Two multiple-precision GPU libraries, one table:

| `lib` | GPU side | CPU baseline |
|-------|----------|--------------|
| `gdtq` | `g{dd,td,qd,ds,ts,qs}linear` device kernels (+ native `d`/`f`) from `libbncmatmul-0.24_cuda.a` | `libbncmatmul-0.24-omp*` (`_bncomp_*`, OpenMP) |
| `mpc_cuda` | `mpc_cuda` `cu_freal<PREC>` kernels in `src/mpflinear_cu.cu` | GMP `mpf_t` at the same precision + OpenMP |

State of the tree: **pre-FMA** — `BNC_USE_NEW_FMA` is undefined, so the
branch-free DW/TW/QW FMA of arXiv:2607.11391 is *not* in the measured code.
That is recorded in the `fma` column (`pre-FMA`).

## Reproducing on another machine (e.g. x86 + H100)

```sh
# 1. GPU architecture for the target (H100 = sm_90)
export ARCH=sm_90

# 2. build both driver families
sh bench/cuda/build_cuda_bench.sh      # gdtq  (needs libbncmatmul-0.24_cuda.a)
sh bench/cuda/build_mpf_bench.sh       # mpc_cuda, one binary per PREC

# 3. run, tagging the platform
PLATFORM=x86-h100 sh bench/cuda/run_gpu_bench.sh
```

Keep the defaults (`DIMS`, `AXPY_DIMS`, `REPS`, `BLOCKS`, `THREADS`, `PRECS`)
identical on both machines or the rows will not line up. They can be overridden
from the environment; whatever is used must be used on both sides.

Result: `bench/cuda/out/gpu_bench_merged.csv`.

## Merging

```sh
python3 bench/cuda/merge_prep.py merge \
    arm/gpu_bench_merged.csv  x86/gpu_bench_merged.csv \
    --out gpu_bench_all.csv
```

`merge` refuses to combine files whose column list differs, and reports any
duplicated `(platform, lib, op, prec, dim)` key.

If the x86 run was made with an older driver that emits only the raw
`op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr` line, annotate it here
instead of re-running — every metadata field can be supplied on the command
line:

```sh
python3 bench/cuda/merge_prep.py annotate \
    --raw x86_raw.csv --lib gdtq --platform x86-h100 \
    --host node01 --cpu "Intel Xeon Gold 6526Y" --cpu-threads 32 \
    --gpu "NVIDIA H100 NVL" --gpu-arch sm_90 --cuda 12.4 \
    --compiler "gcc 13.3" --out x86/gpu_bench_merged.csv
```

## Schema

| column | meaning |
|--------|---------|
| `platform` | join tag you choose, e.g. `arm-gb10`, `x86-h100` |
| `host`, `cpu`, `cpu_threads`, `gpu`, `gpu_arch`, `cuda`, `compiler` | auto-detected environment |
| `lib` | `gdtq` or `mpc_cuda` |
| `op` | `axpy`, `matvec`, `matmul` |
| `kernel` | the expression actually measured (see the caveat below) |
| `prec` | `d f dd td qd ds ts qs` or `mpf128 … mpf1024` |
| `base`, `ncomp`, `prec_bits` | `binary32`/`binary64`/`mpf`, component count, working bits |
| `dim` | vector length (AXPY) or matrix order (GEMV/GEMM) |
| `nnz` | non-zeros (sparse only; 0 here) |
| `cpu_time`, `gpu_time` | seconds per call, minimum over `REPS` runs |
| `speedup` | `cpu_time / gpu_time` (blank when the CPU run was skipped) |
| `cpu_kind` | which CPU baseline was used |
| `relerr` | see below |
| `fma` | `pre-FMA` / `FMA` — state of the branch-free FMA integration |
| `tree`, `date` | source tree tag and run date |

A negative `cpu_time` means the CPU baseline was skipped for that size
(`--max-cpu`); `speedup` is then blank.

## Two caveats worth carrying into the paper

1. **`gdtq` + `axpy` is a SCAL, not an AXPY.** The driver measures
   `cmul_g<P>vector_dev`, i.e. `c = alpha * a`, because the GPU library has no
   fused `alpha*x + y` kernel. The `op` label is kept as `axpy` so the two
   machines join, and the `kernel` column spells out what was really timed.
   `mpc_cuda` + `axpy` *is* a fused `y = alpha*x + y`
   (`cuda_mpf_axpy`, added 2026-07-30).

2. **`relerr` means different things per library.**
   * `gdtq`: `||c_cpu - c_gpu||_2 / ||c_cpu||_2`, both sides at the same
     multi-component precision.
   * `mpc_cuda`: distance to a GMP reference at `4 x PREC` bits. The
     `mpc_cuda` host interface passes operands and results as plain `double`,
     so a GPU-vs-CPU comparison would be identically zero; the reported
     ~1e-16 is the cost of that double interface, not of the kernel.
