# bench/elem — DD/TD/QD elementary function benchmarks

Benchmarks for the dtq-0.0.3 FMA-based elementary functions ported to
plain C (`include/bncelem*.h`, `src/elem_vector.c`).

| program | what it measures |
|---------|------------------|
| `elem_bench.c` | scalar accuracy vs MPFR (512 bits) and ns/call; compares the previous implementations where one existed (`c_dd_exp_orig`, `c_dd_log_orig`) with the new FMA-based ones |
| `elem_vec_bench.c` | SIMD-vectorized `bnc_*_array` functions vs the scalar ones: ns/element, speedup, and the max deviation (accumulated in MPFR, since equal values may be normalized differently) |

## Build

```sh
# scalar accuracy/perf (any machine)
gcc -O2 -ffp-contract=off -Iinclude -I/usr/local/include \
    bench/elem/elem_bench.c -o bench/elem/elem_bench \
    -L/usr/local/lib -lmpfr -lgmp -lm

# vector bench, NEON (aarch64)
gcc -O3 -ffp-contract=off -mcpu=cortex-a76 -DBNC_ENABLE_NEON \
    -Iinclude -I/usr/local/include \
    bench/elem/elem_vec_bench.c src/elem_vector.c \
    -o bench/elem/elem_vec_bench_neon -L/usr/local/lib -lmpfr -lgmp -lm

# vector bench, SVE2 flags (uses the NEON path; VL=128 on target CPUs)
gcc -O3 -ffp-contract=off -mcpu=neoverse-v2 -DBNC_ENABLE_SVE2 -DBNC_ENABLE_NEON ...

# forced scalar fallback (results must be bitwise equal to the scalar API)
gcc -O3 -ffp-contract=off -DBNC_ELEM_VECTOR_FORCE_SERIAL ...

# x86 (AVX2/AVX-512): -mavx2 -mfma  /  -mavx512f -mavx512dq -mfma
```

Measured results from the porting session (Cortex-X925, gcc 13.3) are in
`results/`.
