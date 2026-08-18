/******************************************************************************
 * cuda_mpf_mr.cu -- MPC_CUDA (mpc_cuda / cu_freal<PREC>) GPU vs CPU (MPFR/GMP
 *                   + OpenMP) benchmark for AXPY / GEMV / GEMM.
 *
 *   GPU : cuda_mpf_axpy / cuda_mul_mpfmatrix_vec / cuda_mul_mpfmatrix
 *         (src/mpflinear_cu.cu, register-resident cu_freal<PREC>)
 *   CPU : add_cmul_mpfvector / _bncomp_mul_mpfmatrix_mpfvec / _bncomp_mul_mpfmatrix
 *         (GMP mpf_t at the same working precision, OpenMP for GEMV/GEMM)
 *
 * The GPU side stores operands and results as plain double and accumulates at
 * PREC bits; the CPU side keeps mpf_t at PREC bits throughout, so the CPU is
 * the more accurate of the two.  relerr therefore measures how much the GPU's
 * double-precision I/O costs, not a GPU defect.
 *
 * PREC is a compile-time constant of mpflinear_cu.cu; build one binary per
 * precision (see build_mpf_bench.sh).
 *
 * One CSV line per measurement, matching the other CUDA drivers:
 *   RESULT,op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <gmp.h>

extern "C" {
#include "mpflinear_cu.h"
}
#include "bnc.h"
#include "mpflinear.h"
#include "bncomp.h"

#ifndef PREC
#define PREC 1024
#endif

/* ---------------------------------------------------------------- helpers */
static FILE *g_csv = NULL;

static void emit(const char *op, const char *prec, long dim, long nnz,
                 double tcpu, double tgpu, const char *cpu_kind, double relerr)
{
    printf("RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",
           op, prec, dim, nnz, tcpu, tgpu, cpu_kind, relerr);
    if (g_csv)
        fprintf(g_csv, "RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",
                op, prec, dim, nnz, tcpu, tgpu, cpu_kind, relerr);
    fflush(stdout);
}

static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned long rs = 20260730UL;
static double urand(void)
{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(rs >> 11) / 9007199254740992.0;
}

/* ---------------------------------------------------------------------------
 * High-precision reference (mpf at REFPREC bits).  The GPU returns plain
 * double, so comparing it against the CPU mpf result rounded to double gives
 * an identically-zero error; the informative quantity is the distance to a
 * reference computed well above both.  REFPREC = 4 * PREC.
 * ------------------------------------------------------------------------ */
#define REFPREC (4 * PREC)

/* relative 2-norm distance of a double array to an mpf reference array */
static double relerr_ref(const double *v, mpf_t *ref, long n)
{
    mpf_t num, den, d, t;
    double out;
    mpf_init2(num, REFPREC); mpf_init2(den, REFPREC);
    mpf_init2(d, REFPREC);   mpf_init2(t, REFPREC);
    mpf_set_ui(num, 0); mpf_set_ui(den, 0);
    for (long i = 0; i < n; i++) {
        mpf_set_d(t, v[i]);
        mpf_sub(d, t, ref[i]);
        mpf_mul(d, d, d);   mpf_add(num, num, d);
        mpf_mul(t, ref[i], ref[i]); mpf_add(den, den, t);
    }
    if (mpf_sgn(den) > 0) { mpf_div(num, num, den); }
    mpf_sqrt(num, num);
    out = mpf_get_d(num);
    mpf_clear(num); mpf_clear(den); mpf_clear(d); mpf_clear(t);
    return out;
}

/* ||a - b||_2 / ||a||_2 over plain double arrays */
static double relerr2(const double *a, const double *b, long n)
{
    double num = 0.0, den = 0.0;
    for (long i = 0; i < n; i++) {
        double d = a[i] - b[i];
        num += d * d; den += a[i] * a[i];
    }
    return (den > 0.0) ? sqrt(num / den) : sqrt(num);
}

int main(int argc, char *argv[])
{
    long axpy_sizes[16], sizes[16];
    int n_axpy = 0, n_sizes = 0;
    int reps = 3, blocks = 256, threads = 128, nthreads = omp_get_max_threads();
    long max_cpu = 1024;
    const char *csv_path = NULL;
    char prec_tag[32];

    /* defaults */
    { const long a[] = {4096, 16384, 65536, 262144, 1048576};
      for (unsigned i = 0; i < sizeof(a)/sizeof(a[0]); i++) axpy_sizes[n_axpy++] = a[i]; }
    { const long a[] = {128, 256, 512, 1024};
      for (unsigned i = 0; i < sizeof(a)/sizeof(a[0]); i++) sizes[n_sizes++] = a[i]; }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reps")    && i + 1 < argc) reps    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks")  && i + 1 < argc) blocks  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-cpu") && i + 1 < argc) max_cpu = atol(argv[++i]);
        else if (!strcmp(argv[i], "--omp")     && i + 1 < argc) nthreads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")     && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--sizes")   && i + 1 < argc) {
            n_sizes = 0; char *s = strdup(argv[++i]);
            for (char *t = strtok(s, ","); t && n_sizes < 16; t = strtok(NULL, ","))
                sizes[n_sizes++] = atol(t);
            free(s);
        } else if (!strcmp(argv[i], "--axpy-sizes") && i + 1 < argc) {
            n_axpy = 0; char *s = strdup(argv[++i]);
            for (char *t = strtok(s, ","); t && n_axpy < 16; t = strtok(NULL, ","))
                axpy_sizes[n_axpy++] = atol(t);
            free(s);
        } else {
            fprintf(stderr,
                "usage: %s [--sizes a,b,..] [--axpy-sizes a,b,..] [--reps N]\n"
                "          [--blocks N] [--threads N] [--max-cpu N] [--omp N] [--csv FILE]\n",
                argv[0]);
            return 1;
        }
    }
    if (csv_path) g_csv = fopen(csv_path, "a");
    omp_set_num_threads(nthreads);
    mpf_set_default_prec(PREC);
    snprintf(prec_tag, sizeof(prec_tag), "mpf%d", PREC);

    printf("=========================================================\n");
    printf(" MPC_CUDA (cu_freal<%d>) GPU vs CPU (GMP mpf_t, OpenMP) benchmark\n", PREC);
    printf("=========================================================\n");
    printf(" GPU working precision : %d bits (~%.0f decimal digits)\n",
           PREC, (double)PREC * 0.30103);
    printf(" CPU mpf_t precision   : %lu bits\n", (unsigned long)mpf_get_default_prec());
    printf(" CPU baseline threads  : %d\n", nthreads);
    printf(" reps/measurement      : %d (min wall-clock)\n", reps);
    printf(" CUDA grid             : %d blocks x %d threads\n", blocks, threads);
    printf(" CPU size cap (GEMM)   : %ld\n", max_cpu);
    printf("=========================================================\n");

    /* ================================================================ AXPY */
    printf("\n##### AXPY  y := alpha*x + y  (%s) #####\n", prec_tag);
    printf("  %10s %14s %14s %10s %14s\n", "N", "CPU [s]", "GPU [s]", "speedup", "relerr");
    for (int si = 0; si < n_axpy; si++) {
        long n = axpy_sizes[si];
        double *x  = (double *)malloc(sizeof(double) * n);
        double *y0 = (double *)malloc(sizeof(double) * n);
        double *yg = (double *)malloc(sizeof(double) * n);
        double *yc = (double *)malloc(sizeof(double) * n);
        double alpha = 2.0 * urand() - 1.0;
        for (long i = 0; i < n; i++) { x[i] = 2.0 * urand() - 1.0; y0[i] = 2.0 * urand() - 1.0; }

        /* ---- GPU ---- */
        memcpy(yg, y0, sizeof(double) * n);
        cuda_mpf_axpy(yg, alpha, x, n, blocks, threads);          /* warm-up */
        double tg = 1e300;
        for (int r = 0; r < reps; r++) {
            memcpy(yg, y0, sizeof(double) * n);
            double t0 = wtime();
            cuda_mpf_axpy(yg, alpha, x, n, blocks, threads);
            double t = wtime() - t0;
            if (t < tg) tg = t;
        }

        /* ---- CPU (mpf_t, OpenMP) ---- */
        MPFVector vx = init_mpfvector(n), vy = init_mpfvector(n), vc = init_mpfvector(n);
        mpf_t ma;
        mpf_init(ma); mpf_set_d(ma, alpha);
        for (long i = 0; i < n; i++) {
            set_mpfvector_i_d(vx, i, x[i]);
            set_mpfvector_i_d(vy, i, y0[i]);
        }
        add_cmul_mpfvector(vc, vy, ma, vx);                        /* warm-up */
        double tc = 1e300;
        for (int r = 0; r < reps; r++) {
            double t0 = wtime();
            add_cmul_mpfvector(vc, vy, ma, vx);
            double t = wtime() - t0;
            if (t < tc) tc = t;
        }
        for (long i = 0; i < n; i++) yc[i] = mpf_get_d(get_mpfvector_i(vc, i));

        /* reference at REFPREC bits */
        mpf_t *ref = (mpf_t *)malloc(sizeof(mpf_t) * n);
        { mpf_t ra, rx, ry;
          mpf_init2(ra, REFPREC); mpf_init2(rx, REFPREC); mpf_init2(ry, REFPREC);
          mpf_set_d(ra, alpha);
          for (long i = 0; i < n; i++) {
              mpf_init2(ref[i], REFPREC);
              mpf_set_d(rx, x[i]); mpf_set_d(ry, y0[i]);
              mpf_mul(ref[i], ra, rx); mpf_add(ref[i], ref[i], ry);
          }
          mpf_clear(ra); mpf_clear(rx); mpf_clear(ry);
        }
        double re = relerr_ref(yg, ref, n);
        emit("axpy", prec_tag, n, 0, tc, tg, "mpf-omp", re);
        printf("  %10ld %14.6e %14.6e %10.2f %14.3e\n", n, tc, tg, tc / tg, re);
        for (long i = 0; i < n; i++) mpf_clear(ref[i]);
        free(ref);

        mpf_clear(ma);
        free_mpfvector(vx); free_mpfvector(vy); free_mpfvector(vc);
        free(x); free(y0); free(yg); free(yc);
    }

    /* =============================================================== GEMV */
    printf("\n##### GEMV  y := A*x  (%s) #####\n", prec_tag);
    printf("  %10s %14s %14s %10s %14s\n", "N", "CPU [s]", "GPU [s]", "speedup", "relerr");
    for (int si = 0; si < n_sizes; si++) {
        long n = sizes[si];
        double *A  = (double *)malloc(sizeof(double) * n * n);
        double *x  = (double *)malloc(sizeof(double) * n);
        double *yg = (double *)malloc(sizeof(double) * n);
        double *yc = (double *)malloc(sizeof(double) * n);
        for (long i = 0; i < n * n; i++) A[i] = 2.0 * urand() - 1.0;
        for (long i = 0; i < n; i++)     x[i] = 2.0 * urand() - 1.0;

        cuda_mul_mpfmatrix_vec(yg, A, x, (int)n, blocks, threads);  /* warm-up */
        double tg = 1e300;
        for (int r = 0; r < reps; r++) {
            double t0 = wtime();
            cuda_mul_mpfmatrix_vec(yg, A, x, (int)n, blocks, threads);
            double t = wtime() - t0;
            if (t < tg) tg = t;
        }

        double tc = -1.0;
        if (n <= max_cpu) {
            MPFMatrix mA = init_mpfmatrix(n, n);
            MPFVector vx = init_mpfvector(n), vy = init_mpfvector(n);
            for (long i = 0; i < n; i++) {
                for (long j = 0; j < n; j++) set_mpfmatrix_ij_d(mA, i, j, A[i * n + j]);
                set_mpfvector_i_d(vx, i, x[i]);
            }
            _bncomp_mul_mpfmatrix_mpfvec(vy, mA, vx);               /* warm-up */
            tc = 1e300;
            for (int r = 0; r < reps; r++) {
                double t0 = wtime();
                _bncomp_mul_mpfmatrix_mpfvec(vy, mA, vx);
                double t = wtime() - t0;
                if (t < tc) tc = t;
            }
            for (long i = 0; i < n; i++) yc[i] = mpf_get_d(get_mpfvector_i(vy, i));
            free_mpfmatrix(mA); free_mpfvector(vx); free_mpfvector(vy);
        }

        mpf_t *ref = (mpf_t *)malloc(sizeof(mpf_t) * n);
        { mpf_t a1, x1, t1;
          mpf_init2(a1, REFPREC); mpf_init2(x1, REFPREC); mpf_init2(t1, REFPREC);
          for (long i = 0; i < n; i++) {
              mpf_init2(ref[i], REFPREC); mpf_set_ui(ref[i], 0);
              for (long k = 0; k < n; k++) {
                  mpf_set_d(a1, A[i * n + k]); mpf_set_d(x1, x[k]);
                  mpf_mul(t1, a1, x1); mpf_add(ref[i], ref[i], t1);
              }
          }
          mpf_clear(a1); mpf_clear(x1); mpf_clear(t1);
        }
        double re = relerr_ref(yg, ref, n);
        emit("matvec", prec_tag, n, 0, tc, tg, "mpf-omp", re);
        printf("  %10ld %14.6e %14.6e %10.2f %14.3e\n", n, tc, tg,
               (tc > 0.0) ? tc / tg : 0.0, re);
        for (long i = 0; i < n; i++) mpf_clear(ref[i]);
        free(ref);

        free(A); free(x); free(yg); free(yc);
    }

    /* =============================================================== GEMM */
    printf("\n##### GEMM  C := A*B  (%s) #####\n", prec_tag);
    printf("  %10s %14s %14s %10s %14s\n", "N", "CPU [s]", "GPU [s]", "speedup", "relerr");
    for (int si = 0; si < n_sizes; si++) {
        long n = sizes[si];
        double *A  = (double *)malloc(sizeof(double) * n * n);
        double *B  = (double *)malloc(sizeof(double) * n * n);
        double *Cg = (double *)malloc(sizeof(double) * n * n);
        double *Cc = (double *)malloc(sizeof(double) * n * n);
        for (long i = 0; i < n * n; i++) { A[i] = 2.0 * urand() - 1.0; B[i] = 2.0 * urand() - 1.0; }

        cuda_mul_mpfmatrix(Cg, A, B, (int)n, blocks, threads);      /* warm-up */
        double tg = 1e300;
        for (int r = 0; r < reps; r++) {
            double t0 = wtime();
            cuda_mul_mpfmatrix(Cg, A, B, (int)n, blocks, threads);
            double t = wtime() - t0;
            if (t < tg) tg = t;
        }

        double tc = -1.0;
        if (n <= max_cpu) {
            MPFMatrix mA = init_mpfmatrix(n, n), mB = init_mpfmatrix(n, n),
                      mC = init_mpfmatrix(n, n);
            for (long i = 0; i < n; i++)
                for (long j = 0; j < n; j++) {
                    set_mpfmatrix_ij_d(mA, i, j, A[i * n + j]);
                    set_mpfmatrix_ij_d(mB, i, j, B[i * n + j]);
                }
            _bncomp_mul_mpfmatrix(mC, mA, mB);                      /* warm-up */
            tc = 1e300;
            for (int r = 0; r < reps; r++) {
                double t0 = wtime();
                _bncomp_mul_mpfmatrix(mC, mA, mB);
                double t = wtime() - t0;
                if (t < tc) tc = t;
            }
            for (long i = 0; i < n; i++)
                for (long j = 0; j < n; j++)
                    Cc[i * n + j] = mpf_get_d(get_mpfmatrix_ij(mC, i, j));
            free_mpfmatrix(mA); free_mpfmatrix(mB); free_mpfmatrix(mC);
        }

        /* sampled reference: NS random entries of C at REFPREC bits */
        const long NS = (n * n < 512) ? n * n : 512;
        double *gsel = (double *)malloc(sizeof(double) * NS);
        mpf_t *ref = (mpf_t *)malloc(sizeof(mpf_t) * NS);
        { mpf_t a1, b1, t1;
          unsigned long save = rs;
          rs = 4242424242UL;
          mpf_init2(a1, REFPREC); mpf_init2(b1, REFPREC); mpf_init2(t1, REFPREC);
          for (long e = 0; e < NS; e++) {
              long i = (long)(urand() * (double)n), j = (long)(urand() * (double)n);
              if (i >= n) i = n - 1;
              if (j >= n) j = n - 1;
              mpf_init2(ref[e], REFPREC); mpf_set_ui(ref[e], 0);
              for (long k = 0; k < n; k++) {
                  mpf_set_d(a1, A[i * n + k]); mpf_set_d(b1, B[k * n + j]);
                  mpf_mul(t1, a1, b1); mpf_add(ref[e], ref[e], t1);
              }
              gsel[e] = Cg[i * n + j];
          }
          mpf_clear(a1); mpf_clear(b1); mpf_clear(t1);
          rs = save;
        }
        double re = relerr_ref(gsel, ref, NS);
        for (long e = 0; e < NS; e++) mpf_clear(ref[e]);
        free(ref); free(gsel);
        emit("matmul", prec_tag, n, 0, tc, tg, "mpf-omp", re);
        printf("  %10ld %14.6e %14.6e %10.2f %14.3e\n", n, tc, tg,
               (tc > 0.0) ? tc / tg : 0.0, re);

        free(A); free(B); free(Cg); free(Cc);
    }

    if (g_csv) fclose(g_csv);
    return 0;
}
