/************************************************************/
/* vecbench.c ... Vector arithmetic with AVX, AVX2 or AVX-512 */
/* 2020-02-29 (SAT) Tomonori Kouya                          */
/* gcc -O3 -mavx2 -mfma bncavx.c -L./ -lbnc rdd.o mpfr_dd.o -lmpfr -lgmp -lm */
/************************************************************/
#include <stdio.h>
#include <math.h>

// rdd.h
//#include "rdd.h"
#include "bncavx.h"
// mpfr_dd.h
//#include "mpfr_dd_td_qd.h" 
#include "mpfr_dtq_sd.h" 

#define USE_FLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR
#include "flinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"

#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#include "dlinear.h"

#include "ddlinear.h"
#include "_vecbench_dd.h"        // benchmark routines: AVX2
#include "_vecbench_dd_avx512.h" // benchmark routines: AVX-512

#include "tdlinear.h"
#include "_vecbench_td.h"        // benchmark routines: AVX2
#include "_vecbench_td_avx512.h" // benchmark routines: AVX-512

#include "qdlinear.h"
#include "_vecbench_qd.h"        // benchmark routines: AVX2
#include "_vecbench_qd_avx512.h" // benchmark routines: AVX-512

// get_secv()
#include "get_secv.h"

#define ALIGNMENT 32 // alignof(_m256d)

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_dsvector(dsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    dsfloat dssqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_ds(dssqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rds_set_ui(vec[i].val, i + 1);
        rds_mul(vec[i].val, vec[i].val, dssqrt.val);
    }
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_tsvector(tsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    tsfloat tssqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_ts(tssqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rts_set_ui(vec[i].val, i + 1);
        rts_mul(vec[i].val, vec[i].val, tssqrt.val);
    }
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_qsvector(qsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    qsfloat qssqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_qs(qssqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rqs_set_ui(vec[i].val, i + 1);
        rqs_mul(vec[i].val, vec[i].val, qssqrt.val);
    }
}


// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_ddvector(ddfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    ddfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_dd(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rdd_set_ui(vec[i].val, i + 1);
        rdd_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_tdvector(tdfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    tdfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 256);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_td(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rtd_set_ui(vec[i].val, i + 1);
        rtd_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
}

// qdrel_diff
inline static qdfloat qdrel_diff(qdfloat a, qdfloat b)
{
    qdfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rqd_sub(rel_diff.val, a.val, b.val);
    rqd_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rqd_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rqd_abs(abs_a.val, a.val);
        rqd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}

qdfloat qdrel_diff_array(qdfloat approx_a[], qdfloat approx_b[], int dim, int print_flag)
{
    int i;
    qdfloat rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = qdrel_diff(approx_a[0], approx_b[0]);
    rqd_set(rel_min.val, rel_diff.val);
    rqd_set(rel_max.val, rel_diff.val);
    rqd_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = qdrel_diff(approx_a[i], approx_b[i]);
        if(rqd_cmp(rel_diff.val, rel_min.val) < 0) rqd_set(rel_min.val, rel_diff.val);
        if(rqd_cmp(rel_diff.val, rel_max.val) > 0) rqd_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rqd_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (qdfloat)dim;
    rqd_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rqd_out_str(rel_max.val); printf(" "); rqd_out_str(rel_min.val);  printf(" "); rqd_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}
// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_qdvector(qdfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    qdfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 512);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_qd(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rqd_set_ui(vec[i].val, i + 1);
        rqd_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
}

// MFlops
static inline double mflops(int iterative_times, double seconds)
{
    return (double)iterative_times / seconds / (double)(1024 * 1024);
}

// GFlops
static inline double gflops(int iterative_times, double seconds)
{
    return (double)iterative_times / seconds / (double)(1024 * 1024 * 1024);
}

// check simd functions
int main(int argc, char *argv[])
{
    int dim, min_dim, dim_step, i, max_itimes, itimes;
    float *fa, *fb, *fc, *fd, *fret;
    double *da, *db, *dc, *dd, *dret;
    double stime, etime;

    ddfloat *dda, *ddb, *ddc, *ddd, *ddret;
    tdfloat *dta, *dtb, *dtc, *dtd, *dtret;
    qdfloat *dqa, *dqb, *dqc, *dqd, *dqret;
    DDVector ddv_a, ddv_b, ddv_c, ddv_ret;
    TDVector dtv_a, dtv_b, dtv_c, dtv_ret;
    QDVector dqv_a, dqv_b, dqv_c, dqv_ret;

    dsfloat *dsa, *dsb, *dsc, *dsd, *dsret;
    tsfloat *tsa, *tsb, *tsc, *tsd, *tsret;
    qsfloat *qsa, *qsb, *qsc, *qsd, *qsret;
    DSVector dsv_a, dsv_b, dsv_c, dsv_ret;
    TSVector tsv_a, tsv_b, tsv_c, tsv_ret;
    QSVector qsv_a, qsv_b, qsv_c, qsv_ret;

    double ddtime[10], tdtime[10], qdtime[10];
    int dditer[10], tditer[10], qditer[10];

    if(argc <= 1)
    {
        fprintf(stderr, "[usage] %s [min_dim]\n", argv[0]);
        return 0;
    }
    min_dim = atoi(argv[1]);
    printf("min_dim = %d\n", min_dim);

// Alignment
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 -- float -- ");
	printf("alignof(__m256d) = %ld\n", __alignof__(__m256d));
#elif defined(__AVX512F__)
    printf("-- AVX512 -- float -- ");
	printf("alignof(__m512d) = %ld\n", __alignof__(__m512d));
#endif // defined(__AVX2__)

#ifdef FVEC 
#if defined(__AVX2__) && !defined(__AVX512F__)
    //printf('float---');
// initialize arrays
    fa = (float *)aligned_alloc(32, sizeof(float) * dim);
    fb = (float *)aligned_alloc(32, sizeof(float) * dim);
    fc = (float *)aligned_alloc(32, sizeof(float) * dim);
    fd = (float *)aligned_alloc(32, sizeof(float) * dim);
    fret = (float *)aligned_alloc(32, sizeof(float) * dim);

    for(i = 0; i < dim; i++)
    {
        fa[i] = (float)rand() / (float)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
        fb[i] = (float)rand() / (float)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
        fc[i] = (float)rand() / (float)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
    }

    //#ifdef DVEC
    // FMA
    printf("Start _bncavx2_ffma(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_ffma(fret, fa, fb, fc, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
//    stime = get_secv();
//    _bncavx2_dfma(dret, da, db, dc, dim);
//    etime = get_secv();
//    printf("end ... %f(avx) ", etime - stime);
   printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // test fma
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            fd[i] = fa[i] * fb[i] + fc[i];
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    for(i = 0; i < dim; i++)
//        dd[i] = da[i] * db[i] + dc[i];
//    etime = get_secv();
//    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    frel_diff_array(fret, fd, dim, 0);

#if 0
    // MUL
    printf("Start _bncavx2_fmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
         itimes = 0;
        stime = get_secv();
        do {
            fd[i] = fa[i] * fb[i];
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    _bncavx2_dmul(dret, da, db, dim);
//    etime = get_secv();
//    printf("end ... %f(avx) ", etime - stime);

    // test mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_fmul(fret, fa, fb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    for(i = 0; i < dim; i++)
//        dd[i] = da[i] * db[i];
//    etime = get_secv();
//    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    frel_diff_array(fret, fd, dim, 0);

    // DIV
    printf("Start _bncavx2_fdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_fdiv(dret, da, db, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %f(avx) ", etime - stime);

    // test normal div
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                fd[i] = fa[i] / fb[i];
        } while(++itimes < max_itimes);
        etime = get_secv();
        max_itimes *= 2;
    } while(etime < 1.0);
    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("fa, fb, fc, fret[%d], fd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, fa[i], fb[i], fc[i], fret[i], fd[i]);

    // rel_diff: abs(dret - dd) / dd
    frel_diff_array(fret, fd, dim, 0);

    // ADD
    printf("Start _bncavx2_fadd(dim = %d) ... ", dim);
    stime = get_secv();
    _bncavx2_dadd(dret, da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal add
    stime = get_secv();
    for(i = 0; i < dim; i++)
        dd[i] = da[i] + db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // SUB
    printf("Start _bncavx2_ddiv(dim = %d) ... ", dim);
    stime = get_secv();
    _bncavx2_dsub(dret, da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal sub
    stime = get_secv();
    for(i = 0; i < dim; i++)
        dd[i] = da[i] - db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // dotp
    printf("Start _bncavx2_ddotp(dim = %d) ... ", dim);
    stime = get_secv();
    dret[0] = _bncavx2_ddotp(da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal mul
    stime = get_secv();
    dd[0] = 0.0;
    for(i = 0; i < dim; i++)
        dd[0] += da[i] * db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

 //   for(i = 0; i < dim; i++)
//        printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

   // rel_diff: abs(dret - dd) / dd
    printf("fret, fd = %25.17e, %25.17e\n", fret[0], fd[0]);
    frel_diff_array(fret, fd, 1, 0);
#endif // 0
#endif //defined(__AVX2__) && !defined(__AVX512F__)
#endif // ifdef FVEC

//#ifdef DVEC
// initialize arrays
/*    da = (double *)calloc(dim, sizeof(double));
    db = (double *)calloc(dim, sizeof(double));
    dc = (double *)calloc(dim, sizeof(double));
    dd = (double *)calloc(dim, sizeof(double));
    dret = (double *)calloc(dim, sizeof(double));
*/
    da = (double *)aligned_alloc(32, sizeof(double) * dim);
    db = (double *)aligned_alloc(32, sizeof(double) * dim);
    dc = (double *)aligned_alloc(32, sizeof(double) * dim);
    dd = (double *)aligned_alloc(32, sizeof(double) * dim);
    dret = (double *)aligned_alloc(32, sizeof(double) * dim);

    for(i = 0; i < dim; i++)
    {
        da[i] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
        db[i] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
        dc[i] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
    }
#ifdef DVEC
#if defined(__AVX2__) && !defined(__AVX512F__)
#elif defined(__AVX512F__)
#endif //defined(__AVX2__) && !defined(__AVX512F__)

    // FMA
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("Start _bncavx2_dfma(dim = %d) ... ", dim);
#elif defined(__AVX512F__)
    printf("Start _bncavx512_dfma(dim = %d) ... ", dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_dfma(dret, da, db, dc, dim);
#elif defined(__AVX512F__)
            _bncavx512_dfma(dret, da, db, dc, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
//    stime = get_secv();
//    _bncavx2_dfma(dret, da, db, dc, dim);
//    etime = get_secv();
//    printf("end ... %f(avx) ", etime - stime);
   printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // test fma
    do {
        itimes = 0;
        stime = get_secv();
        do {
            dd[i] = da[i] * db[i] + dc[i];
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    for(i = 0; i < dim; i++)
//        dd[i] = da[i] * db[i] + dc[i];
//    etime = get_secv();
//    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // MUL
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("Start _bncavx2_dmul(dim = %d) ... ", dim);
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dmul(dret, da, db, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    _bncavx2_dmul(dret, da, db, dim);
//    etime = get_secv();
//    printf("end ... %f(avx) ", etime - stime);

    // test mul
    do {
        itimes = 0;
        stime = get_secv();
        do {
            dd[i] = da[i] * db[i];
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    printf("end ... %d, %f\n", itimes, etime / (double)itimes);
//    stime = get_secv();
//    for(i = 0; i < dim; i++)
//        dd[i] = da[i] * db[i];
//    etime = get_secv();
//    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // DIV
    printf("Start _bncavx2_ddiv(dim = %d) ... ", dim);
    stime = get_secv();
    _bncavx2_ddiv(dret, da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal div
    stime = get_secv();
    for(i = 0; i < dim; i++)
        dd[i] = da[i] / db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

    for(i = 0; i < dim; i++)
        printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // ADD
    printf("Start _bncavx2_dadd(dim = %d) ... ", dim);
    stime = get_secv();
    _bncavx2_dadd(dret, da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal add
    stime = get_secv();
    for(i = 0; i < dim; i++)
        dd[i] = da[i] + db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // SUB
    printf("Start _bncavx2_ddiv(dim = %d) ... ", dim);
    stime = get_secv();
    _bncavx2_dsub(dret, da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal sub
    stime = get_secv();
    for(i = 0; i < dim; i++)
        dd[i] = da[i] - db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

    //for(i = 0; i < dim; i++)
    //    printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

    // rel_diff: abs(dret - dd) / dd
    drel_diff_array(dret, dd, dim, 0);

    // dotp
    printf("Start _bncavx2_ddotp(dim = %d) ... ", dim);
    stime = get_secv();
    dret[0] = _bncavx2_ddotp(da, db, dim);
    etime = get_secv();
    printf("end ... %f(avx) ", etime - stime);

    // test normal mul
    stime = get_secv();
    dd[0] = 0.0;
    for(i = 0; i < dim; i++)
        dd[0] += da[i] * db[i];
    etime = get_secv();
    printf(" vs. %f(normal)\n", etime - stime);

 //   for(i = 0; i < dim; i++)
//        printf("da, db, dc, dret[%d], dd[%d] = %10.3e, %10.3e, %10.3e, %25.17e, %25.17e\n", i, i, da[i], db[i], dc[i], dret[i], dd[i]);

   // rel_diff: abs(dret - dd) / dd
    printf("dret, dd = %25.17e, %25.17e\n", dret[0], dd[0]);
    drel_diff_array(dret, dd, 1, 0);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
#endif // ifdef DVEC

#ifdef DSVEC
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 --\n");
    printf("DSVEC ... dim = %d\n", dim);
    printf("                           MFLOPS\n");
//    printf("       dim     ddvadd      ddadd    rdd_add     ddvmul      ddmul    rdd_mul\n");
    printf("       dim     dsvadd      dsadd    rds_add     dsvmul      dsmul    rds_mul     dsvdiv      dsdiv    rds_div\n");
//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
//  printf("        32   2190.374    812.698   1211.834   2592.405    846.281   1329.870    860.504    474.074    453.097
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;

    dsa = (dsfloat *)aligned_alloc(32, sizeof(dsfloat) * dim);
    dsb = (dsfloat *)aligned_alloc(32, sizeof(dsfloat) * dim);
    dsc = (dsfloat *)aligned_alloc(32, sizeof(dsfloat) * dim);
    dsd = (dsfloat *)aligned_alloc(32, sizeof(dsfloat) * dim);
    dsret = (dsfloat *)aligned_alloc(32, sizeof(dsfloat) * dim);

//    printf("init_ddvector...dim = %d\n", dim);
	dsv_a = init_dsvector(dim); //printf("ddv_a ");
	dsv_b = init_dsvector(dim); //printf("ddv_b ");
	dsv_c = init_dsvector(dim); //printf("ddv_c ");
	dsv_ret = init_dsvector(dim); //printf("ddv_ret ");
//    printf("end init_ddvector...dim = %d\n", dim);

//    printf("set_ddvector...dim = %d\n", dim);
    set_test_dsvector(dsa, 5, dim);
    set_test_dsvector(dsb, 3, dim);
    set_test_dsvector(dsc, 7, dim);

    set_dsvector_dsfloat(dsv_a, dsa, dim);
    set_dsvector_dsfloat(dsv_b, dsb, dim);
    set_dsvector_dsfloat(dsv_c, dsc, dim);

    // ADD
    //printf("DSADD\n");
    //printf("Start _bncavx2_dsvadd(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsvadd(dsv_ret, dsv_a, dsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[0] = dim; ddtime[0] = etime / (double)itimes;
//    printf("end ... %d, ddvadd = %f\n", itimes, etime / (double)itimes);
    set_dsfloat_dsvec(dsret, dim, dsv_ret);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsadd(dsc, dsa, dsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[1] = dim; ddtime[1] = etime / (double)itimes;
  //  printf("end ... %d, dsadd = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsc, dim, 0);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rds_add(dsd[i].val, dsa[i].val, dsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[2] = dim; ddtime[2] = etime / (double)itimes;
    //printf("end ... %d, rdd_add = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsd, dim, 0); // no printing

   // MUL(d * d)vec
   // MUL(d * d)vec
    //printf("Start _bncavx2_dsvmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsvmul(dsv_ret, dsv_a, dsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[3] = dim; ddtime[3] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);
/*    stime = get_secv();
    _bncavx2_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
    etime = get_secv();
    printf("end ... %f(avx, vec) ", etime - stime);
*/
    set_dsfloat_dsvec(dsret, dim, dsv_ret);

    // dsmul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsmul(dsc, dsa, dsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[4] = dim; ddtime[4] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsc, dim, 0);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rds_mul(dsd[i].val, dsa[i].val, dsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[5] = dim; ddtime[5] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsd, dim, 0);

    //printf("Start _bncavx2_dsvdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsvdiv(dsv_ret, dsv_a, dsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[6] = dim; ddtime[6] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_dsfloat_dsvec(dsret, dim, dsv_ret);

    // dsdiv
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_dsdiv(dsc, dsa, dsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[7] = dim; ddtime[7] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsc, dim, 0);

    // test normal div
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rds_div(dsd[i].val, dsa[i].val, dsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[8] = dim; ddtime[8] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    dsrel_diff_array(dsret, dsd, dim, 0);

    //printf("dim=%d ddv_add ddadd rdd_add\n", dim);
    //printf("sec    %10.3g %10.3g %10.3g\n", ddtime[0], ddtime[1], ddtime[2]);
    //printf("mflops %10.3g %10.3g %10.3g\n", (double)dditer[0] / ddtime[0] / (double)(1024 * 1024), (double)dditer[1] / ddtime[1] / (double)(1024 * 1024), (double)dditer[2] / ddtime[2] / (double)(1024 * 1024));
    printf("%10d ", dim);
//    printf("%10.3g ", ddtime[0]);
//    printf("%10.3g ", ddtime[1]);
//    printf("%10.3g ", ddtime[2]);
//    printf("%10.3g ", ddtime[3]);
//    printf("%10.3g ", ddtime[4]);
//    printf("%10.3g ", ddtime[5]);
    printf("%10.3f ", mflops(dditer[0], ddtime[0]));
    printf("%10.3f ", mflops(dditer[1], ddtime[1]));
    printf("%10.3f ", mflops(dditer[2], ddtime[2]));
    printf("%10.3f ", mflops(dditer[3], ddtime[3]));
    printf("%10.3f ", mflops(dditer[4], ddtime[4]));
    printf("%10.3f ", mflops(dditer[5], ddtime[5]));
    printf("%10.3f ", mflops(dditer[6], ddtime[6]));
    printf("%10.3f ", mflops(dditer[7], ddtime[7]));
    printf("%10.3f\n", mflops(dditer[8], ddtime[8]));
//    printf("dim=%d ddv_mul ddmul rdd_mul\n", dim);
//    printf("sec    %10.3g %10.3g %10.3g\n", ddtime[3], ddtime[4], ddtime[5]);
//    printf("gflops %10.3g %10.3g %10.3g\n", (double)dditer[3] / ddtime[3] / (double)(1024 * 1024), (double)dditer[4] / ddtime[4] / (double)(1024 * 1024), (double)dditer[5] / ddtime[2] / (double)(1024 * 1024));

    // free
    free(dsret);
    free(dsa);
    free(dsb);
    free(dsc);
    free(dsd);
    free_dsvector(dsv_a);
    free_dsvector(dsv_b);
    free_dsvector(dsv_c);
    free_dsvector(dsv_ret);
    } // for(dim = 10000; dim <= 10000000; dim *= 10)

    return 0;
#endif //defined(__AVX2__) && !defined(__AVX512F__)
#endif //DSVEC 

#ifdef TSVEC
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 --\n");
    printf("TSVEC ... dim = %d\n", dim);
    printf("                           MFLOPS\n");
#ifdef USE_RTS_ADD
    printf("       dim     tsvaddt     tsaddt   rts_addt    tsvmul      tsmul    rts_mul     tsvdiv      tsdiv    rts_div\n");
#else // USE_RTS_ADD
    printf("       dim     tsvadd      tsadd    rts_add     tsvmul      tsmul    rts_mul     tsvdiv      tsdiv    rts_div\n");
#endif // USE_RTD_ADD
//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
//  printf("        32   2190.374    812.698   1211.834   2592.405    846.281   1329.870    860.504    474.074    453.097
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;

    tsa = (tsfloat *)aligned_alloc(32, sizeof(tsfloat) * dim);
    tsb = (tsfloat *)aligned_alloc(32, sizeof(tsfloat) * dim);
    tsc = (tsfloat *)aligned_alloc(32, sizeof(tsfloat) * dim);
    tsd = (tsfloat *)aligned_alloc(32, sizeof(tsfloat) * dim);
    tsret = (tsfloat *)aligned_alloc(32, sizeof(tsfloat) * dim);

//    printf("init_ddvector...dim = %d\n", dim);
	tsv_a = init_tsvector(dim); //printf("ddv_a ");
	tsv_b = init_tsvector(dim); //printf("ddv_b ");
	tsv_c = init_tsvector(dim); //printf("ddv_c ");
	tsv_ret = init_tsvector(dim); //printf("ddv_ret ");
//    printf("end init_ddvector...dim = %d\n", dim);

//    printf("set_ddvector...dim = %d\n", dim);
    set_test_tsvector(tsa, 5, dim);
    set_test_tsvector(tsb, 3, dim);
    set_test_tsvector(tsc, 7, dim);

    set_tsvector_tsfloat(tsv_a, tsa, dim);
    set_tsvector_tsfloat(tsv_b, tsb, dim);
    set_tsvector_tsfloat(tsv_c, tsc, dim);

    // ADD
    //printf("DSADD\n");
    //printf("Start _bncavx2_dsvadd(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsvadd(tsv_ret, tsv_a, tsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[0] = dim; ddtime[0] = etime / (double)itimes;
//    printf("end ... %d, ddvadd = %f\n", itimes, etime / (double)itimes);
    set_tsfloat_tsvec(tsret, dim, tsv_ret);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsadd(tsc, tsa, tsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[1] = dim; ddtime[1] = etime / (double)itimes;
  //  printf("end ... %d, dsadd = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsc, dim, 0);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rts_add(tsd[i].val, tsa[i].val, tsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[2] = dim; ddtime[2] = etime / (double)itimes;
    //printf("end ... %d, rdd_add = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsd, dim, 0); // no printing

   // MUL(d * d)vec
   // MUL(d * d)vec
    //printf("Start _bncavx2_tsvmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsvmul(tsv_ret, tsv_a, tsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[3] = dim; ddtime[3] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);
/*    stime = get_secv();
    _bncavx2_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
    etime = get_secv();
    printf("end ... %f(avx, vec) ", etime - stime);
*/
    set_tsfloat_tsvec(tsret, dim, tsv_ret);

    // dsmul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsmul(tsc, tsa, tsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[4] = dim; ddtime[4] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsc, dim, 0);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rts_mul(tsd[i].val, tsa[i].val, tsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[5] = dim; ddtime[5] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsd, dim, 0);

    //printf("Start _bncavx2_tsvdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsvdiv(tsv_ret, tsv_a, tsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[6] = dim; ddtime[6] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_tsfloat_tsvec(tsret, dim, tsv_ret);

    // tsdiv
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_tsdiv(tsc, tsa, tsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[7] = dim; ddtime[7] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsc, dim, 0);

    // test normal div
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rts_div(tsd[i].val, tsa[i].val, tsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[8] = dim; ddtime[8] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    tsrel_diff_array(tsret, tsd, dim, 0);

    //printf("dim=%d ddv_add ddadd rdd_add\n", dim);
    //printf("sec    %10.3g %10.3g %10.3g\n", ddtime[0], ddtime[1], ddtime[2]);
    //printf("mflops %10.3g %10.3g %10.3g\n", (double)dditer[0] / ddtime[0] / (double)(1024 * 1024), (double)dditer[1] / ddtime[1] / (double)(1024 * 1024), (double)dditer[2] / ddtime[2] / (double)(1024 * 1024));
    printf("%10d ", dim);
//    printf("%10.3g ", ddtime[0]);
//    printf("%10.3g ", ddtime[1]);
//    printf("%10.3g ", ddtime[2]);
//    printf("%10.3g ", ddtime[3]);
//    printf("%10.3g ", ddtime[4]);
//    printf("%10.3g ", ddtime[5]);
    printf("%10.3f ", mflops(dditer[0], ddtime[0]));
    printf("%10.3f ", mflops(dditer[1], ddtime[1]));
    printf("%10.3f ", mflops(dditer[2], ddtime[2]));
    printf("%10.3f ", mflops(dditer[3], ddtime[3]));
    printf("%10.3f ", mflops(dditer[4], ddtime[4]));
    printf("%10.3f ", mflops(dditer[5], ddtime[5]));
    printf("%10.3f ", mflops(dditer[6], ddtime[6]));
    printf("%10.3f ", mflops(dditer[7], ddtime[7]));
    printf("%10.3f\n", mflops(dditer[8], ddtime[8]));
//    printf("dim=%d ddv_mul ddmul rdd_mul\n", dim);
//    printf("sec    %10.3g %10.3g %10.3g\n", ddtime[3], ddtime[4], ddtime[5]);
//    printf("gflops %10.3g %10.3g %10.3g\n", (double)dditer[3] / ddtime[3] / (double)(1024 * 1024), (double)dditer[4] / ddtime[4] / (double)(1024 * 1024), (double)dditer[5] / ddtime[2] / (double)(1024 * 1024));

    // free
    free(tsret);
    free(tsa);
    free(tsb);
    free(tsc);
    free(tsd);
    free_tsvector(tsv_a);
    free_tsvector(tsv_b);
    free_tsvector(tsv_c);
    free_tsvector(tsv_ret);
    } // for(dim = 10000; dim <= 10000000; dim *= 10)

    return 0;
#endif //defined(__AVX2__) && !defined(__AVX512F__)
#endif //TSVEC 

#ifdef QSVEC
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 --\n");
    printf("QSVEC ... dim = %d\n", dim);
    printf("                           MFLOPS\n");
//    printf("       dim     ddvadd      ddadd    rdd_add     ddvmul      ddmul    rdd_mul\n");
    printf("       dim     qsvadd      qsadd    rqs_add     qsvmul      qsmul    rqs_mul     qsvdiv      qsdiv    rqs_div\n");
//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
//  printf("        32   2190.374    812.698   1211.834   2592.405    846.281   1329.870    860.504    474.074    453.097
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;

    qsa = (qsfloat *)aligned_alloc(32, sizeof(qsfloat) * dim);
    qsb = (qsfloat *)aligned_alloc(32, sizeof(qsfloat) * dim);
    qsc = (qsfloat *)aligned_alloc(32, sizeof(qsfloat) * dim);
    qsd = (qsfloat *)aligned_alloc(32, sizeof(qsfloat) * dim);
    qsret = (qsfloat *)aligned_alloc(32, sizeof(qsfloat) * dim);

//    printf("init_ddvector...dim = %d\n", dim);
	qsv_a = init_qsvector(dim); //printf("ddv_a ");
	qsv_b = init_qsvector(dim); //printf("ddv_b ");
	qsv_c = init_qsvector(dim); //printf("ddv_c ");
	qsv_ret = init_qsvector(dim); //printf("ddv_ret ");
//    printf("end init_ddvector...dim = %d\n", dim);

//    printf("set_ddvector...dim = %d\n", dim);
    set_test_qsvector(qsa, 5, dim);
    set_test_qsvector(qsb, 3, dim);
    set_test_qsvector(qsc, 7, dim);

    set_qsvector_qsfloat(qsv_a, qsa, dim);
    set_qsvector_qsfloat(qsv_b, qsb, dim);
    set_qsvector_qsfloat(qsv_c, qsc, dim);

    // ADD
    //printf("DSADD\n");
    //printf("Start _bncavx2_dsvadd(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsvadd(qsv_ret, qsv_a, qsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[0] = dim; ddtime[0] = etime / (double)itimes;
//    printf("end ... %d, ddvadd = %f\n", itimes, etime / (double)itimes);
    set_qsfloat_qsvec(qsret, dim, qsv_ret);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsadd(qsc, qsa, qsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[1] = dim; ddtime[1] = etime / (double)itimes;
  //  printf("end ... %d, dsadd = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsc, dim, 0);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqs_add(qsd[i].val, qsa[i].val, qsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[2] = dim; ddtime[2] = etime / (double)itimes;
    //printf("end ... %d, rdd_add = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsd, dim, 0); // no printing

   // MUL(d * d)vec
   // MUL(d * d)vec
    //printf("Start _bncavx2_tsvmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsvmul(qsv_ret, qsv_a, qsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[3] = dim; ddtime[3] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);
/*    stime = get_secv();
    _bncavx2_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
    etime = get_secv();
    printf("end ... %f(avx, vec) ", etime - stime);
*/
    set_qsfloat_qsvec(qsret, dim, qsv_ret);

    // dsmul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsmul(qsc, qsa, qsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[4] = dim; ddtime[4] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsc, dim, 0);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqs_mul(qsd[i].val, qsa[i].val, qsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[5] = dim; ddtime[5] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsd, dim, 0);

    //printf("Start _bncavx2_tsvdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsvdiv(qsv_ret, qsv_a, qsv_b, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[6] = dim; ddtime[6] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_qsfloat_qsvec(qsret, dim, qsv_ret);

    // tsdiv
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            _bncavx2_qsdiv(qsc, qsa, qsb, dim);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[7] = dim; ddtime[7] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsc, dim, 0);

    // test normal div
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqs_div(qsd[i].val, qsa[i].val, qsb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[8] = dim; ddtime[8] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qsrel_diff_array(qsret, qsd, dim, 0);

    //printf("dim=%d ddv_add ddadd rdd_add\n", dim);
    //printf("sec    %10.3g %10.3g %10.3g\n", ddtime[0], ddtime[1], ddtime[2]);
    //printf("mflops %10.3g %10.3g %10.3g\n", (double)dditer[0] / ddtime[0] / (double)(1024 * 1024), (double)dditer[1] / ddtime[1] / (double)(1024 * 1024), (double)dditer[2] / ddtime[2] / (double)(1024 * 1024));
    printf("%10d ", dim);
//    printf("%10.3g ", ddtime[0]);
//    printf("%10.3g ", ddtime[1]);
//    printf("%10.3g ", ddtime[2]);
//    printf("%10.3g ", ddtime[3]);
//    printf("%10.3g ", ddtime[4]);
//    printf("%10.3g ", ddtime[5]);
    printf("%10.3f ", mflops(dditer[0], ddtime[0]));
    printf("%10.3f ", mflops(dditer[1], ddtime[1]));
    printf("%10.3f ", mflops(dditer[2], ddtime[2]));
    printf("%10.3f ", mflops(dditer[3], ddtime[3]));
    printf("%10.3f ", mflops(dditer[4], ddtime[4]));
    printf("%10.3f ", mflops(dditer[5], ddtime[5]));
    printf("%10.3f ", mflops(dditer[6], ddtime[6]));
    printf("%10.3f ", mflops(dditer[7], ddtime[7]));
    printf("%10.3f\n", mflops(dditer[8], ddtime[8]));
//    printf("dim=%d ddv_mul ddmul rdd_mul\n", dim);
//    printf("sec    %10.3g %10.3g %10.3g\n", ddtime[3], ddtime[4], ddtime[5]);
//    printf("gflops %10.3g %10.3g %10.3g\n", (double)dditer[3] / ddtime[3] / (double)(1024 * 1024), (double)dditer[4] / ddtime[4] / (double)(1024 * 1024), (double)dditer[5] / ddtime[2] / (double)(1024 * 1024));

    // free
    free(qsret);
    free(qsa);
    free(qsb);
    free(qsc);
    free(qsd);
    free_qsvector(qsv_a);
    free_qsvector(qsv_b);
    free_qsvector(qsv_c);
    free_qsvector(qsv_ret);
    } // for(dim = 10000; dim <= 10000000; dim *= 10)

    return 0;
#endif //defined(__AVX2__) && !defined(__AVX512F__)
#endif //QSVEC 


#ifdef DDVEC
// AVX2 or AVX-512Alignment
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 -- \n");
#elif defined(__AVX512F__)
    printf("-- AVX-512 --\n");
#endif // defined(__AVX2__)
    printf("DDVEC ... dim = %d\n", dim);
    printf("                           MFLOPS\n");
//    printf("       dim     ddvadd      ddadd    rdd_add     ddvmul      ddmul    rdd_mul\n");
    printf("       dim     ddvadd      ddadd    rdd_add     ddvmul      ddmul    rdd_mul     ddvdiv      dddiv    rdd_div\n");
//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
//  printf("        32   2190.374    812.698   1211.834   2592.405    846.281   1329.870    860.504    474.074    453.097
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;

#if defined(__AVX2__) && !defined(__AVX512F__)
        dda = (ddfloat *)aligned_alloc(32, sizeof(ddfloat) * dim);
        ddb = (ddfloat *)aligned_alloc(32, sizeof(ddfloat) * dim);
        ddc = (ddfloat *)aligned_alloc(32, sizeof(ddfloat) * dim);
        ddd = (ddfloat *)aligned_alloc(32, sizeof(ddfloat) * dim);
        ddret = (ddfloat *)aligned_alloc(32, sizeof(ddfloat) * dim);
#elif defined(__AVX512F__)
        dda = (ddfloat *)aligned_alloc(64, sizeof(ddfloat) * dim);
        ddb = (ddfloat *)aligned_alloc(64, sizeof(ddfloat) * dim);
        ddc = (ddfloat *)aligned_alloc(64, sizeof(ddfloat) * dim);
        ddd = (ddfloat *)aligned_alloc(64, sizeof(ddfloat) * dim);
        ddret = (ddfloat *)aligned_alloc(64, sizeof(ddfloat) * dim);
#endif // defined(__AVX2__) && !defined(__AVX512F__)

    //    printf("init_ddvector...dim = %d\n", dim);
        ddv_a = init_ddvector(dim); //printf("ddv_a ");
        ddv_b = init_ddvector(dim); //printf("ddv_b ");
        ddv_c = init_ddvector(dim); //printf("ddv_c ");
        ddv_ret = init_ddvector(dim); //printf("ddv_ret ");
    //    printf("end init_ddvector...dim = %d\n", dim);

    //    printf("set_ddvector...dim = %d\n", dim);
        set_test_ddvector(dda, 5, dim);
        set_test_ddvector(ddb, 3, dim);
        set_test_ddvector(ddc, 7, dim);

        set_ddvector_ddfloat(ddv_a, dda, dim);
        set_ddvector_ddfloat(ddv_b, ddb, dim);
        set_ddvector_ddfloat(ddv_c, ddc, dim);

    // ADD
//    printf("DDADD\n");
//    printf("Start _bncavx2_ddvadd(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_ddvadd(ddv_ret, ddv_a, ddv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_ddvadd(ddv_ret, ddv_a, ddv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)    
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[0] = dim; ddtime[0] = etime / (double)itimes;
//    printf("end ... %d, ddvadd = %f\n", itimes, etime / (double)itimes);
    set_ddfloat_ddvec(ddret, dim, ddv_ret);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_ddadd(ddc, dda, ddb, dim);
#elif defined(__AVX512F__)
            _bncavx512_ddadd(ddc, dda, ddb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[1] = dim; ddtime[1] = etime / (double)itimes;
  //  printf("end ... %d, ddadd = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddc, dim, 0);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rdd_add(ddd[i].val, dda[i].val, ddb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[2] = dim; ddtime[2] = etime / (double)itimes;
    //printf("end ... %d, rdd_add = %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddd, dim, 0); // no printing

   // MUL(d * d)vec
   // MUL(d * d)vec
    //printf("Start _bncavx2_ddvmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[3] = dim; ddtime[3] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);
/*    stime = get_secv();
    _bncavx2_ddvmul(ddv_ret, ddv_a, ddv_b, dim);
    etime = get_secv();
    printf("end ... %f(avx, vec) ", etime - stime);
*/
    set_ddfloat_ddvec(ddret, dim, ddv_ret);

    // ddmul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_ddmul(ddc, dda, ddb, dim);
#elif defined(__AVX512F__)
            _bncavx512_ddmul(ddc, dda, ddb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[4] = dim; ddtime[4] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddc, dim, 0);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rdd_mul(ddd[i].val, dda[i].val, ddb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[5] = dim; ddtime[5] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddd, dim, 0);

    //printf("Start _bncavx2_ddvdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_ddvdiv(ddv_ret, ddv_a, ddv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_ddvdiv(ddv_ret, ddv_a, ddv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[6] = dim; ddtime[6] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_ddfloat_ddvec(ddret, dim, ddv_ret);

    // dddiv
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_dddiv(ddc, dda, ddb, dim);
#elif defined(__AVX512F__)
            _bncavx512_dddiv(ddc, dda, ddb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[7] = dim; ddtime[7] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddc, dim, 0);

    // test normal div
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rdd_div(ddd[i].val, dda[i].val, ddb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    dditer[8] = dim; ddtime[8] = etime / (double)itimes;
    //printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    ddrel_diff_array(ddret, ddd, dim, 0);

    //printf("dim=%d ddv_add ddadd rdd_add\n", dim);
    //printf("sec    %10.3g %10.3g %10.3g\n", ddtime[0], ddtime[1], ddtime[2]);
    //printf("mflops %10.3g %10.3g %10.3g\n", (double)dditer[0] / ddtime[0] / (double)(1024 * 1024), (double)dditer[1] / ddtime[1] / (double)(1024 * 1024), (double)dditer[2] / ddtime[2] / (double)(1024 * 1024));
    printf("%10d ", dim);
//    printf("%10.3g ", ddtime[0]);
//    printf("%10.3g ", ddtime[1]);
//    printf("%10.3g ", ddtime[2]);
//    printf("%10.3g ", ddtime[3]);
//    printf("%10.3g ", ddtime[4]);
//    printf("%10.3g ", ddtime[5]);
    printf("%10.3f ", mflops(dditer[0], ddtime[0]));
    printf("%10.3f ", mflops(dditer[1], ddtime[1]));
    printf("%10.3f ", mflops(dditer[2], ddtime[2]));
    printf("%10.3f ", mflops(dditer[3], ddtime[3]));
    printf("%10.3f ", mflops(dditer[4], ddtime[4]));
    printf("%10.3f ", mflops(dditer[5], ddtime[5]));
    printf("%10.3f ", mflops(dditer[6], ddtime[6]));
    printf("%10.3f ", mflops(dditer[7], ddtime[7]));
    printf("%10.3f\n", mflops(dditer[8], ddtime[8]));
//    printf("dim=%d ddv_mul ddmul rdd_mul\n", dim);
//    printf("sec    %10.3g %10.3g %10.3g\n", ddtime[3], ddtime[4], ddtime[5]);
//    printf("gflops %10.3g %10.3g %10.3g\n", (double)dditer[3] / ddtime[3] / (double)(1024 * 1024), (double)dditer[4] / ddtime[4] / (double)(1024 * 1024), (double)dditer[5] / ddtime[2] / (double)(1024 * 1024));

    // free
    free(ddret);
    free(dda);
    free(ddb);
    free(ddc);
    free(ddd);
    free_ddvector(ddv_a);
    free_ddvector(ddv_b);
    free_ddvector(ddv_c);
    free_ddvector(ddv_ret);
    } // for(dim = 10000; dim <= 10000000; dim *= 10)

    return 0;
#endif // DDVEC

// Triple-double
#ifdef TDVEC
// AVX2 or AVX-512Alignment
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 -- \n");
#elif defined(__AVX512F__)
    printf("-- AVX-512 --\n");
#endif // defined(__AVX2__)
    printf("TDVEC ... dim = %d\n", dim);
    printf("                           TD MFLOPS\n");

//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
    #ifdef USE_RTD_ADD
        printf("       dim     tdvadd      tdadd    rtd_add     tdvmul      tdmul    rtd_mul\n");
    #else // USE_RTD_ADD
        //printf("       dim    tdvaddq     tdaddq   rtd_addq     tdvmul      tdmul    rtd_mul\n");
        printf("       dim    tdvaddq     tdaddq   rtd_addq     tdvmul      tdmul    rtd_mul     tdvdivq     tddivq   rtd_divq\n");
    #endif // USE_RTD_ADD                
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;
#if defined(__AVX2__) && !defined(__AVX512F__)
        dta = (tdfloat *)aligned_alloc(32, sizeof(tdfloat) * dim);
        dtb = (tdfloat *)aligned_alloc(32, sizeof(tdfloat) * dim);
        dtc = (tdfloat *)aligned_alloc(32, sizeof(tdfloat) * dim);
        dtd = (tdfloat *)aligned_alloc(32, sizeof(tdfloat) * dim);
        dtret = (tdfloat *)aligned_alloc(32, sizeof(tdfloat) * dim);
#elif defined(__AVX512F__)
        dta = (tdfloat *)aligned_alloc(64, sizeof(tdfloat) * dim);
        dtb = (tdfloat *)aligned_alloc(64, sizeof(tdfloat) * dim);
        dtc = (tdfloat *)aligned_alloc(64, sizeof(tdfloat) * dim);
        dtd = (tdfloat *)aligned_alloc(64, sizeof(tdfloat) * dim);
        dtret = (tdfloat *)aligned_alloc(64, sizeof(tdfloat) * dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        dtv_a = init_tdvector(dim);
        dtv_b = init_tdvector(dim);
        dtv_c = init_tdvector(dim);
        dtv_ret = init_tdvector(dim);

    /*	for(i = 0; i < dim; i++)
        {
            dta[i].val[0] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtb[i].val[0] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtc[i].val[0] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dta[i].val[1] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtb[i].val[1] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtc[i].val[1] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dta[i].val[2] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtb[i].val[2] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
            dtc[i].val[2] = (double)rand() / (double)RAND_MAX * ((rand() % 2) ? -1.0 : 1.0);
        }
    */
        set_test_tdvector(dta, 5, dim);
        set_test_tdvector(dtb, 3, dim);
        set_test_tdvector(dtc, 7, dim);

        set_tdvector_tdfloat(dtv_a, dta, dim);
        set_tdvector_tdfloat(dtv_b, dtb, dim);
        set_tdvector_tdfloat(dtv_c, dtc, dim);

        // ADDv
    //    printf("TDADD\n");
    //    printf("Start _bncavx2_tdvadd(dim = %d) ... ", dim);
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
#if defined(__AVX2__) && !defined(__AVX512F__)
                _bncavx2_tdvadd(dtv_ret, dtv_a, dtv_b, dim);
#elif defined(__AVX512F__)
                _bncavx512_tdvadd(dtv_ret, dtv_a, dtv_b, dim);
#endif // defined(__AVX2__) && !defined(__AVX512F__)

            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[0] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        set_tdfloat_tdvec(dtret, dim, dtv_ret);
        // test normal add
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
                _bncavx2_tdadd(dtc, dta, dtb, dim);
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[1] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        tdrel_diff_array(dtret, dtc, dim, 0);

        // test normal add
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
                for(i = 0; i < dim; i++)
                {
    #ifdef USE_RTD_ADD
                    rtd_addt(dtd[i].val, dta[i].val, dtb[i].val);
    #else // USE_RTD_ADD
                    rtd_addq(dtd[i].val, dta[i].val, dtb[i].val);
    #endif // USE_RTD_ADD                
                }

            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[2] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        tdrel_diff_array(dtret, dtd, dim, 0);

    // MUL(d * d)vec
    //    printf("Start _bncavx2_tdvmul(dim = %d) ... ", dim);
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
#if defined(__AVX2__) && !defined(__AVX512F__)
                _bncavx2_tdvmul(dtv_ret, dtv_a, dtv_b, dim);
#elif defined(__AVX512F__)
                _bncavx512_tdvmul(dtv_ret, dtv_a, dtv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[3] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        set_tdfloat_tdvec(dtret, dim, dtv_ret);

        // test normal mul
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
#if defined(__AVX2__) && !defined(__AVX512F__)
                _bncavx2_tdmul(dtc, dta, dtb, dim);
#elif defined(__AVX512F__)
                _bncavx512_tdmul(dtc, dta, dtb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[4] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        tdrel_diff_array(dtret, dtc, dim, 0);

        // test normal mul
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
                for(i = 0; i < dim; i++)
                    rtd_mul(dtd[i].val, dta[i].val, dtb[i].val);
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[5] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        tdrel_diff_array(dtret, dtd, dim, 0);

    // DIV
    //    printf("Start _bncavx2_tdvdiv(dim = %d) ... ", dim);
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
#if defined(__AVX2__) && !defined(__AVX512F__)
                _bncavx2_tdvdiv(dtv_ret, dtv_a, dtv_b, dim);
#elif defined(__AVX512F__)
                _bncavx512_tdvdiv(dtv_ret, dtv_a, dtv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[6] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        set_tdfloat_tdvec(dtret, dim, dtv_ret);

        // test normal mul
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
#if defined(__AVX2__) && !defined(__AVX512F__)
                _bncavx2_tddiv(dtc, dta, dtb, dim);
#elif defined(__AVX512F__)
                _bncavx512_tddiv(dtc, dta, dtb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[7] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        tdrel_diff_array(dtret, dtc, dim, 0);

        // test normal mul
        max_itimes = 1;
        do {
            itimes = 0;
            stime = get_secv();
            do {
                for(i = 0; i < dim; i++)
    #ifdef USE_RTD_ADD
                    rtd_divt(dtd[i].val, dta[i].val, dtb[i].val);
    #else //ifdef USE_RTD_ADD
                    rtd_divtq(dtd[i].val, dta[i].val, dtb[i].val);
    #endif // ifdef USE_RTD_ADD
            } while(++itimes < max_itimes);
            etime = get_secv() - stime;
            max_itimes *= 2;
        } while(etime < 1.0);
        tdtime[8] = etime / (double)itimes;
    //    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

        // rel_diff: abs(dret - dd) / dd
        //printf("tdrel = ");
        tdrel_diff_array(dtret, dtd, dim, 0);

    #ifdef USE_RTD_ADD
        //printf("       dim     tdvadd      tdadd    rtd_add     tdvmul      tdmul    rtd_mul\n");
    #else // USE_RTD_ADD
        //printf("       dim    tdvaddq     tdaddq   rtd_addq     tdvmul      tdmul    rtd_mul\n");
        //printf("       dim    tdvaddq     tdaddq   rtd_addq     tdvmul      tdmul    rtd_mul     tdvdivq     tddivq   rtd_divq\n");
    #endif // USE_RTD_ADD                
        printf("%10d ", dim);
    //    printf("%10.3g ", tdtime[0]);
    //    printf("%10.3g ", tdtime[1]);
    //    printf("%10.3g ", tdtime[2]);
    //    printf("%10.3g ", tdtime[3]);
    //    printf("%10.3g ", tdtime[4]);
    //    printf("%10.3g ", tdtime[5]);
        printf("%10.3f ", mflops(dim, tdtime[0]));
        printf("%10.3f ", mflops(dim, tdtime[1]));
        printf("%10.3f ", mflops(dim, tdtime[2]));
        printf("%10.3f ", mflops(dim, tdtime[3]));
        printf("%10.3f ", mflops(dim, tdtime[4]));
        printf("%10.3f ", mflops(dim, tdtime[5]));
        printf("%10.3f ", mflops(dim, tdtime[6]));
        printf("%10.3f ", mflops(dim, tdtime[7]));
        printf("%10.3f\n", mflops(dim, tdtime[8]));

        // free
        free(dtret);
        free(dta);
        free(dtb);
        free(dtc);
        free(dtd);
        free_tdvector(dtv_a);
        free_tdvector(dtv_b);
        free_tdvector(dtv_c);
        free_tdvector(dtv_ret);
    }
    return 0;
#endif // TDVEC

#ifdef QDVEC
// AVX2 or AVX-512Alignment
#if defined(__AVX2__) && !defined(__AVX512F__)
    printf("-- AVX2 -- \n");
#elif defined(__AVX512F__)
    printf("-- AVX-512 --\n");
#endif // defined(__AVX2__)
    printf("QDVEC ... dim = %d\n", dim);
    printf("                           MFLOPS\n");
    printf("       dim     qdvadd      qdadd    rqd_add     qdvmul      qdmul    rqd_mul\n");
//  printf("      1024   1812.389    716.084   1055.670   1780.870    736.691   1211.834\n");
    for(dim = min_dim; dim <= 1024 * 1024;)
    {
        if(dim <= 128) dim_step = 16;
        else if(dim <= 256) dim_step = 64;
        else if(dim <= 512) dim_step = 128;
        else if(dim <= 1024) dim_step = 256;
        else dim_step = 1024;
        dim = (dim / dim_step) * dim_step;
        dim += dim_step;
#if defined(__AVX2__) && !defined(__AVX512F__)
        dqa = (qdfloat *)aligned_alloc(32, sizeof(qdfloat) * dim);
        dqb = (qdfloat *)aligned_alloc(32, sizeof(qdfloat) * dim);
        dqc = (qdfloat *)aligned_alloc(32, sizeof(qdfloat) * dim);
        dqd = (qdfloat *)aligned_alloc(32, sizeof(qdfloat) * dim);
        dqret = (qdfloat *)aligned_alloc(32, sizeof(qdfloat) * dim);
#elif defined(__AVX512F__)
        dqa = (qdfloat *)aligned_alloc(64, sizeof(qdfloat) * dim);
        dqb = (qdfloat *)aligned_alloc(64, sizeof(qdfloat) * dim);
        dqc = (qdfloat *)aligned_alloc(64, sizeof(qdfloat) * dim);
        dqd = (qdfloat *)aligned_alloc(64, sizeof(qdfloat) * dim);
        dqret = (qdfloat *)aligned_alloc(64, sizeof(qdfloat) * dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)

	dqv_a = init_qdvector(dim);
	dqv_b = init_qdvector(dim);
	dqv_c = init_qdvector(dim);
	dqv_ret = init_qdvector(dim);

    set_test_qdvector(dqa, 5, dim);
    set_test_qdvector(dqb, 3, dim);
    set_test_qdvector(dqc, 7, dim);

    set_qdvector_qdfloat(dqv_a, dqa, dim);
    set_qdvector_qdfloat(dqv_b, dqb, dim);
    set_qdvector_qdfloat(dqv_c, dqc, dim);

    // ADDv
//    printf("QDADD\n");
//    printf("Start _bncavx2_qdvadd(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qdvadd(dqv_ret, dqv_a, dqv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_qdvadd(dqv_ret, dqv_a, dqv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[0] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_qdfloat_qdvec(dqret, dim, dqv_ret);

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qdadd(dqc, dqa, dqb, dim);
#elif defined(__AVX512F__)
            _bncavx512_qdadd(dqc, dqa, dqb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[1] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqc, dim, 0);
#if defined(__AVX2__) && !defined(__AVX512F__)
    //printf("relerr(_bncavx2_qdadd) = %10.3e\n", dqret[0]);
#elif defined(__AVX512F__)
    //printf("relerr(_bncavx512_qdadd) = %10.3e\n", dqret[0]);
#endif // defined(__AVX2__) && !defined(__AVX512F__)

    // test normal add
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqd_add(dqd[i].val, dqa[i].val, dqb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[2] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqd, dim, 0);

   // MUL(d * d)vec
//    printf("Start _bncavx2_qdvmul(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qdvmul(dqv_ret, dqv_a, dqv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_qdvmul(dqv_ret, dqv_a, dqv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[3] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_qdfloat_qdvec(dqret, dim, dqv_ret);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qdmul(dqc, dqa, dqb, dim);
#elif defined(__AVX512F__)
            _bncavx512_qdmul(dqc, dqa, dqb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[4] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqc, dim, 0);
#if defined(__AVX2__) && !defined(__AVX512F__)
    //printf("relerr(_bncavx2_qdmul) = %10.3e\n", dqret[0]);
#elif defined(__AVX512F__)
    //printf("relerr(_bncavx512_qdmul) = %10.3e\n", dqret[0]);
#endif // defined(__AVX2__) && !defined(__AVX512F__)


    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqd_mul(dqd[i].val, dqa[i].val, dqb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[5] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqd, dim, 0);
    //printf("relerr(rqd_mul) = %10.3e\n", dqret[0]);

   // DIV
//    printf("Start _bncavx2_qdvdiv(dim = %d) ... ", dim);
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qdvdiv(dqv_ret, dqv_a, dqv_b, dim);
#elif defined(__AVX512F__)
            _bncavx512_qdvdiv(dqv_ret, dqv_a, dqv_b, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[6] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    set_qdfloat_qdvec(dqret, dim, dqv_ret);

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
#if defined(__AVX2__) && !defined(__AVX512F__)
            _bncavx2_qddiv(dqc, dqa, dqb, dim);
#elif defined(__AVX512F__)
            _bncavx512_qddiv(dqc, dqa, dqb, dim);
#endif //defined(__AVX2__) && !defined(__AVX512F__)
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[7] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqc, dim, 0);
#if defined(__AVX2__) && !defined(__AVX512F__)
    //printf("relerr(_bncavx2_qddiv) = %10.3e\n", dqret[0]);
#elif defined(__AVX512F__)
    //printf("relerr(_bncavx512_qddiv) = %10.3e\n", dqret[0]);
#endif // defined(__AVX2__) && !defined(__AVX512F__)

    // test normal mul
    max_itimes = 1;
    do {
        itimes = 0;
        stime = get_secv();
        do {
            for(i = 0; i < dim; i++)
                rqd_div(dqd[i].val, dqa[i].val, dqb[i].val);
        } while(++itimes < max_itimes);
        etime = get_secv() - stime;
        max_itimes *= 2;
    } while(etime < 1.0);
    qdtime[8] = etime / (double)itimes;
//    printf("end ... %d, %f\n", itimes, etime / (double)itimes);

    // rel_diff: abs(dret - dd) / dd
    qdrel_diff_array(dqret, dqd, dim, 0);
    //printf("relerr(rqd_div) = %10.3e\n", dqret[0]);

    printf("%10d ", dim);
//    printf("%10.3g ", qdtime[0]);
//    printf("%10.3g ", qdtime[1]);
//    printf("%10.3g ", qdtime[2]);
//    printf("%10.3g ", qdtime[3]);
//    printf("%10.3g ", qdtime[4]);
//    printf("%10.3g ", qdtime[5]);
    printf("%10.3f ", mflops(dim, qdtime[0]));
    printf("%10.3f ", mflops(dim, qdtime[1]));
    printf("%10.3f ", mflops(dim, qdtime[2]));
    printf("%10.3f ", mflops(dim, qdtime[3]));
    printf("%10.3f ", mflops(dim, qdtime[4]));
    printf("%10.3f ", mflops(dim, qdtime[5]));
    printf("%10.3f ", mflops(dim, qdtime[6]));
    printf("%10.3f ", mflops(dim, qdtime[7]));
    printf("%10.3f\n", mflops(dim, qdtime[8]));

    // free
    free(dqret);
    free(dqa);
    free(dqb);
    free(dqc);
    free(dqd);
    free_qdvector(dqv_a);
    free_qdvector(dqv_b);
    free_qdvector(dqv_c);
    free_qdvector(dqv_ret);
    }
    return 0;
#endif // QDVEC

    return 0;
}
