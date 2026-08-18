/* qdlinear.c: Quadruple double precision Linear Computation Library
   (NEON最適化分岐を追加済み・コンパイル可能版)
   Copyright (C) 2020 Tomonori Kouya
   License: LGPL-3.0-or-later
*/
#include "qdlinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif

#if 0
#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #include "_bncneon_qd.h"
#endif
#endif // 0

#if defined(USE_GMP) && defined(USE_MPFR)
/* --- Frobenius norm for qdfloat array (MPFR sqrt) --- */
qdfloat qdnormf(qdfloat array[], int dim)
{
    int i;
    qdfloat ret, tmp;
    mpfr_t mpfr_ret;

    rqd_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++){
        rqd_mul(tmp.val, array[i].val, array[i].val);
        rqd_add(ret.val, ret.val, tmp.val);
    }
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_dd(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_dd(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}
#endif /* USE_GMP && USE_MPFR */

/* =========================
   QDVector: init / free
   ========================= */
QDVector init_qdvector(long int dimension)
{
    QDVector ret = NULL;
    long int i, real_dim;

    if(dimension <= 0){
        fprintf(stderr, "ERROR: init_qdvector\n");
        return NULL;
    }

    ret = (QDVector)malloc(sizeof(qdvector));
    if(!ret) return NULL;

    /* real_dim is the nearest positive multiplier of _BNC_D_WIDTH */
    real_dim = (long int)ceil((double)dimension / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

    ret->element[0] = (double *)BNC_CALLOC(real_dim, sizeof(double));
    if(!ret->element[0]){ free(ret); return NULL; }
    ret->element[1] = (double *)BNC_CALLOC(real_dim, sizeof(double));
    if(!ret->element[1]){ free(ret->element[0]); free(ret); return NULL; }
    ret->element[2] = (double *)BNC_CALLOC(real_dim, sizeof(double));
    if(!ret->element[2]){ free(ret->element[1]); free(ret->element[0]); free(ret); return NULL; }
    ret->element[3] = (double *)BNC_CALLOC(real_dim, sizeof(double));
    if(!ret->element[3]){ free(ret->element[2]); free(ret->element[1]); free(ret->element[0]); free(ret); return NULL; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d zero4 = _mm256_setzero_pd();
    for(i = 0; i < real_dim; i += _BNC_D_WIDTH){
        _mm256_store_pd(&(ret->element[0][i]), zero4);
        _mm256_store_pd(&(ret->element[1][i]), zero4);
        _mm256_store_pd(&(ret->element[2][i]), zero4);
        _mm256_store_pd(&(ret->element[3][i]), zero4);
    }
#elif defined(__AVX512F__)
    __m512d zero8 = _mm512_setzero_pd();
    for(i = 0; i < real_dim; i += _BNC_D_WIDTH){
        _mm512_store_pd(&(ret->element[0][i]), zero8);
        _mm512_store_pd(&(ret->element[1][i]), zero8);
        _mm512_store_pd(&(ret->element[2][i]), zero8);
        _mm512_store_pd(&(ret->element[3][i]), zero8);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t z = svdup_f64(0.0);
    for(i = 0; i < real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_dim));
        svst1_f64(pg, &(ret->element[0][i]), z);
        svst1_f64(pg, &(ret->element[1][i]), z);
        svst1_f64(pg, &(ret->element[2][i]), z);
        svst1_f64(pg, &(ret->element[3][i]), z);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
    float64x2_t z = vdupq_n_f64(0.0);
    for(i = 0; i < real_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&(ret->element[0][i]), z);
        vst1q_f64(&(ret->element[1][i]), z);
        vst1q_f64(&(ret->element[2][i]), z);
        vst1q_f64(&(ret->element[3][i]), z);
    }
#else
    for(i = 0; i < dimension; i++){
        ret->element[0][i] = 0.0;
        ret->element[1][i] = 0.0;
        ret->element[2][i] = 0.0;
        ret->element[3][i] = 0.0;
    }
#endif

    ret->dim = dimension;
    ret->real_dim = real_dim;
    return ret;
}

void free_qdvector(QDVector vec)
{
    long int i;
    if(!vec) return;
    for(i = 0; i < QDSIZE; i++) free(vec->element[i]);
    free(vec);
}

/* ----- helpers between qdfloat[] and QDVector ----- */
void set_qdfloat_qdvec(qdfloat ret[], int ret_dim, QDVector vec)
{
    int index, j, dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;
    for(index = 0; index < dim; index++)
        for(j = 0; j < QDSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
}
void set_qdvector_qdfloat(QDVector ret, qdfloat array[], int array_dim)
{
    int index, j, dim = (ret->dim < array_dim) ? ret->dim : array_dim;
    for(index = 0; index < dim; index++)
        for(j = 0; j < QDSIZE; j++)
            ret->element[j][index] = array[index].val[j];
}

/* =========================
   QDVector: utilities
   ========================= */
void print_qdvector(QDVector vec)
{
    long int index;
    for(index = 0; index < vec->dim; index++){
        printf("%4ld: ", index);
        rqd_out_str(GET_QDVECTOR_I(vec, index));
        printf("\n");
    }
}

void set0_qdvector(QDVector vec)
{
    long int i;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d z = _mm256_setzero_pd();
    for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH){
        _mm256_store_pd(&(vec->element[0][i]), z);
        _mm256_store_pd(&(vec->element[1][i]), z);
        _mm256_store_pd(&(vec->element[2][i]), z);
        _mm256_store_pd(&(vec->element[3][i]), z);
    }
#elif defined(__AVX512F__)
    __m512d z = _mm512_setzero_pd();
    for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH){
        _mm512_store_pd(&(vec->element[0][i]), z);
        _mm512_store_pd(&(vec->element[1][i]), z);
        _mm512_store_pd(&(vec->element[2][i]), z);
        _mm512_store_pd(&(vec->element[3][i]), z);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t z = svdup_f64(0.0);
    for(i = 0; i < vec->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(vec->real_dim));
        svst1_f64(pg, &(vec->element[0][i]), z);
        svst1_f64(pg, &(vec->element[1][i]), z);
        svst1_f64(pg, &(vec->element[2][i]), z);
        svst1_f64(pg, &(vec->element[3][i]), z);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
    float64x2_t z = vdupq_n_f64(0.0);
    for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&(vec->element[0][i]), z);
        vst1q_f64(&(vec->element[1][i]), z);
        vst1q_f64(&(vec->element[2][i]), z);
        vst1q_f64(&(vec->element[3][i]), z);
    }
#else
    for(i = 0; i < vec->dim; i++){
        vec->element[0][i] = 0.0;
        vec->element[1][i] = 0.0;
        vec->element[2][i] = 0.0;
        vec->element[3][i] = 0.0;
    }
#endif
}

void set_qdvector_i_str(QDVector vec, long int index, const char *str)
{
    double tmp[QDSIZE];
    rqd_set_str(tmp, str);
    set_qdvector_i(vec, index, tmp);
}

/* =========================
   QDVector: operations
   ========================= */
/* c = a + b */
void add_qdvector(QDVector c, QDVector a, QDVector b)
{
    long int i, index;
    if((a->dim != b->dim) || (c->dim != a->dim)){ fprintf(stderr,"ERROR: add_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        bx[0]=_mm256_load_pd(&(b->element[0][index])); bx[1]=_mm256_load_pd(&(b->element[1][index]));
        bx[2]=_mm256_load_pd(&(b->element[2][index])); bx[3]=_mm256_load_pd(&(b->element[3][index]));
        _bncavx2_rqd_add(r, ax, bx);
        _mm256_store_pd(&(c->element[0][index]), r[0]);
        _mm256_store_pd(&(c->element[1][index]), r[1]);
        _mm256_store_pd(&(c->element[2][index]), r[2]);
        _mm256_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__AVX512F__)
    __m512d r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        bx[0]=_mm512_load_pd(&(b->element[0][index])); bx[1]=_mm512_load_pd(&(b->element[1][index]));
        bx[2]=_mm512_load_pd(&(b->element[2][index])); bx[3]=_mm512_load_pd(&(b->element[3][index]));
        _bncavx512_rqd_add(r, ax, bx);
        _mm512_store_pd(&(c->element[0][index]), r[0]);
        _mm512_store_pd(&(c->element[1][index]), r[1]);
        _mm512_store_pd(&(c->element[2][index]), r[2]);
        _mm512_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t r_0, r_1, r_2, r_3;
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t bx_0, bx_1, bx_2, bx_3;
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        bx_0=svld1_f64(pg, &(b->element[0][index])); bx_1=svld1_f64(pg, &(b->element[1][index]));
        bx_2=svld1_f64(pg, &(b->element[2][index])); bx_3=svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_add(pg, &r_0, &r_1, &r_2, &r_3, ax_0, ax_1, ax_2, ax_3, bx_0, bx_1, bx_2, bx_3);
        svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
        svst1_f64(pg, &(c->element[2][index]), r_2);
        svst1_f64(pg, &(c->element[3][index]), r_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        bx[0]=vld1q_f64(&(b->element[0][index])); bx[1]=vld1q_f64(&(b->element[1][index]));
        bx[2]=vld1q_f64(&(b->element[2][index])); bx[3]=vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_add(r, ax, bx);
        vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
        vst1q_f64(&(c->element[2][index]), r[2]);
        vst1q_f64(&(c->element[3][index]), r[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_add(tmp, get_qdvector_i(a, i), get_qdvector_i(b, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c += a */
void add2_qdvector(QDVector c, QDVector a)
{
    long int i, index;
    if(c->dim != a->dim){ fprintf(stderr,"ERROR: add2_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        cx[0]=_mm256_load_pd(&(c->element[0][index])); cx[1]=_mm256_load_pd(&(c->element[1][index]));
        cx[2]=_mm256_load_pd(&(c->element[2][index])); cx[3]=_mm256_load_pd(&(c->element[3][index]));
        _bncavx2_rqd_add(r, cx, ax);
        _mm256_store_pd(&(c->element[0][index]), r[0]);
        _mm256_store_pd(&(c->element[1][index]), r[1]);
        _mm256_store_pd(&(c->element[2][index]), r[2]);
        _mm256_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__AVX512F__)
    __m512d r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        cx[0]=_mm512_load_pd(&(c->element[0][index])); cx[1]=_mm512_load_pd(&(c->element[1][index]));
        cx[2]=_mm512_load_pd(&(c->element[2][index])); cx[3]=_mm512_load_pd(&(c->element[3][index]));
        _bncavx512_rqd_add(r, cx, ax);
        _mm512_store_pd(&(c->element[0][index]), r[0]);
        _mm512_store_pd(&(c->element[1][index]), r[1]);
        _mm512_store_pd(&(c->element[2][index]), r[2]);
        _mm512_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t r_0, r_1, r_2, r_3;
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        cx_0=svld1_f64(pg, &(c->element[0][index])); cx_1=svld1_f64(pg, &(c->element[1][index]));
        cx_2=svld1_f64(pg, &(c->element[2][index])); cx_3=svld1_f64(pg, &(c->element[3][index]));
        _bncsve2_rqd_add(pg, &r_0, &r_1, &r_2, &r_3, cx_0, cx_1, cx_2, cx_3, ax_0, ax_1, ax_2, ax_3);
        svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
        svst1_f64(pg, &(c->element[2][index]), r_2);
        svst1_f64(pg, &(c->element[3][index]), r_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        cx[0]=vld1q_f64(&(c->element[0][index])); cx[1]=vld1q_f64(&(c->element[1][index]));
        cx[2]=vld1q_f64(&(c->element[2][index])); cx[3]=vld1q_f64(&(c->element[3][index]));
        _bncneon_rqd_add(r, cx, ax);
        vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
        vst1q_f64(&(c->element[2][index]), r[2]);
        vst1q_f64(&(c->element[3][index]), r[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_add(tmp, get_qdvector_i(c, i), get_qdvector_i(a, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c = a - b */
void sub_qdvector(QDVector c, QDVector a, QDVector b)
{
    long int i, index;
    if((a->dim != b->dim) || (c->dim != a->dim)){ fprintf(stderr,"ERROR: sub_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        bx[0]=_mm256_load_pd(&(b->element[0][index])); bx[1]=_mm256_load_pd(&(b->element[1][index]));
        bx[2]=_mm256_load_pd(&(b->element[2][index])); bx[3]=_mm256_load_pd(&(b->element[3][index]));
        _bncavx2_rqd_sub(r, ax, bx);
        _mm256_store_pd(&(c->element[0][index]), r[0]);
        _mm256_store_pd(&(c->element[1][index]), r[1]);
        _mm256_store_pd(&(c->element[2][index]), r[2]);
        _mm256_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__AVX512F__)
    __m512d r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        bx[0]=_mm512_load_pd(&(b->element[0][index])); bx[1]=_mm512_load_pd(&(b->element[1][index]));
        bx[2]=_mm512_load_pd(&(b->element[2][index])); bx[3]=_mm512_load_pd(&(b->element[3][index]));
        _bncavx512_rqd_sub(r, ax, bx);
        _mm512_store_pd(&(c->element[0][index]), r[0]);
        _mm512_store_pd(&(c->element[1][index]), r[1]);
        _mm512_store_pd(&(c->element[2][index]), r[2]);
        _mm512_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t r_0, r_1, r_2, r_3;
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t bx_0, bx_1, bx_2, bx_3;
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        bx_0=svld1_f64(pg, &(b->element[0][index])); bx_1=svld1_f64(pg, &(b->element[1][index]));
        bx_2=svld1_f64(pg, &(b->element[2][index])); bx_3=svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_neg(pg, &r_0, &r_1, &r_2, &r_3, bx_0, bx_1, bx_2, bx_3);
		_bncsve2_rqd_add(pg, &r_0, &r_1, &r_2, &r_3, ax_0, ax_1, ax_2, ax_3, r_0, r_1, r_2, r_3);
        svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
        svst1_f64(pg, &(c->element[2][index]), r_2);
        svst1_f64(pg, &(c->element[3][index]), r_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t r[QDSIZE], ax[QDSIZE], bx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        bx[0]=vld1q_f64(&(b->element[0][index])); bx[1]=vld1q_f64(&(b->element[1][index]));
        bx[2]=vld1q_f64(&(b->element[2][index])); bx[3]=vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_sub(r, ax, bx);
        vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
        vst1q_f64(&(c->element[2][index]), r[2]);
        vst1q_f64(&(c->element[3][index]), r[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_sub(tmp, get_qdvector_i(a, i), get_qdvector_i(b, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c -= a */
void sub2_qdvector(QDVector c, QDVector a)
{
    long int i, index;
    if(c->dim != a->dim){ fprintf(stderr,"ERROR: sub2_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        cx[0]=_mm256_load_pd(&(c->element[0][index])); cx[1]=_mm256_load_pd(&(c->element[1][index]));
        cx[2]=_mm256_load_pd(&(c->element[2][index])); cx[3]=_mm256_load_pd(&(c->element[3][index]));
        _bncavx2_rqd_sub(r, cx, ax);
        _mm256_store_pd(&(c->element[0][index]), r[0]);
        _mm256_store_pd(&(c->element[1][index]), r[1]);
        _mm256_store_pd(&(c->element[2][index]), r[2]);
        _mm256_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__AVX512F__)
    __m512d r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        cx[0]=_mm512_load_pd(&(c->element[0][index])); cx[1]=_mm512_load_pd(&(c->element[1][index]));
        cx[2]=_mm512_load_pd(&(c->element[2][index])); cx[3]=_mm512_load_pd(&(c->element[3][index]));
        _bncavx512_rqd_sub(r, cx, ax);
        _mm512_store_pd(&(c->element[0][index]), r[0]);
        _mm512_store_pd(&(c->element[1][index]), r[1]);
        _mm512_store_pd(&(c->element[2][index]), r[2]);
        _mm512_store_pd(&(c->element[3][index]), r[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t r_0, r_1, r_2, r_3;
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        cx_0=svld1_f64(pg, &(c->element[0][index])); cx_1=svld1_f64(pg, &(c->element[1][index]));
        cx_2=svld1_f64(pg, &(c->element[2][index])); cx_3=svld1_f64(pg, &(c->element[3][index]));
        _bncsve2_rqd_neg(pg, &r_0, &r_1, &r_2, &r_3, ax_0, ax_1, ax_2, ax_3);
		_bncsve2_rqd_add(pg, &r_0, &r_1, &r_2, &r_3, cx_0, cx_1, cx_2, cx_3, r_0, r_1, r_2, r_3);
        svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
        svst1_f64(pg, &(c->element[2][index]), r_2);
        svst1_f64(pg, &(c->element[3][index]), r_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t r[QDSIZE], ax[QDSIZE], cx[QDSIZE];
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        cx[0]=vld1q_f64(&(c->element[0][index])); cx[1]=vld1q_f64(&(c->element[1][index]));
        cx[2]=vld1q_f64(&(c->element[2][index])); cx[3]=vld1q_f64(&(c->element[3][index]));
        _bncneon_rqd_sub(r, cx, ax);
        vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
        vst1q_f64(&(c->element[2][index]), r[2]);
        vst1q_f64(&(c->element[3][index]), r[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_sub(tmp, get_qdvector_i(c, i), get_qdvector_i(a, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c = val * a */
void cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
{
    long int i, index;
    if(c->dim != a->dim){ fprintf(stderr,"ERROR: cmul_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d ax[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm256_set1_pd(val[0]); vx[1]=_mm256_set1_pd(val[1]);
    vx[2]=_mm256_set1_pd(val[2]); vx[3]=_mm256_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        _bncavx2_rqd_mul(cx, vx, ax);
        _mm256_store_pd(&(c->element[0][index]), cx[0]);
        _mm256_store_pd(&(c->element[1][index]), cx[1]);
        _mm256_store_pd(&(c->element[2][index]), cx[2]);
        _mm256_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__AVX512F__)
    __m512d ax[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm512_set1_pd(val[0]); vx[1]=_mm512_set1_pd(val[1]);
    vx[2]=_mm512_set1_pd(val[2]); vx[3]=_mm512_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        _bncavx512_rqd_mul(cx, vx, ax);
        _mm512_store_pd(&(c->element[0][index]), cx[0]);
        _mm512_store_pd(&(c->element[1][index]), cx[1]);
        _mm512_store_pd(&(c->element[2][index]), cx[2]);
        _mm512_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
    vx_0=svdup_f64(val[0]); vx_1=svdup_f64(val[1]);
    vx_2=svdup_f64(val[2]); vx_3=svdup_f64(val[3]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        _bncsve2_rqd_mul(pg, &cx_0, &cx_1, &cx_2, &cx_3, vx_0, vx_1, vx_2, vx_3, ax_0, ax_1, ax_2, ax_3);
        svst1_f64(pg, &(c->element[0][index]), cx_0);
        svst1_f64(pg, &(c->element[1][index]), cx_1);
        svst1_f64(pg, &(c->element[2][index]), cx_2);
        svst1_f64(pg, &(c->element[3][index]), cx_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t ax[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=vdupq_n_f64(val[0]); vx[1]=vdupq_n_f64(val[1]);
    vx[2]=vdupq_n_f64(val[2]); vx[3]=vdupq_n_f64(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        _bncneon_rqd_mul(cx, vx, ax);
        vst1q_f64(&(c->element[0][index]), cx[0]);
        vst1q_f64(&(c->element[1][index]), cx[1]);
        vst1q_f64(&(c->element[2][index]), cx[2]);
        vst1q_f64(&(c->element[3][index]), cx[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_mul(tmp, val, get_qdvector_i(a, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c *= val */
void cmul2_qdvector(QDVector c, double val[QDSIZE])
{
    long int i, index;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d cx[QDSIZE], vx[QDSIZE], t[QDSIZE];
    vx[0]=_mm256_set1_pd(val[0]); vx[1]=_mm256_set1_pd(val[1]);
    vx[2]=_mm256_set1_pd(val[2]); vx[3]=_mm256_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        cx[0]=_mm256_load_pd(&(c->element[0][index])); cx[1]=_mm256_load_pd(&(c->element[1][index]));
        cx[2]=_mm256_load_pd(&(c->element[2][index])); cx[3]=_mm256_load_pd(&(c->element[3][index]));
        _bncavx2_rqd_mul(t, vx, cx);
        _mm256_store_pd(&(c->element[0][index]), t[0]);
        _mm256_store_pd(&(c->element[1][index]), t[1]);
        _mm256_store_pd(&(c->element[2][index]), t[2]);
        _mm256_store_pd(&(c->element[3][index]), t[3]);
    }
#elif defined(__AVX512F__)
    __m512d cx[QDSIZE], vx[QDSIZE], t[QDSIZE];
    vx[0]=_mm512_set1_pd(val[0]); vx[1]=_mm512_set1_pd(val[1]);
    vx[2]=_mm512_set1_pd(val[2]); vx[3]=_mm512_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        cx[0]=_mm512_load_pd(&(c->element[0][index])); cx[1]=_mm512_load_pd(&(c->element[1][index]));
        cx[2]=_mm512_load_pd(&(c->element[2][index])); cx[3]=_mm512_load_pd(&(c->element[3][index]));
        _bncavx512_rqd_mul(t, vx, cx);
        _mm512_store_pd(&(c->element[0][index]), t[0]);
        _mm512_store_pd(&(c->element[1][index]), t[1]);
        _mm512_store_pd(&(c->element[2][index]), t[2]);
        _mm512_store_pd(&(c->element[3][index]), t[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
	svfloat64_t t_0, t_1, t_2, t_3;
    vx_0=svdup_f64(val[0]); vx_1=svdup_f64(val[1]);
    vx_2=svdup_f64(val[2]); vx_3=svdup_f64(val[3]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        cx_0=svld1_f64(pg, &(c->element[0][index])); cx_1=svld1_f64(pg, &(c->element[1][index]));
        cx_2=svld1_f64(pg, &(c->element[2][index])); cx_3=svld1_f64(pg, &(c->element[3][index]));
        _bncsve2_rqd_mul(pg, &t_0, &t_1, &t_2, &t_3, vx_0, vx_1, vx_2, vx_3, cx_0, cx_1, cx_2, cx_3);
        svst1_f64(pg, &(c->element[0][index]), t_0);
        svst1_f64(pg, &(c->element[1][index]), t_1);
        svst1_f64(pg, &(c->element[2][index]), t_2);
        svst1_f64(pg, &(c->element[3][index]), t_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t cx[QDSIZE], vx[QDSIZE], t[QDSIZE];
    vx[0]=vdupq_n_f64(val[0]); vx[1]=vdupq_n_f64(val[1]);
    vx[2]=vdupq_n_f64(val[2]); vx[3]=vdupq_n_f64(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        cx[0]=vld1q_f64(&(c->element[0][index])); cx[1]=vld1q_f64(&(c->element[1][index]));
        cx[2]=vld1q_f64(&(c->element[2][index])); cx[3]=vld1q_f64(&(c->element[3][index]));
        _bncneon_rqd_mul(t, vx, cx);
        vst1q_f64(&(c->element[0][index]), t[0]);
        vst1q_f64(&(c->element[1][index]), t[1]);
        vst1q_f64(&(c->element[2][index]), t[2]);
        vst1q_f64(&(c->element[3][index]), t[3]);
    }
#else
    double tmp[QDSIZE];
    long int i2;
    for(i2 = 0; i2 < c->dim; i2++){
        rqd_mul(tmp, val, get_qdvector_i(c, i2));
        set_qdvector_i(c, i2, tmp);
    }
#endif
}

/* c = a + val * b */
void add_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b)
{
    long int index, i;
    if((a->dim != b->dim) || (c->dim != a->dim)){ fprintf(stderr,"ERROR: add_cmul_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm256_set1_pd(val[0]); vx[1]=_mm256_set1_pd(val[1]);
    vx[2]=_mm256_set1_pd(val[2]); vx[3]=_mm256_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        bx[0]=_mm256_load_pd(&(b->element[0][index])); bx[1]=_mm256_load_pd(&(b->element[1][index]));
        bx[2]=_mm256_load_pd(&(b->element[2][index])); bx[3]=_mm256_load_pd(&(b->element[3][index]));
        _bncavx2_rqd_mul(cx, vx, bx);
        _bncavx2_rqd_add(cx, ax, cx);
        _mm256_store_pd(&(c->element[0][index]), cx[0]);
        _mm256_store_pd(&(c->element[1][index]), cx[1]);
        _mm256_store_pd(&(c->element[2][index]), cx[2]);
        _mm256_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__AVX512F__)
    __m512d ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm512_set1_pd(val[0]); vx[1]=_mm512_set1_pd(val[1]);
    vx[2]=_mm512_set1_pd(val[2]); vx[3]=_mm512_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        bx[0]=_mm512_load_pd(&(b->element[0][index])); bx[1]=_mm512_load_pd(&(b->element[1][index]));
        bx[2]=_mm512_load_pd(&(b->element[2][index])); bx[3]=_mm512_load_pd(&(b->element[3][index]));
        _bncavx512_rqd_mul(cx, vx, bx);
        _bncavx512_rqd_add(cx, ax, cx);
        _mm512_store_pd(&(c->element[0][index]), cx[0]);
        _mm512_store_pd(&(c->element[1][index]), cx[1]);
        _mm512_store_pd(&(c->element[2][index]), cx[2]);
        _mm512_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t bx_0, bx_1, bx_2, bx_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
    vx_0=svdup_f64(val[0]); vx_1=svdup_f64(val[1]);
    vx_2=svdup_f64(val[2]); vx_3=svdup_f64(val[3]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        bx_0=svld1_f64(pg, &(b->element[0][index])); bx_1=svld1_f64(pg, &(b->element[1][index]));
        bx_2=svld1_f64(pg, &(b->element[2][index])); bx_3=svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_mul(pg, &cx_0, &cx_1, &cx_2, &cx_3, vx_0, vx_1, vx_2, vx_3, bx_0, bx_1, bx_2, bx_3);
        _bncsve2_rqd_add(pg, &cx_0, &cx_1, &cx_2, &cx_3, ax_0, ax_1, ax_2, ax_3, cx_0, cx_1, cx_2, cx_3);
        svst1_f64(pg, &(c->element[0][index]), cx_0);
        svst1_f64(pg, &(c->element[1][index]), cx_1);
        svst1_f64(pg, &(c->element[2][index]), cx_2);
        svst1_f64(pg, &(c->element[3][index]), cx_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=vdupq_n_f64(val[0]); vx[1]=vdupq_n_f64(val[1]);
    vx[2]=vdupq_n_f64(val[2]); vx[3]=vdupq_n_f64(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        bx[0]=vld1q_f64(&(b->element[0][index])); bx[1]=vld1q_f64(&(b->element[1][index]));
        bx[2]=vld1q_f64(&(b->element[2][index])); bx[3]=vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_mul(cx, vx, bx);
        _bncneon_rqd_add(cx, ax, cx);
        vst1q_f64(&(c->element[0][index]), cx[0]);
        vst1q_f64(&(c->element[1][index]), cx[1]);
        vst1q_f64(&(c->element[2][index]), cx[2]);
        vst1q_f64(&(c->element[3][index]), cx[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_mul(tmp, val, get_qdvector_i(b, i));
        rqd_add(tmp, tmp, get_qdvector_i(a, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* c = a - val * b */
void sub_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b)
{
    long int index, i;
    if((a->dim != b->dim) || (c->dim != a->dim)){ fprintf(stderr,"ERROR: sub_cmul_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm256_set1_pd(val[0]); vx[1]=_mm256_set1_pd(val[1]);
    vx[2]=_mm256_set1_pd(val[2]); vx[3]=_mm256_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        bx[0]=_mm256_load_pd(&(b->element[0][index])); bx[1]=_mm256_load_pd(&(b->element[1][index]));
        bx[2]=_mm256_load_pd(&(b->element[2][index])); bx[3]=_mm256_load_pd(&(b->element[3][index]));
        _bncavx2_rqd_mul(cx, vx, bx);
        _bncavx2_rqd_sub(cx, ax, cx);
        _mm256_store_pd(&(c->element[0][index]), cx[0]);
        _mm256_store_pd(&(c->element[1][index]), cx[1]);
        _mm256_store_pd(&(c->element[2][index]), cx[2]);
        _mm256_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__AVX512F__)
    __m512d ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=_mm512_set1_pd(val[0]); vx[1]=_mm512_set1_pd(val[1]);
    vx[2]=_mm512_set1_pd(val[2]); vx[3]=_mm512_set1_pd(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        bx[0]=_mm512_load_pd(&(b->element[0][index])); bx[1]=_mm512_load_pd(&(b->element[1][index]));
        bx[2]=_mm512_load_pd(&(b->element[2][index])); bx[3]=_mm512_load_pd(&(b->element[3][index]));
        _bncavx512_rqd_mul(cx, vx, bx);
        _bncavx512_rqd_sub(cx, ax, cx);
        _mm512_store_pd(&(c->element[0][index]), cx[0]);
        _mm512_store_pd(&(c->element[1][index]), cx[1]);
        _mm512_store_pd(&(c->element[2][index]), cx[2]);
        _mm512_store_pd(&(c->element[3][index]), cx[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t bx_0, bx_1, bx_2, bx_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
    vx_0=svdup_f64(val[0]); vx_1=svdup_f64(val[1]);
    vx_2=svdup_f64(val[2]); vx_3=svdup_f64(val[3]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        bx_0=svld1_f64(pg, &(b->element[0][index])); bx_1=svld1_f64(pg, &(b->element[1][index]));
        bx_2=svld1_f64(pg, &(b->element[2][index])); bx_3=svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_mul(pg, &cx_0, &cx_1, &cx_2, &cx_3, vx_0, vx_1, vx_2, vx_3, bx_0, bx_1, bx_2, bx_3);
        _bncsve2_rqd_neg(pg, &cx_0, &cx_1, &cx_2, &cx_3, cx_0, cx_1, cx_2, cx_3);
		_bncsve2_rqd_add(pg, &cx_0, &cx_1, &cx_2, &cx_3, ax_0, ax_1, ax_2, ax_3, cx_0, cx_1, cx_2, cx_3);
        svst1_f64(pg, &(c->element[0][index]), cx_0);
        svst1_f64(pg, &(c->element[1][index]), cx_1);
        svst1_f64(pg, &(c->element[2][index]), cx_2);
        svst1_f64(pg, &(c->element[3][index]), cx_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t ax[QDSIZE], bx[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=vdupq_n_f64(val[0]); vx[1]=vdupq_n_f64(val[1]);
    vx[2]=vdupq_n_f64(val[2]); vx[3]=vdupq_n_f64(val[3]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        bx[0]=vld1q_f64(&(b->element[0][index])); bx[1]=vld1q_f64(&(b->element[1][index]));
        bx[2]=vld1q_f64(&(b->element[2][index])); bx[3]=vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_mul(cx, vx, bx);
        _bncneon_rqd_sub(cx, ax, cx);
        vst1q_f64(&(c->element[0][index]), cx[0]);
        vst1q_f64(&(c->element[1][index]), cx[1]);
        vst1q_f64(&(c->element[2][index]), cx[2]);
        vst1q_f64(&(c->element[3][index]), cx[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < c->dim; i++){
        rqd_mul(tmp, val, get_qdvector_i(b, i));
        rqd_sub(tmp, get_qdvector_i(a, i), tmp);
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* (a, b) */
void ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b)
{
    long int i, index;
    if(a->dim != b->dim){ fprintf(stderr,"ERROR: ip_qdvector\n"); return; }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d ax[QDSIZE], bx[QDSIZE], sum[QDSIZE], tmp[QDSIZE];
    _bncavx2_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm256_load_pd(&(a->element[0][index])); ax[1]=_mm256_load_pd(&(a->element[1][index]));
        ax[2]=_mm256_load_pd(&(a->element[2][index])); ax[3]=_mm256_load_pd(&(a->element[3][index]));
        bx[0]=_mm256_load_pd(&(b->element[0][index])); bx[1]=_mm256_load_pd(&(b->element[1][index]));
        bx[2]=_mm256_load_pd(&(b->element[2][index])); bx[3]=_mm256_load_pd(&(b->element[3][index]));
        _bncavx2_rqd_mul(tmp, ax, bx);
        _bncavx2_rqd_add(sum, sum, tmp);
    }
    _bncavx2_rqd_sum256d(ret, sum);
#elif defined(__AVX512F__)
    __m512d ax[QDSIZE], bx[QDSIZE], sum[QDSIZE], tmp[QDSIZE];
    _bncavx512_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        ax[0]=_mm512_load_pd(&(a->element[0][index])); ax[1]=_mm512_load_pd(&(a->element[1][index]));
        ax[2]=_mm512_load_pd(&(a->element[2][index])); ax[3]=_mm512_load_pd(&(a->element[3][index]));
        bx[0]=_mm512_load_pd(&(b->element[0][index])); bx[1]=_mm512_load_pd(&(b->element[1][index]));
        bx[2]=_mm512_load_pd(&(b->element[2][index])); bx[3]=_mm512_load_pd(&(b->element[3][index]));
        _bncavx512_rqd_mul(tmp, ax, bx);
        _bncavx512_rqd_add(sum, sum, tmp);
    }
    _bncavx512_rqd_sum512d(ret, sum);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t bx_0, bx_1, bx_2, bx_3;
	svfloat64_t sum_0, sum_1, sum_2, sum_3;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
    _bncsve2_rqd_set0(&sum_0, &sum_1, &sum_2, &sum_3);
    for(index = 0; index < a->real_dim; index += (long int)svcntd()){
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        ax_0=svld1_f64(pg, &(a->element[0][index])); ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index])); ax_3=svld1_f64(pg, &(a->element[3][index]));
        bx_0=svld1_f64(pg, &(b->element[0][index])); bx_1=svld1_f64(pg, &(b->element[1][index]));
        bx_2=svld1_f64(pg, &(b->element[2][index])); bx_3=svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, ax_0, ax_1, ax_2, ax_3, bx_0, bx_1, bx_2, bx_3);
        _bncsve2_rqd_add(svptrue_b64(), &sum_0, &sum_1, &sum_2, &sum_3, sum_0, sum_1, sum_2, sum_3, tmp2_0, tmp2_1, tmp2_2, tmp2_3);
    }
    _bncsve2_rqd_sum128d(svptrue_b64(), ret, sum_0, sum_1, sum_2, sum_3);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t ax[QDSIZE], bx[QDSIZE], sum[QDSIZE], tmp2[QDSIZE];
    _bncneon_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        ax[0]=vld1q_f64(&(a->element[0][index])); ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index])); ax[3]=vld1q_f64(&(a->element[3][index]));
        bx[0]=vld1q_f64(&(b->element[0][index])); bx[1]=vld1q_f64(&(b->element[1][index]));
        bx[2]=vld1q_f64(&(b->element[2][index])); bx[3]=vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_mul(tmp2, ax, bx);
        _bncneon_rqd_add(sum, sum, tmp2);
    }
    _bncneon_rqd_sum128d(ret, sum);
#else
    double tmp[QDSIZE];
    set0_qd(ret);
    for(i = 0; i < a->dim; i++){
        rqd_mul(tmp, get_qdvector_i(a, i), get_qdvector_i(b, i));
        rqd_add(ret, ret, tmp);
    }
#endif
}

/* c := a */
void subst_qdvector(QDVector c, QDVector a)
{
    long int i;
#if defined(__AVX2__) && !defined(__AVX512F__)
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        _mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
        _mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
        _mm256_store_pd(&(c->element[2][i]), _mm256_load_pd(&(a->element[2][i])));
        _mm256_store_pd(&(c->element[3][i]), _mm256_load_pd(&(a->element[3][i])));
    }
#elif defined(__AVX512F__)
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        _mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
        _mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
        _mm512_store_pd(&(c->element[2][i]), _mm512_load_pd(&(a->element[2][i])));
        _mm512_store_pd(&(c->element[3][i]), _mm512_load_pd(&(a->element[3][i])));
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for(i = 0; i < a->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        svst1_f64(pg, &(c->element[0][i]), svld1_f64(pg, &(a->element[0][i])));
        svst1_f64(pg, &(c->element[1][i]), svld1_f64(pg, &(a->element[1][i])));
        svst1_f64(pg, &(c->element[2][i]), svld1_f64(pg, &(a->element[2][i])));
        svst1_f64(pg, &(c->element[3][i]), svld1_f64(pg, &(a->element[3][i])));
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&(c->element[0][i]), vld1q_f64(&(a->element[0][i])));
        vst1q_f64(&(c->element[1][i]), vld1q_f64(&(a->element[1][i])));
        vst1q_f64(&(c->element[2][i]), vld1q_f64(&(a->element[2][i])));
        vst1q_f64(&(c->element[3][i]), vld1q_f64(&(a->element[3][i])));
    }
#else
    for(i = 0; i < a->dim; i++) set_qdvector_i(c, i, get_qdvector_i(a, i));
#endif
}

/* c := -a */
void neg_qdvector(QDVector c, QDVector a)
{
    long int i;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d t[QDSIZE];
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        t[0]=_bncavx2_dneg(_mm256_load_pd(&(a->element[0][i])));
        t[1]=_bncavx2_dneg(_mm256_load_pd(&(a->element[1][i])));
        t[2]=_bncavx2_dneg(_mm256_load_pd(&(a->element[2][i])));
        t[3]=_bncavx2_dneg(_mm256_load_pd(&(a->element[3][i])));
        _mm256_store_pd(&(c->element[0][i]), t[0]);
        _mm256_store_pd(&(c->element[1][i]), t[1]);
        _mm256_store_pd(&(c->element[2][i]), t[2]);
        _mm256_store_pd(&(c->element[3][i]), t[3]);
    }
#elif defined(__AVX512F__)
    __m512d t[QDSIZE];
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        t[0]=_bncavx512_dneg(_mm512_load_pd(&(a->element[0][i])));
        t[1]=_bncavx512_dneg(_mm512_load_pd(&(a->element[1][i])));
        t[2]=_bncavx512_dneg(_mm512_load_pd(&(a->element[2][i])));
        t[3]=_bncavx512_dneg(_mm512_load_pd(&(a->element[3][i])));
        _mm512_store_pd(&(c->element[0][i]), t[0]);
        _mm512_store_pd(&(c->element[1][i]), t[1]);
        _mm512_store_pd(&(c->element[2][i]), t[2]);
        _mm512_store_pd(&(c->element[3][i]), t[3]);
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t t_0, t_1, t_2, t_3;
    for(i = 0; i < a->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        t_0 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[0][i])));
        t_1 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[1][i])));
        t_2 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[2][i])));
        t_3 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[3][i])));
        svst1_f64(pg, &(c->element[0][i]), t_0);
        svst1_f64(pg, &(c->element[1][i]), t_1);
        svst1_f64(pg, &(c->element[2][i]), t_2);
        svst1_f64(pg, &(c->element[3][i]), t_3);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t t[QDSIZE];
    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        t[0] = vnegq_f64(vld1q_f64(&(a->element[0][i])));
        t[1] = vnegq_f64(vld1q_f64(&(a->element[1][i])));
        t[2] = vnegq_f64(vld1q_f64(&(a->element[2][i])));
        t[3] = vnegq_f64(vld1q_f64(&(a->element[3][i])));
        vst1q_f64(&(c->element[0][i]), t[0]);
        vst1q_f64(&(c->element[1][i]), t[1]);
        vst1q_f64(&(c->element[2][i]), t[2]);
        vst1q_f64(&(c->element[3][i]), t[3]);
    }
#else
    double tmp[QDSIZE];
    for(i = 0; i < a->dim; i++){
        rqd_neg(tmp, get_qdvector_i(a, i));
        set_qdvector_i(c, i, tmp);
    }
#endif
}

/* ||a||_1 */
void norm1_qdvector(double ret[QDSIZE], QDVector a)
{
    long int i, index;
#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d vx[QDSIZE], sum[QDSIZE], tmp[QDSIZE];
    _bncavx2_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        vx[0]=_mm256_load_pd(&(a->element[0][index])); vx[1]=_mm256_load_pd(&(a->element[1][index]));
        vx[2]=_mm256_load_pd(&(a->element[2][index])); vx[3]=_mm256_load_pd(&(a->element[3][index]));
        _bncavx2_rqd_abs(tmp, vx);
        _bncavx2_rqd_add(sum, sum, tmp);
    }
    _bncavx2_rqd_abssum256d(ret, sum);
#elif defined(__AVX512F__)
    __m512d vx[QDSIZE], sum[QDSIZE], tmp[QDSIZE];
    _bncavx512_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        vx[0]=_mm512_load_pd(&(a->element[0][index])); vx[1]=_mm512_load_pd(&(a->element[1][index]));
        vx[2]=_mm512_load_pd(&(a->element[2][index])); vx[3]=_mm512_load_pd(&(a->element[3][index]));
        _bncavx512_rqd_abs(tmp, vx);
        _bncavx512_rqd_add(sum, sum, tmp);
    }
    _bncavx512_rqd_abssum512d(ret, sum);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
	svfloat64_t sum_0, sum_1, sum_2, sum_3;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
    _bncsve2_rqd_set0(&sum_0, &sum_1, &sum_2, &sum_3);
    for(index = 0; index < a->real_dim; index += (long int)svcntd()){
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        vx_0=svld1_f64(pg, &(a->element[0][index])); vx_1=svld1_f64(pg, &(a->element[1][index]));
        vx_2=svld1_f64(pg, &(a->element[2][index])); vx_3=svld1_f64(pg, &(a->element[3][index]));
        _bncsve2_rqd_abs(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, vx_0, vx_1, vx_2, vx_3);
        _bncsve2_rqd_add(svptrue_b64(), &sum_0, &sum_1, &sum_2, &sum_3, sum_0, sum_1, sum_2, sum_3, tmp2_0, tmp2_1, tmp2_2, tmp2_3);
    }
    _bncsve2_rqd_abssum128d(svptrue_b64(), ret, sum_0, sum_1, sum_2, sum_3);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t vx[QDSIZE], sum[QDSIZE], tmp2[QDSIZE];
    _bncneon_set0_qd(sum);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        vx[0]=vld1q_f64(&(a->element[0][index])); vx[1]=vld1q_f64(&(a->element[1][index]));
        vx[2]=vld1q_f64(&(a->element[2][index])); vx[3]=vld1q_f64(&(a->element[3][index]));
        _bncneon_rqd_abs(tmp2, vx);
        _bncneon_rqd_add(sum, sum, tmp2);
    }
    _bncneon_rqd_abssum128d(ret, sum);
#else
    double tmp[QDSIZE];
    set0_qd(ret);
    for(i = 0; i < a->dim; i++){
        rqd_abs(tmp, get_qdvector_i(a, i));
        rqd_add(ret, ret, tmp);
    }
#endif
}

/* ||a||_infty */
void normi_qdvector(double ret[QDSIZE], QDVector a)
{
    double tmp[QDSIZE];
    long int i;
    rqd_abs(ret, get_qdvector_i(a, 0));
    for(i = 1; i < a->dim; i++){
        rqd_abs(tmp, get_qdvector_i(a, i));
        if(rqd_cmp(ret, tmp) < 0) rqd_set(ret, tmp);
    }
}

/* ||a||_2 */
void norm2_qdvector(double ret[QDSIZE], QDVector vec)
{
    long int i;
    double acc[QDSIZE], tmp[QDSIZE];
    rqd_set0(acc);
    rqd_set0(ret);
    for(i = 0; i < vec->dim; i++){
        rqd_mul(ret, get_qdvector_i(vec, i), get_qdvector_i(vec, i));
        rqd_add(acc, acc, ret);
    }
#ifdef USE_GMP
    rqd_sqrt_mpfr(ret, acc);
#else
    rqd_sqrt(ret, acc);
#endif
}

/* =========================
   Matrix × Vector (QD)
   ========================= */
#if 0
void mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
    long int i, j, ij, real_col = a->real_col_dim;
    double tmp[QDSIZE], acc[QDSIZE];

    if((v->dim != a->row_dim) || (vb->dim != a->col_dim)){
        fprintf(stderr, "ERROR: mul_qdmatrix_qdvec\n"); return;
    }

#if defined(__AVX2__) && !defined(__AVX512F__)
    __m256d acc4[QDSIZE], aij4[QDSIZE], vb4[QDSIZE], t4[QDSIZE];
    for(i = 0; i < a->row_dim; i++){
        _bncavx2_set0_qd(acc4);
        for(j = 0; j < real_col; j += _BNC_D_WIDTH){
            ij = i * real_col + j;
            aij4[0]=_mm256_load_pd(&(a->element[0][ij])); aij4[1]=_mm256_load_pd(&(a->element[1][ij]));
            aij4[2]=_mm256_load_pd(&(a->element[2][ij])); aij4[3]=_mm256_load_pd(&(a->element[3][ij]));
            vb4[0]=_mm256_load_pd(&(vb->element[0][j]));  vb4[1]=_mm256_load_pd(&(vb->element[1][j]));
            vb4[2]=_mm256_load_pd(&(vb->element[2][j]));  vb4[3]=_mm256_load_pd(&(vb->element[3][j]));
            _bncavx2_rqd_mul(t4, aij4, vb4);
            _bncavx2_rqd_add(acc4, acc4, t4);
        }
        _bncavx2_rqd_sum256d(tmp, acc4);
        v->element[0][i]=tmp[0]; v->element[1][i]=tmp[1];
        v->element[2][i]=tmp[2]; v->element[3][i]=tmp[3];
    }
#elif defined(__AVX512F__)
    __m512d acc8[QDSIZE], aij8[QDSIZE], vb8[QDSIZE], t8[QDSIZE];
    for(i = 0; i < a->row_dim; i++){
        _bncavx512_set0_qd(acc8);
        for(j = 0; j < real_col; j += _BNC_D_WIDTH){
            ij = i * real_col + j;
            aij8[0]=_mm512_load_pd(&(a->element[0][ij])); aij8[1]=_mm512_load_pd(&(a->element[1][ij]));
            aij8[2]=_mm512_load_pd(&(a->element[2][ij])); aij8[3]=_mm512_load_pd(&(a->element[3][ij]));
            vb8[0]=_mm512_load_pd(&(vb->element[0][j]));  vb8[1]=_mm512_load_pd(&(vb->element[1][j]));
            vb8[2]=_mm512_load_pd(&(vb->element[2][j]));  vb8[3]=_mm512_load_pd(&(vb->element[3][j]));
            _bncavx512_rqd_mul(t8, aij8, vb8);
            _bncavx512_rqd_add(acc8, acc8, t8);
        }
        _bncavx512_rqd_sum512d(tmp, acc8);
        v->element[0][i]=tmp[0]; v->element[1][i]=tmp[1];
        v->element[2][i]=tmp[2]; v->element[3][i]=tmp[3];
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t acc2_0, acc2_1, acc2_2, acc2_3;
	svfloat64_t aij2_0, aij2_1, aij2_2, aij2_3;
	svfloat64_t vb2_0, vb2_1, vb2_2, vb2_3;
	svfloat64_t t2_0, t2_1, t2_2, t2_3;
    for(i = 0; i < a->row_dim; i++){
        _bncsve2_rqd_set0(&acc2_0, &acc2_1, &acc2_2, &acc2_3);
        for(j = 0; j < real_col; j += (long int)svcntd()){
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col);
            ij = i * real_col + j;
            aij2_0=svld1_f64(pg, &(a->element[0][ij])); aij2_1=svld1_f64(pg, &(a->element[1][ij]));
            aij2_2=svld1_f64(pg, &(a->element[2][ij])); aij2_3=svld1_f64(pg, &(a->element[3][ij]));
            vb2_0=svld1_f64(pg, &(vb->element[0][j]));   vb2_1=svld1_f64(pg, &(vb->element[1][j]));
            vb2_2=svld1_f64(pg, &(vb->element[2][j]));   vb2_3=svld1_f64(pg, &(vb->element[3][j]));
            _bncsve2_rqd_mul(svptrue_b64(), &t2_0, &t2_1, &t2_2, &t2_3, aij2_0, aij2_1, aij2_2, aij2_3, vb2_0, vb2_1, vb2_2, vb2_3);
            _bncsve2_rqd_add(svptrue_b64(), &acc2_0, &acc2_1, &acc2_2, &acc2_3, acc2_0, acc2_1, acc2_2, acc2_3, t2_0, t2_1, t2_2, t2_3);
        }
        _bncsve2_rqd_sum128d(svptrue_b64(), tmp, acc2_0, acc2_1, acc2_2, acc2_3);
        v->element[0][i]=tmp[0]; v->element[1][i]=tmp[1];
        v->element[2][i]=tmp[2]; v->element[3][i]=tmp[3];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t acc2[QDSIZE], aij2[QDSIZE], vb2[QDSIZE], t2[QDSIZE];
    for(i = 0; i < a->row_dim; i++){
        _bncneon_set0_qd(acc2);
        for(j = 0; j < real_col; j += _BNC_D_WIDTH){
            ij = i * real_col + j;
            aij2[0]=vld1q_f64(&(a->element[0][ij])); aij2[1]=vld1q_f64(&(a->element[1][ij]));
            aij2[2]=vld1q_f64(&(a->element[2][ij])); aij2[3]=vld1q_f64(&(a->element[3][ij]));
            vb2[0]=vld1q_f64(&(vb->element[0][j]));   vb2[1]=vld1q_f64(&(vb->element[1][j]));
            vb2[2]=vld1q_f64(&(vb->element[2][j]));   vb2[3]=vld1q_f64(&(vb->element[3][j]));
            _bncneon_rqd_mul(t2, aij2, vb2);
            _bncneon_rqd_add(acc2, acc2, t2);
        }
        _bncneon_rqd_sum128d(tmp, acc2);
        v->element[0][i]=tmp[0]; v->element[1][i]=tmp[1];
        v->element[2][i]=tmp[2]; v->element[3][i]=tmp[3];
    }
#else
    for(i = 0; i < a->row_dim; i++){
        rqd_set_ui(acc, 0UL);
        for(j = 0; j < a->col_dim; j++){
            rqd_mul(tmp, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
            rqd_add(acc, acc, tmp);
        }
        set_qdvector_i(v, i, acc);
    }
#endif
}
#endif // 0

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void set0_qdmatrix(QDMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(mat->element[0][i]), zero4);
		_mm256_store_pd(&(mat->element[1][i]), zero4);
		_mm256_store_pd(&(mat->element[2][i]), zero4);
		_mm256_store_pd(&(mat->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(mat->element[0][i]), zero8);
		_mm512_store_pd(&(mat->element[1][i]), zero8);
		_mm512_store_pd(&(mat->element[2][i]), zero8);
		_mm512_store_pd(&(mat->element[3][i]), zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t z = svdup_f64(0.0);
    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(mat->element[0][i]), z);
        svst1_f64(pg, &(mat->element[1][i]), z);
        svst1_f64(pg, &(mat->element[2][i]), z);
        svst1_f64(pg, &(mat->element[3][i]), z);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t z = vdupq_n_f64(0.0);
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(mat->element[0][i]), z);
        vst1q_f64(&(mat->element[1][i]), z);
        vst1q_f64(&(mat->element[2][i]), z);
        vst1q_f64(&(mat->element[3][i]), z);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
		mat->element[2][i] = 0.0;	
		mat->element[3][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize qdvector
QDMatrix init_qdmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	QDMatrix ret = NULL;

	ret = (QDMatrix)BNC_MALLOC(sizeof(qdmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_row_dim = (long int)ceil((double)(row_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_col_dim = (long int)ceil((double)(col_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_qdmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
	ret->element[0] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[0] == NULL)
	{ 	free(ret);
		return NULL;
	}
	ret->element[1] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}
	ret->element[2] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[2] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret);
		return NULL;
	}
	ret->element[3] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[3] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret->element[2]);
		free(ret);
		return NULL;
	}

	//printf("init_qdmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][i]), zero4);
		_mm256_store_pd(&(ret->element[1][i]), zero4);
		_mm256_store_pd(&(ret->element[2][i]), zero4);
		_mm256_store_pd(&(ret->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
		_mm512_store_pd(&(ret->element[2][i]), zero8);
		_mm512_store_pd(&(ret->element[3][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	svfloat64_t z2 = svdup_f64(0.0);
    for(i = 0; i < real_total_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(ret->element[0][i]), z2);
        svst1_f64(pg, &(ret->element[1][i]), z2);
        svst1_f64(pg, &(ret->element[2][i]), z2);
        svst1_f64(pg, &(ret->element[3][i]), z2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float64x2_t z2 = vdupq_n_f64(0.0);
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&(ret->element[0][i]), z2);
        vst1q_f64(&(ret->element[1][i]), z2);
        vst1q_f64(&(ret->element[2][i]), z2);
        vst1q_f64(&(ret->element[3][i]), z2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
		ret->element[2][i] = 0.0;
		ret->element[3][i] = 0.0;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free qdvector
void free_qdmatrix(QDMatrix mat)
{
	long int i;

	for(i = 0; i < QDSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print qdvector
void print_qdmatrix(QDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_qd_write((vec->element + index * QDSIZE));
			c_qd_write(GET_QDMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// QDMatrix mat -> qdfloat array
void set_qdfloat_qdmat(qdfloat ret[], int ret_dim, QDMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < QDSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// qdfloat array -> QDmatrix ret
void set_qdmatrix_qdfloat(QDMatrix ret, qdfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < QDSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}

// matrix multiplication
// ret := A * B
void mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[4][QDSIZE];
    __m256d cij[QDSIZE], aik[QDSIZE], bkj[QDSIZE], tmp_mul[QDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rqd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
            cij[2] = _mm256_setzero_pd();
            cij[3] = _mm256_setzero_pd();
			for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            
			    aik[0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
            /*    aik[0] = _mm256_set_pd(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3]
                );
            */
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
                aik[1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
            /*    aik[1] = _mm256_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
            */
                aik[2] = _mm256_load_pd(&(a->element[2][i * real_mid_dim + k]));
                aik[3] = _mm256_load_pd(&(a->element[3][i * real_mid_dim + k]));
		    /*   aik[2] = _mm256_set_pd(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3]
                );
                aik[3] = _mm256_set_pd(
                    a->element[3][i * real_mid_dim + k],
                    a->element[3][i * real_mid_dim + k + 1],
                    a->element[3][i * real_mid_dim + k + 2],
                    a->element[3][i * real_mid_dim + k + 3]
                );
            */    
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm256_set_pd(
                    //b->element[0][ k      * real_col_dim + j],
                    //b->element[0][(k + 1) * real_col_dim + j],
                    //b->element[0][(k + 2) * real_col_dim + j],
                    //b->element[0][(k + 3) * real_col_dim + j]
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm256_set_pd(
                    //b->element[1][ k      * real_col_dim + j],
                    //b->element[1][(k + 1) * real_col_dim + j],
                    //b->element[1][(k + 2) * real_col_dim + j],
                    //b->element[1][(k + 3) * real_col_dim + j]
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );
                bkj[2] = _mm256_set_pd(
                    //b->element[2][ k      * real_col_dim + j],
                    //b->element[2][(k + 1) * real_col_dim + j],
                    //b->element[2][(k + 2) * real_col_dim + j],
                    //b->element[2][(k + 3) * real_col_dim + j]
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );
                bkj[3] = _mm256_set_pd(
                    //b->element[3][ k      * real_col_dim + j],
                    //b->element[3][(k + 1) * real_col_dim + j],
                    //b->element[3][(k + 2) * real_col_dim + j],
                    //b->element[3][(k + 3) * real_col_dim + j]
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

            /*
                rqd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rqd_mul(tmp_mul, aik, bkj);

            /*
                rqd_add(cij.val, cij.val, tmp_mul[0].val);
                rqd_add(cij.val, cij.val, tmp_mul[1].val);
                rqd_add(cij.val, cij.val, tmp_mul[2].val);
                rqd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rqd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0]; cijval[0][3] = cij[3][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1]; cijval[1][3] = cij[3][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2]; cijval[2][3] = cij[3][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3]; cijval[3][3] = cij[3][3];
            rqd_add(cijval[0], cijval[0], cijval[1]);
            rqd_add(cijval[0], cijval[0], cijval[2]);
            rqd_add(cijval[0], cijval[0], cijval[3]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
            ret->element[3][i * real_col_dim + j] = cijval[0][3];
       }
    }

#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[8][QDSIZE];
    __m512d cij[QDSIZE], aik[QDSIZE], bkj[QDSIZE], tmp_mul[QDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rqd_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_pd();
            cij[1] = _mm512_setzero_pd();
            cij[2] = _mm512_setzero_pd();
			cij[3] = _mm512_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            //    aik[0] = _mm256_load_pd(&(a->element[0][i * mid_dim + k]));
 			    aik[0] = _mm512_load_pd(&(a->element[0][i * real_mid_dim + k]));
            /*   
                aik[0] = _mm512_set_pd(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3],
                    a->element[0][i * real_mid_dim + k + 4],
                    a->element[0][i * real_mid_dim + k + 5],
                    a->element[0][i * real_mid_dim + k + 6],
                    a->element[0][i * real_mid_dim + k + 7]
                );
            */
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
            //    aik[1] = _mm256_load_pd(&(a->element[1][i * mid_dim + k]));
 			    aik[1] = _mm512_load_pd(&(a->element[1][i * real_mid_dim + k]));                
 			    aik[2] = _mm512_load_pd(&(a->element[2][i * real_mid_dim + k]));  
 			    aik[3] = _mm512_load_pd(&(a->element[3][i * real_mid_dim + k]));  
			/*    aik[1] = _mm512_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3],
                    a->element[1][i * real_mid_dim + k + 4],
                    a->element[1][i * real_mid_dim + k + 5],
                    a->element[1][i * real_mid_dim + k + 6],
                    a->element[1][i * real_mid_dim + k + 7]
                );
                aik[2] = _mm512_set_pd(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3],
                    a->element[2][i * real_mid_dim + k + 4],
                    a->element[2][i * real_mid_dim + k + 5],
                    a->element[2][i * real_mid_dim + k + 6],
                    a->element[2][i * real_mid_dim + k + 7]
                );
                aik[3] = _mm512_set_pd(
                    a->element[3][i * real_mid_dim + k],
                    a->element[3][i * real_mid_dim + k + 1],
                    a->element[3][i * real_mid_dim + k + 2],
                    a->element[3][i * real_mid_dim + k + 3],
                    a->element[3][i * real_mid_dim + k + 4],
                    a->element[3][i * real_mid_dim + k + 5],
                    a->element[3][i * real_mid_dim + k + 6],
                    a->element[3][i * real_mid_dim + k + 7]
                );
			*/
                 
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm512_set_pd(
                /*    b->element[0][ k      * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 7) * real_col_dim + j]
				*/
                    b->element[0][(k + 7) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm512_set_pd(
                /*
					b->element[1][ k      * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 7) * real_col_dim + j]
				*/
                    b->element[1][(k + 7) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]

                );
                bkj[2] = _mm512_set_pd(
                /*
				    b->element[2][ k      * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 7) * real_col_dim + j]
				*/
                    b->element[2][(k + 7) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );
                bkj[3] = _mm512_set_pd(
				/*
                    b->element[3][ k      * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 7) * real_col_dim + j]
				*/
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]

                );

            /*
                rqd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rqd_mul(tmp_mul, aik, bkj);

            /*
                rqd_add(cij.val, cij.val, tmp_mul[0].val);
                rqd_add(cij.val, cij.val, tmp_mul[1].val);
                rqd_add(cij.val, cij.val, tmp_mul[2].val);
                rqd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rqd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0]; cijval[0][3] = cij[3][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1]; cijval[1][3] = cij[3][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2]; cijval[2][3] = cij[3][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3]; cijval[3][3] = cij[3][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4]; cijval[4][3] = cij[3][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5]; cijval[5][3] = cij[3][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6]; cijval[6][3] = cij[3][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7]; cijval[7][3] = cij[3][7];
            rqd_add(cijval[0], cijval[0], cijval[1]);
            rqd_add(cijval[0], cijval[0], cijval[2]);
            rqd_add(cijval[0], cijval[0], cijval[3]);
            rqd_add(cijval[0], cijval[0], cijval[4]);
            rqd_add(cijval[0], cijval[0], cijval[5]);
            rqd_add(cijval[0], cijval[0], cijval[6]);
            rqd_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
  			ret->element[3][i * real_col_dim + j] = cijval[0][3];
        }
    }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    /* Outer-product GEMM, per-limb-variable API. */
    {
        long real_row_dim = a->real_row_dim;
        long real_col_dim = b->real_col_dim;
        long real_mid_dim = a->real_col_dim;
        long vl = (long)svcntd();

        for(i = 0; i < real_row_dim; i++){
            for(j = 0; j < real_col_dim; j += vl){
                svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
                svfloat64_t cij0, cij1, cij2, cij3;
                _bncsve2_rqd_set0(&cij0, &cij1, &cij2, &cij3);
                for(k = 0; k < real_mid_dim; k++){
                    svfloat64_t aik0 = svdup_n_f64(a->element[0][i*real_mid_dim + k]);
                    svfloat64_t aik1 = svdup_n_f64(a->element[1][i*real_mid_dim + k]);
                    svfloat64_t aik2 = svdup_n_f64(a->element[2][i*real_mid_dim + k]);
                    svfloat64_t aik3 = svdup_n_f64(a->element[3][i*real_mid_dim + k]);
                    svfloat64_t bkj0 = svld1_f64(pg, &(b->element[0][k*real_col_dim + j]));
                    svfloat64_t bkj1 = svld1_f64(pg, &(b->element[1][k*real_col_dim + j]));
                    svfloat64_t bkj2 = svld1_f64(pg, &(b->element[2][k*real_col_dim + j]));
                    svfloat64_t bkj3 = svld1_f64(pg, &(b->element[3][k*real_col_dim + j]));
                    svfloat64_t t0, t1, t2, t3;
                    _bncsve2_rqd_mul(pg, &t0, &t1, &t2, &t3,
                                     aik0, aik1, aik2, aik3,
                                     bkj0, bkj1, bkj2, bkj3);
                    _bncsve2_rqd_add(pg, &cij0, &cij1, &cij2, &cij3,
                                     cij0, cij1, cij2, cij3,
                                     t0, t1, t2, t3);
                }
                svst1_f64(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
                svst1_f64(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
                svst1_f64(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
                svst1_f64(pg, &(ret->element[3][i*real_col_dim + j]), cij3);
            }
        }
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long real_row_dim = a->real_row_dim;
	long real_col_dim = b->real_col_dim;
	long real_mid_dim = a->real_col_dim;
    float64x2_t cij2[QDSIZE], aik2[QDSIZE], bkj2[QDSIZE], tmp_mul2[QDSIZE];
	double __attribute__ ((aligned(16))) bkj2l[QDSIZE][2];
    double cijval[QDSIZE];

#if 0
    for(i = 0; i < ret->real_row_dim; i += 1){
        for(j = 0; j < ret->real_col_dim; j += _BNC_D_WIDTH){
            _bncneon_set0_qd(cij2);
            for(k = 0; k < real_mid_dim; k += 1){
                // aik2[0] = vld1q_f64(&(a->element[0][i * real_mid_dim + k]));
                // aik2[1] = vld1q_f64(&(a->element[1][i * real_mid_dim + k]));
                // aik2[2] = vld1q_f64(&(a->element[2][i * real_mid_dim + k]));
                // aik2[3] = vld1q_f64(&(a->element[3][i * real_mid_dim + k]));
                aik2[0] = vdupq_n_f64(a->element[0][i * real_mid_dim + k]);
                aik2[1] = vdupq_n_f64(a->element[1][i * real_mid_dim + k]);
                aik2[2] = vdupq_n_f64(a->element[2][i * real_mid_dim + k]);
                aik2[3] = vdupq_n_f64(a->element[3][i * real_mid_dim + k]);

				bkj2[0] = vld1q_f64(&(b->element[0][k * real_col_dim + j]));
                bkj2[1] = vld1q_f64(&(b->element[1][k * real_col_dim + j]));
                bkj2[2] = vld1q_f64(&(b->element[2][k * real_col_dim + j]));
                bkj2[3] = vld1q_f64(&(b->element[3][k * real_col_dim + j]));
                _bncneon_rqd_mul(tmp_mul2, aik2, bkj2);
                _bncneon_rqd_add(cij2, cij2, tmp_mul2);
            }
			vst1q_f64(&(ret->element[0][i*ret->real_col_dim + j]), cij2[0]);
            vst1q_f64(&(ret->element[1][i*ret->real_col_dim + j]), cij2[1]);
            vst1q_f64(&(ret->element[2][i*ret->real_col_dim + j]), cij2[2]);
            vst1q_f64(&(ret->element[3][i*ret->real_col_dim + j]), cij2[3]);
		}
	}
#endif // 0
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            _bncneon_set0_qd(cij2);
			for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik2[0] = vld1q_f64(&(a->element[0][i * real_mid_dim + k]));
                aik2[1] = vld1q_f64(&(a->element[1][i * real_mid_dim + k]));
                aik2[2] = vld1q_f64(&(a->element[2][i * real_mid_dim + k]));
                aik2[3] = vld1q_f64(&(a->element[3][i * real_mid_dim + k]));
                /*
				aik2[0] = vdupq_n_f64(a->element[0][i * real_mid_dim + k]);
                aik2[1] = vdupq_n_f64(a->element[1][i * real_mid_dim + k]);
                aik2[2] = vdupq_n_f64(a->element[2][i * real_mid_dim + k]);
                aik2[3] = vdupq_n_f64(a->element[3][i * real_mid_dim + k]);
				*/

				bkj2l[0][0] = b->element[0][(k    ) * real_col_dim + j];
				bkj2l[0][1] = b->element[0][(k + 1) * real_col_dim + j];
				bkj2[0] = vld1q_f64(bkj2l[0]); // [(&(b->element[0][k * real_col_dim + j]));

				bkj2l[1][0] = b->element[1][(k    ) * real_col_dim + j];
				bkj2l[1][1] = b->element[1][(k + 1) * real_col_dim + j];
				bkj2[1] = vld1q_f64(bkj2l[1]); // [(&(b->element[0][k * real_col_dim + j]));

				bkj2l[2][0] = b->element[2][(k    ) * real_col_dim + j];
				bkj2l[2][1] = b->element[2][(k + 1) * real_col_dim + j];
				bkj2[2] = vld1q_f64(bkj2l[2]); // [(&(b->element[0][k * real_col_dim + j]));

				bkj2l[3][0] = b->element[3][(k    ) * real_col_dim + j];
				bkj2l[3][1] = b->element[3][(k + 1) * real_col_dim + j];
				bkj2[3] = vld1q_f64(bkj2l[3]); // [(&(b->element[0][k * real_col_dim + j]));

				_bncneon_rqd_mul(tmp_mul2, aik2, bkj2);
                _bncneon_rqd_add(cij2, cij2, tmp_mul2);
            }
			_bncneon_rqd_sum128d(cijval, cij2);
           	//set_qdmatrix_ij(ret, i, j, cijval2);
			ret->element[0][i * real_col_dim + j] = cijval[0];
            ret->element[1][i * real_col_dim + j] = cijval[1];
            ret->element[2][i * real_col_dim + j] = cijval[2];
  			ret->element[3][i * real_col_dim + j] = cijval[3];
        }
    }

#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	double tmp[QDSIZE], ret_ij[QDSIZE], b_kj[QDSIZE];

	//printf("Non SIMD mul_qdmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

//#if 0
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rqd_set0(GET_QDMATRIX_IJ(ret, i, j));
			rqd_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rqd_mul(tmp, GET_QDMATRIX_IJ(a, i, k), GET_QDMATRIX_IJ(b, k, j));
				//rqd_add(GET_QDMATRIX_IJ(ret, i, j), tmp, GET_QDMATRIX_IJ(ret, i, j));
				rqd_add(ret_ij, tmp, ret_ij);
			}
			set_qdmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
/*#else // 0
	set0_qdmatrix(ret);
	// From codes of Rgemm
	for(j = 0; j < col_dim; j++)
	{
		for(k = 0; k < mid_dim; k++)
		{
			rqd_set(b_kj, get_qdmatrix_ij(b, k, j));
			for(i = 0; i < row_dim; i++)
			{
				rqd_mul(tmp, get_qdmatrix_ij(a, i, k), b_kj);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), tmp);
			}
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
*/
//#endif // 0
#endif // __AVX2__

}

// Frobenius norm
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[QDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d ret4[QDSIZE], mat4[QDSIZE], tmp4[QDSIZE];

	ret4[0] = _mm256_setzero_pd();
	ret4[1] = _mm256_setzero_pd();
	ret4[2] = _mm256_setzero_pd();
	ret4[3] = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat4[0] = _mm256_load_pd(&(mat->element[0][i]));
		mat4[1] = _mm256_load_pd(&(mat->element[1][i]));
		mat4[2] = _mm256_load_pd(&(mat->element[2][i]));
		mat4[3] = _mm256_load_pd(&(mat->element[3][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rqd_mul(tmp4, mat4, mat4);
		_bncavx2_rqd_add(ret4, ret4, tmp4);
	}

	_bncavx2_rqd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d ret8[QDSIZE], mat8[QDSIZE], tmp8[QDSIZE];

	ret8[0] = _mm512_setzero_pd();
	ret8[1] = _mm512_setzero_pd();
	ret8[2] = _mm512_setzero_pd();
	ret8[3] = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat8[0] = _mm512_load_pd(&(mat->element[0][i]));
		mat8[1] = _mm512_load_pd(&(mat->element[1][i]));
		mat8[2] = _mm512_load_pd(&(mat->element[2][i]));
		mat8[3] = _mm512_load_pd(&(mat->element[3][i]));

		// tmp8 := mat8[i]^2
		// ret8 += tmp8
		_bncavx512_rqd_mul(tmp8, mat8, mat8);
		_bncavx512_rqd_add(ret8, ret8, tmp8);
	}

	_bncavx512_rqd_sum512d(ret, ret8);


#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t ret2_0, ret2_1, ret2_2, ret2_3;
	svfloat64_t mat2_0, mat2_1, mat2_2, mat2_3;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;

    // ret2 := 0
    _bncsve2_rqd_set0(&ret2_0, &ret2_1, &ret2_2, &ret2_3);

    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)i, (int)real_total_dim);
        // mat2[j] := [ mat->element[j][i], mat->element[j][i+1] ]
        mat2_0 = svld1_f64(pg, &(mat->element[0][i]));
        mat2_1 = svld1_f64(pg, &(mat->element[1][i]));
        mat2_2 = svld1_f64(pg, &(mat->element[2][i]));
        mat2_3 = svld1_f64(pg, &(mat->element[3][i]));

        // tmp2 := mat2^2
        // ret2 += tmp2
        _bncsve2_rqd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, mat2_0, mat2_1, mat2_2, mat2_3, mat2_0, mat2_1, mat2_2, mat2_3);
        _bncsve2_rqd_add(svptrue_b64(), &ret2_0, &ret2_1, &ret2_2, &ret2_3, ret2_0, ret2_1, ret2_2, ret2_3, tmp2_0, tmp2_1, tmp2_2, tmp2_3);
    }

    // ret := sum(ret2)
    _bncsve2_rqd_sum128d(svptrue_b64(), ret, ret2_0, ret2_1, ret2_2, ret2_3);

#elif defined(__ARM_NEON) && defined(__aarch64__) && defined(BNC_ENABLE_NEON)  // ARM NEON (64bit, float64x2_t)
    float64x2_t ret2[QDSIZE], mat2[QDSIZE], tmp2[QDSIZE];

    // ret2 := 0
    _bncneon_set0_qd(ret2);

    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        // mat2[j] := [ mat->element[j][i], mat->element[j][i+1] ]
        mat2[0] = vld1q_f64(&(mat->element[0][i]));
        mat2[1] = vld1q_f64(&(mat->element[1][i]));
        mat2[2] = vld1q_f64(&(mat->element[2][i]));
        mat2[3] = vld1q_f64(&(mat->element[3][i]));

        // tmp2 := mat2^2
        // ret2 += tmp2
        _bncneon_rqd_mul(tmp2, mat2, mat2);
        _bncneon_rqd_add(ret2, ret2, tmp2);
    }

    // ret := sum(ret2)
    _bncneon_rqd_sum128d(ret, ret2);

#else // others
	double mat1[QDSIZE];

	rqd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];
		mat1[2] = mat->element[2][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rqd_mul(tmp, mat1, mat1);
		rqd_add(ret, ret, tmp);
	}

#endif // __AVX2__
#ifdef USE_GMP
	rqd_sqrt_mpfr(tmp, ret);   /* tmp := sqrt(sum-of-squares in ret) */
#else // USE_GMP
	rqd_sqrt(tmp, ret);
#endif // USE_GMP
	rqd_set(ret, tmp);

}

// print normf
void print_normf_qdmatrix(const char *str, QDMatrix mat)
{
	static double tmp[QDSIZE];

	normf_qdmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rqd_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for QDMatrix            */
/*
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix_ddvec(QDVector v, QDMatrix a, QDVector vb)
void mul_qdmatrixt_ddvec(QDVector v, QDMatrix a, QDVector vb)
void transpose_qdmatrix(QDMatrix c, QDMatrix a);
void inv_qdmatrix(QDMatrix a);
void subst_qdmatrix(QDMatrix c, QDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat)
{
	long int i, j;
	double tmp[QDSIZE], sum[QDSIZE];

	set0_qd(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_qd(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rqd_abs(tmp, get_qdmatrix_ij(mat, i, j));
			rqd_add(sum, sum, tmp);
		}
		if(rqd_cmp(ret, sum) < 0)
			rqd_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat)
{
	long int i, j;
	double tmp[QDSIZE], sum[QDSIZE];

	rqd_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rqd_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rqd_abs(tmp, get_qdmatrix_ij(mat, i, j));
			rqd_add(sum, sum, tmp);
		}
		if(rqd_cmp(ret, sum) < 0)
			rqd_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//#ifdef BNC_USE_GCC
	__m256d tmp4[QDSIZE], aij4[QDSIZE], bij4[QDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[3] = _mm256_load_pd(&(a->element[3][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[2] = _mm256_load_pd(&(b->element[2][index]));
		bij4[3] = _mm256_load_pd(&(b->element[3][index]));

		_bncavx2_rqd_add(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[3]); 
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[QDSIZE], aij8[QDSIZE], bij8[QDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[3] = _mm512_load_pd(&(a->element[3][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[2] = _mm512_load_pd(&(b->element[2][index]));
		bij8[3] = _mm512_load_pd(&(b->element[3][index]));

		_bncavx512_rqd_add(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[3]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
	svfloat64_t aij2_0, aij2_1, aij2_2, aij2_3;
	svfloat64_t bij2_0, bij2_1, bij2_2, bij2_3;
    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        aij2_2 = svld1_f64(pg, &(a->element[2][index]));
        aij2_3 = svld1_f64(pg, &(a->element[3][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));
        bij2_2 = svld1_f64(pg, &(b->element[2][index]));
        bij2_3 = svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, aij2_0, aij2_1, aij2_2, aij2_3, bij2_0, bij2_1, bij2_2, bij2_3);
        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        svst1_f64(pg, &(c->element[2][index]), tmp2_2);
        svst1_f64(pg, &(c->element[3][index]), tmp2_3);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t tmp2[QDSIZE], aij2[QDSIZE], bij2[QDSIZE];
    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        aij2[2] = vld1q_f64(&(a->element[2][index]));
        aij2[3] = vld1q_f64(&(a->element[3][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));
        bij2[2] = vld1q_f64(&(b->element[2][index]));
        bij2[3] = vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_add(tmp2, aij2, bij2);
        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]);
        vst1q_f64(&(c->element[2][index]), tmp2[2]);
        vst1q_f64(&(c->element[3][index]), tmp2[3]);
    }

#else // others
	double tmp[QDSIZE], aij[QDSIZE], bij[QDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];
		bij[3] = b->element[3][index];

		rqd_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
/*
	double tmp[QDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_add(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	double tmp[QDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_sub(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[QDSIZE], aij4[QDSIZE], bij4[QDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[3] = _mm256_load_pd(&(a->element[3][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[2] = _mm256_load_pd(&(b->element[2][index]));
		bij4[3] = _mm256_load_pd(&(b->element[3][index]));

		_bncavx2_rqd_sub(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[QDSIZE], aij8[QDSIZE], bij8[QDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[3] = _mm512_load_pd(&(a->element[3][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[2] = _mm512_load_pd(&(b->element[2][index]));
		bij8[3] = _mm512_load_pd(&(b->element[3][index]));

		_bncavx512_rqd_sub(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[3]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
	svfloat64_t aij2_0, aij2_1, aij2_2, aij2_3;
	svfloat64_t bij2_0, bij2_1, bij2_2, bij2_3;
    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        aij2_2 = svld1_f64(pg, &(a->element[2][index]));
        aij2_3 = svld1_f64(pg, &(a->element[3][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));
        bij2_2 = svld1_f64(pg, &(b->element[2][index]));
        bij2_3 = svld1_f64(pg, &(b->element[3][index]));
        _bncsve2_rqd_neg(pg, &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, bij2_0, bij2_1, bij2_2, bij2_3);
		_bncsve2_rqd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, aij2_0, aij2_1, aij2_2, aij2_3, tmp2_0, tmp2_1, tmp2_2, tmp2_3);
        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        svst1_f64(pg, &(c->element[2][index]), tmp2_2);
        svst1_f64(pg, &(c->element[3][index]), tmp2_3);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t tmp2[QDSIZE], aij2[QDSIZE], bij2[QDSIZE];
    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        aij2[2] = vld1q_f64(&(a->element[2][index]));
        aij2[3] = vld1q_f64(&(a->element[3][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));
        bij2[2] = vld1q_f64(&(b->element[2][index]));
        bij2[3] = vld1q_f64(&(b->element[3][index]));
        _bncneon_rqd_sub(tmp2, aij2, bij2);
        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]);
        vst1q_f64(&(c->element[2][index]), tmp2[2]);
        vst1q_f64(&(c->element[3][index]), tmp2[3]);
    }

#else // others
	double tmp[QDSIZE], aij[QDSIZE], bij[QDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];
		bij[3] = b->element[3][index];

		rqd_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[QDSIZE], sc4[QDSIZE], aij4[QDSIZE];

	sc4[0] = _mm256_set1_pd(sc[0]);
	sc4[1] = _mm256_set1_pd(sc[1]);
	sc4[2] = _mm256_set1_pd(sc[2]);
	sc4[3] = _mm256_set1_pd(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[3] = _mm256_load_pd(&(a->element[3][index]));

		_bncavx2_rqd_mul(tmp4, sc4, aij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[QDSIZE], aij8[QDSIZE], sc8[QDSIZE];

	sc8[0] = _mm512_set1_pd(sc[0]);
	sc8[1] = _mm512_set1_pd(sc[1]);
	sc8[2] = _mm512_set1_pd(sc[2]);
	sc8[3] = _mm512_set1_pd(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[3] = _mm512_load_pd(&(a->element[3][index]));

		_bncavx512_rqd_mul(tmp8, sc8, aij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[3]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t ax_0, ax_1, ax_2, ax_3;
	svfloat64_t cx_0, cx_1, cx_2, cx_3;
	svfloat64_t vx_0, vx_1, vx_2, vx_3;
    vx_0=svdup_f64(sc[0]);
	vx_1=svdup_f64(sc[1]);
    vx_2=svdup_f64(sc[2]);
	vx_3=svdup_f64(sc[3]);

	for(index = 0; index < real_total_dim; index += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        ax_0=svld1_f64(pg, &(a->element[0][index]));
		ax_1=svld1_f64(pg, &(a->element[1][index]));
        ax_2=svld1_f64(pg, &(a->element[2][index]));
		ax_3=svld1_f64(pg, &(a->element[3][index]));

		_bncsve2_rqd_mul(pg, &cx_0, &cx_1, &cx_2, &cx_3, vx_0, vx_1, vx_2, vx_3, ax_0, ax_1, ax_2, ax_3);

		svst1_f64(pg, &(c->element[0][index]), cx_0);
        svst1_f64(pg, &(c->element[1][index]), cx_1);
        svst1_f64(pg, &(c->element[2][index]), cx_2);
        svst1_f64(pg, &(c->element[3][index]), cx_3);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t ax[QDSIZE], cx[QDSIZE], vx[QDSIZE];
    vx[0]=vdupq_n_f64(sc[0]);
	vx[1]=vdupq_n_f64(sc[1]);
    vx[2]=vdupq_n_f64(sc[2]);
	vx[3]=vdupq_n_f64(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH) {
        ax[0]=vld1q_f64(&(a->element[0][index]));
		ax[1]=vld1q_f64(&(a->element[1][index]));
        ax[2]=vld1q_f64(&(a->element[2][index]));
		ax[3]=vld1q_f64(&(a->element[3][index]));

		_bncneon_rqd_mul(cx, vx, ax);

		vst1q_f64(&(c->element[0][index]), cx[0]);
        vst1q_f64(&(c->element[1][index]), cx[1]);
        vst1q_f64(&(c->element[2][index]), cx[2]);
        vst1q_f64(&(c->element[3][index]), cx[3]);
    }

#else // others
	double tmp[QDSIZE], aij[QDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];

		rqd_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
/*
	double tmp[QDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_mul(tmp, sc, get_qdmatrix_ij(a, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	/* Check Dimentions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_dmatrix\n");
		return;
	}

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;
// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d aji4[QDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji4[0] = _mm256_set_pd(
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aji4[1] = _mm256_set_pd(
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aji4[2] = _mm256_set_pd(
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aji4[3] = _mm256_set_pd(
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), aji4[0]);
			_mm256_store_pd(&(c->element[1][index]), aji4[1]);
			_mm256_store_pd(&(c->element[2][index]), aji4[2]);
			_mm256_store_pd(&(c->element[3][index]), aji4[3]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d aji8[QDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji8[0] = _mm512_set_pd(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aji8[1] = _mm512_set_pd(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aji8[2] = _mm512_set_pd(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aji8[3] = _mm512_set_pd(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), aji8[0]);
			_mm512_store_pd(&(c->element[1][index]), aji8[1]);
			_mm512_store_pd(&(c->element[2][index]), aji8[2]);
			_mm512_store_pd(&(c->element[3][index]), aji8[3]);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat64_t aji2_0, aji2_1, aji2_2, aji2_3;
    svint64_t vidx;
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aji2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aji2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            aji2_2 = svld1_gather_s64index_f64(pg, a->element[2], vidx);
            aji2_3 = svld1_gather_s64index_f64(pg, a->element[3], vidx);
            index = i * real_col_dim + j;
            svst1_f64(pg, &(c->element[0][index]), aji2_0);
            svst1_f64(pg, &(c->element[1][index]), aji2_1);
            svst1_f64(pg, &(c->element[2][index]), aji2_2);
            svst1_f64(pg, &(c->element[3][index]), aji2_3);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t aji2[QDSIZE];
    double __attribute__ ((aligned(16))) buf[QDSIZE][2];

    for(i = 0; i < real_row_dim; i++)
    {
        //for(j = 0; j < a->col_dim; j++)
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            // gather a(j  ,i), a(j+1,i) for each QD component
            buf[0][0] = a->element[0][(j    ) * real_col_dim + i];
            buf[0][1] = a->element[0][(j + 1) * real_col_dim + i];
            buf[1][0] = a->element[1][(j    ) * real_col_dim + i];
            buf[1][1] = a->element[1][(j + 1) * real_col_dim + i];
            buf[2][0] = a->element[2][(j    ) * real_col_dim + i];
            buf[2][1] = a->element[2][(j + 1) * real_col_dim + i];
            buf[3][0] = a->element[3][(j    ) * real_col_dim + i];
            buf[3][1] = a->element[3][(j + 1) * real_col_dim + i];

            aji2[0] = vld1q_f64(buf[0]);
            aji2[1] = vld1q_f64(buf[1]);
            aji2[2] = vld1q_f64(buf[2]);
            aji2[3] = vld1q_f64(buf[3]);

            index = i * real_col_dim + j;
            vst1q_f64(&(c->element[0][index]), aji2[0]);
            vst1q_f64(&(c->element[1][index]), aji2[1]);
            vst1q_f64(&(c->element[2][index]), aji2[2]);
            vst1q_f64(&(c->element[3][index]), aji2[3]);
        }
    }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_qdmatrix_ij(c, i, j, get_qdmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), _mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][index]), _mm256_load_pd(&(a->element[1][index])));
			_mm256_store_pd(&(c->element[2][index]), _mm256_load_pd(&(a->element[2][index])));
			_mm256_store_pd(&(c->element[3][index]), _mm256_load_pd(&(a->element[3][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), _mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][index]), _mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[2][index]), _mm512_load_pd(&(a->element[2][index])));
			_mm512_store_pd(&(c->element[3][index]), _mm512_load_pd(&(a->element[3][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t t0;
	svfloat64_t t1;
	svfloat64_t t2;
	svfloat64_t t3;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
		svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(real_col_dim));
            index = i * real_col_dim + j;
            t0 = svld1_f64(pg, &(a->element[0][index]));
            t1 = svld1_f64(pg, &(a->element[1][index]));
            t2 = svld1_f64(pg, &(a->element[2][index]));
            t3 = svld1_f64(pg, &(a->element[3][index]));
            svst1_f64(pg, &(c->element[0][index]), t0);
            svst1_f64(pg, &(c->element[1][index]), t1);
            svst1_f64(pg, &(c->element[2][index]), t2);
            svst1_f64(pg, &(c->element[3][index]), t3);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t t0, t1, t2, t3;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;
            t0 = vld1q_f64(&(a->element[0][index]));
            t1 = vld1q_f64(&(a->element[1][index]));
            t2 = vld1q_f64(&(a->element[2][index]));
            t3 = vld1q_f64(&(a->element[3][index]));
            vst1q_f64(&(c->element[0][index]), t0);
            vst1q_f64(&(c->element[1][index]), t1);
            vst1q_f64(&(c->element[2][index]), t2);
            vst1q_f64(&(c->element[3][index]), t3);
        }
    }
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qdmatrix_ij(c, i, j, get_qdmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := -a */
void neg_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: neg_qdmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[QDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][index])));
			tmp[2] = _bncavx2_dneg(_mm256_load_pd(&(a->element[2][index])));
			tmp[3] = _bncavx2_dneg(_mm256_load_pd(&(a->element[3][index])));
			_mm256_store_pd(&(c->element[0][index]), tmp[0]); //_mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][index]), tmp[1]); //_mm256_load_pd(&(a->element[1][index])));
			_mm256_store_pd(&(c->element[2][index]), tmp[2]); //_mm256_load_pd(&(a->element[2][index])));
			_mm256_store_pd(&(c->element[3][index]), tmp[3]); //_mm256_load_pd(&(a->element[2][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[QDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][index])));
			tmp[2] = _bncavx512_dneg(_mm512_load_pd(&(a->element[2][index])));
			tmp[3] = _bncavx512_dneg(_mm512_load_pd(&(a->element[3][index])));
			_mm512_store_pd(&(c->element[0][index]), tmp[0]); //_mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][index]), tmp[1]); //_mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[2][index]), tmp[2]); //_mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[3][index]), tmp[3]); //_mm512_load_pd(&(a->element[1][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t t0;
	svfloat64_t t1;
	svfloat64_t t2;
	svfloat64_t t3;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
		svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(real_col_dim));
            index = i * real_col_dim + j;
            t0 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[0][index])));
            t1 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[1][index])));
            t2 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[2][index])));
            t3 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[3][index])));
            svst1_f64(pg, &(c->element[0][index]), t0);
            svst1_f64(pg, &(c->element[1][index]), t1);
            svst1_f64(pg, &(c->element[2][index]), t2);
            svst1_f64(pg, &(c->element[3][index]), t3);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t t0, t1, t2, t3;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;
            t0 = vnegq_f64(vld1q_f64(&(a->element[0][index])));
            t1 = vnegq_f64(vld1q_f64(&(a->element[1][index])));
            t2 = vnegq_f64(vld1q_f64(&(a->element[2][index])));
            t3 = vnegq_f64(vld1q_f64(&(a->element[3][index])));
            vst1q_f64(&(c->element[0][index]), t0);
            vst1q_f64(&(c->element[1][index]), t1);
            vst1q_f64(&(c->element[2][index]), t2);
            vst1q_f64(&(c->element[3][index]), t3);
        }
    }
#else // others
	double tmp[QDSIZE];

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_neg(tmp, get_qdmatrix_ij(a, i, j));
			set_qdmatrix_ij(c, i, j, tmp); //get_qdmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_qdmatrix(QDMatrix c)
{
	long int i, j;
	long int real_total_dim;
	double tmp1[QDSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[0][i]), zero4);
		_mm256_store_pd(&(c->element[1][i]), zero4);
		_mm256_store_pd(&(c->element[2][i]), zero4);
		_mm256_store_pd(&(c->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[0][i]), zero8);
		_mm512_store_pd(&(c->element[1][i]), zero8);
		_mm512_store_pd(&(c->element[2][i]), zero8);
		_mm512_store_pd(&(c->element[3][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t zero2;

    zero2 = svdup_f64(0.0);
    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(c->element[0][i]), zero2);
        svst1_f64(pg, &(c->element[1][i]), zero2);
        svst1_f64(pg, &(c->element[2][i]), zero2);
        svst1_f64(pg, &(c->element[3][i]), zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t zero2;

    zero2 = vdupq_n_f64(0.0);
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(c->element[0][i]), zero2);
        vst1q_f64(&(c->element[1][i]), zero2);
        vst1q_f64(&(c->element[2][i]), zero2);
        vst1q_f64(&(c->element[3][i]), zero2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
		c->element[2][i] = 0.0;	
		c->element[3][i] = 0.0;	
	}
#endif // __AVX2__

	rqd_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_qdmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j;
	double tmp[QDSIZE], tmp1[QDSIZE];

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrix_qdvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256d tmp4[QDSIZE], tmp1_4[QDSIZE];
	__m256d aij4[QDSIZE], vbj4[QDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rqd_set_ui(tmp, 0UL);
		_bncavx2_set0_qd(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_pd(&(a->element[1][ij_index]));
			aij4[2] = _mm256_load_pd(&(a->element[2][ij_index]));
			aij4[3] = _mm256_load_pd(&(a->element[3][ij_index]));
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_pd(&(vb->element[2][j]));
			vbj4[3] = _mm256_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx2_rqd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rqd_add(tmp4, tmp4, tmp1_4);
		}
		//set_qdvector_i(v, i, tmp);
		_bncavx2_rqd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__AVX512F__) // __AVX512F__
#if 0
	for(i = 0; i < a->row_dim; i++)
	{
		rqd_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			rqd_add(tmp, tmp, tmp1);
		}
		set_qdvector_i(v, i, tmp);
	}
#endif //if 0
	long int ij_index, real_col_dim;
	__m512d tmp8[QDSIZE], tmp1_8[QDSIZE];
	__m512d aij8[QDSIZE], vbj8[QDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_qd(tmp8);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[1] = _mm512_load_pd(&(a->element[1][ij_index]));
			aij8[2] = _mm512_load_pd(&(a->element[2][ij_index]));
			aij8[3] = _mm512_load_pd(&(a->element[3][ij_index]));
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[2] = _mm512_load_pd(&(vb->element[2][j]));
			vbj8[3] = _mm512_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx512_rqd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rqd_add(tmp8, tmp8, tmp1_8);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx512_rqd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}
//#endif // if 0

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    long int ij_index, real_col_dim;
    
	svfloat64_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;
	svfloat64_t tmp1_4_0, tmp1_4_1, tmp1_4_2, tmp1_4_3;
	svfloat64_t aij4_0, aij4_1, aij4_2, aij4_3;
	svfloat64_t vbj4_0, vbj4_1, vbj4_2, vbj4_3;
    real_col_dim = a->real_col_dim;
    for(i = 0; i < a->row_dim; i++)
    {
        _bncsve2_rqd_set0(&tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3);
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col_dim);
            ij_index = i * real_col_dim + j;
            aij4_0 = svld1_f64(pg, &(a->element[0][ij_index]));
            aij4_1 = svld1_f64(pg, &(a->element[1][ij_index]));
            aij4_2 = svld1_f64(pg, &(a->element[2][ij_index]));
            aij4_3 = svld1_f64(pg, &(a->element[3][ij_index]));
            vbj4_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj4_1 = svld1_f64(pg, &(vb->element[1][j]));
            vbj4_2 = svld1_f64(pg, &(vb->element[2][j]));
            vbj4_3 = svld1_f64(pg, &(vb->element[3][j]));
            _bncsve2_rqd_mul(svptrue_b64(), &tmp1_4_0, &tmp1_4_1, &tmp1_4_2, &tmp1_4_3, aij4_0, aij4_1, aij4_2, aij4_3, vbj4_0, vbj4_1, vbj4_2, vbj4_3);
            _bncsve2_rqd_add(svptrue_b64(), &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3, tmp1_4_0, tmp1_4_1, tmp1_4_2, tmp1_4_3);
        }
        _bncsve2_rqd_sum128d(svptrue_b64(), tmp, tmp4_0, tmp4_1, tmp4_2, tmp4_3);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
        v->element[3][i] = tmp[3];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long int ij_index, real_col_dim;
    float64x2_t tmp4[QDSIZE], tmp1_4[QDSIZE], aij4[QDSIZE], vbj4[QDSIZE];
    real_col_dim = a->real_col_dim;
    for(i = 0; i < a->row_dim; i++)
    {
        _bncneon_set0_qd(tmp4);
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            ij_index = i * real_col_dim + j;
            aij4[0] = vld1q_f64(&(a->element[0][ij_index]));
            aij4[1] = vld1q_f64(&(a->element[1][ij_index]));
            aij4[2] = vld1q_f64(&(a->element[2][ij_index]));
            aij4[3] = vld1q_f64(&(a->element[3][ij_index]));
            vbj4[0] = vld1q_f64(&(vb->element[0][j]));
            vbj4[1] = vld1q_f64(&(vb->element[1][j]));
            vbj4[2] = vld1q_f64(&(vb->element[2][j]));
            vbj4[3] = vld1q_f64(&(vb->element[3][j]));
            _bncneon_rqd_mul(tmp1_4, aij4, vbj4);
            _bncneon_rqd_add(tmp4, tmp4, tmp1_4);
        }
        _bncneon_rqd_sum128d(tmp, tmp4);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
        v->element[3][i] = tmp[3];
    }
#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rqd_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			rqd_add(tmp, tmp, tmp1);
		}
		set_qdvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j;
	double tmp[QDSIZE], tmp1[QDSIZE];

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrixt_ddvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256d tmp4[QDSIZE], tmp1_4[QDSIZE];
	__m256d aij4[QDSIZE], vbj4[QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rqd_set_ui(tmp, 0UL);
		_bncavx2_set0_qd(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij4[0] = _mm256_set_pd(
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij4[1] = _mm256_set_pd(
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij4[2] = _mm256_set_pd(
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aij4[3] = _mm256_set_pd(
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_pd(&(vb->element[2][j]));
			vbj4[3] = _mm256_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx2_rqd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rqd_add(tmp4, tmp4, tmp1_4);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rqd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ji_index, real_row_dim, real_col_dim;
	__m512d tmp8[QDSIZE], tmp1_8[QDSIZE];
	__m512d aij8[QDSIZE], vbj8[QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_qd(tmp8);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij8[0] = _mm512_set_pd(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij8[1] = _mm512_set_pd(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij8[2] = _mm512_set_pd(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aij8[3] = _mm512_set_pd(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[2] = _mm512_load_pd(&(vb->element[2][j]));
			vbj8[3] = _mm512_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx512_rqd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rqd_add(tmp8, tmp8, tmp1_8);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx512_rqd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}


#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    long int real_row_dim, real_col_dim;
    svfloat64_t tmp2_0, tmp2_1, tmp2_2, tmp2_3;
    svfloat64_t tmp1_2_0, tmp1_2_1, tmp1_2_2, tmp1_2_3;
    svfloat64_t aij2_0, aij2_1, aij2_2, aij2_3;
    svfloat64_t vbj2_0, vbj2_1, vbj2_2, vbj2_3;
    svint64_t vidx;

    real_row_dim = a->real_row_dim;
    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->col_dim; i++)
    {
        _bncsve2_rqd_set0(&tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3);
        for(j = 0; j < real_row_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_row_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aij2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aij2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            aij2_2 = svld1_gather_s64index_f64(pg, a->element[2], vidx);
            aij2_3 = svld1_gather_s64index_f64(pg, a->element[3], vidx);
            vbj2_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj2_1 = svld1_f64(pg, &(vb->element[1][j]));
            vbj2_2 = svld1_f64(pg, &(vb->element[2][j]));
            vbj2_3 = svld1_f64(pg, &(vb->element[3][j]));
            _bncsve2_rqd_mul(svptrue_b64(), &tmp1_2_0, &tmp1_2_1, &tmp1_2_2, &tmp1_2_3, aij2_0, aij2_1, aij2_2, aij2_3, vbj2_0, vbj2_1, vbj2_2, vbj2_3);
            _bncsve2_rqd_add(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, &tmp2_3, tmp2_0, tmp2_1, tmp2_2, tmp2_3, tmp1_2_0, tmp1_2_1, tmp1_2_2, tmp1_2_3);
        }
        _bncsve2_rqd_sum128d(svptrue_b64(), tmp, tmp2_0, tmp2_1, tmp2_2, tmp2_3);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
        v->element[3][i] = tmp[3];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long real_row_dim = a->real_row_dim;
    float64x2_t tmp4[QDSIZE], tmp1_4[QDSIZE], aij4[QDSIZE], vbi4[QDSIZE];
	double __attribute__ ((aligned(64))) aij4l[QDSIZE][2];
    for(j = 0; j < a->col_dim; j++)
    {
        _bncneon_set0_qd(tmp4);
        for(i = 0; i < real_row_dim; i += _BNC_D_WIDTH)
        {
			aij4l[0][0] = a->element[0][(i    ) * a->real_col_dim + j];
			aij4l[0][1] = a->element[0][(i + 1) * a->real_col_dim + j];
			aij4[0] = vld1q_f64(aij4l[0]); // &(a->element[0][i * a->real_col_dim + j]));

			aij4l[1][0] = a->element[1][(i    ) * a->real_col_dim + j];
			aij4l[1][1] = a->element[1][(i + 1) * a->real_col_dim + j];
			aij4[1] = vld1q_f64(aij4l[1]); //&(a->element[1][i * a->real_col_dim + j]));

			aij4l[2][0] = a->element[2][(i    ) * a->real_col_dim + j];
			aij4l[2][1] = a->element[2][(i + 1) * a->real_col_dim + j];
			aij4[2] = vld1q_f64(aij4l[2]); //&(a->element[2][i * a->real_col_dim + j]));

			aij4l[3][0] = a->element[3][(i    ) * a->real_col_dim + j];
			aij4l[3][1] = a->element[3][(i + 1) * a->real_col_dim + j];
			aij4[3] = vld1q_f64(aij4l[3]); //(&(a->element[3][i * a->real_col_dim + j]));

			vbi4[0] = vld1q_f64(&(vb->element[0][i]));
            vbi4[1] = vld1q_f64(&(vb->element[1][i]));
            vbi4[2] = vld1q_f64(&(vb->element[2][i]));
            vbi4[3] = vld1q_f64(&(vb->element[3][i]));
            _bncneon_rqd_mul(tmp1_4, aij4, vbi4);
            _bncneon_rqd_add(tmp4, tmp4, tmp1_4);
        }
        _bncneon_rqd_sum128d(tmp, tmp4);
        v->element[0][j] = tmp[0];
        v->element[1][j] = tmp[1];
        v->element[2][j] = tmp[2];
        v->element[3][j] = tmp[3];
    }
#else // others

	for(i = 0; i < a->col_dim; i++)
	{
		set0_qd(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rqd_mul(tmp1, get_qdmatrix_ij(a, j, i), get_qdvector_i(vb, j));
			rqd_add(tmp, tmp, tmp1);
		}
		set_qdvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_qdmatrix(QDMatrix a)
{
	long int i, j, k, dim;
	double tmp[QDSIZE], aii[QDSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_qdmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rqd_cmp_ui(get_qdmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_qdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rqd_ui_div(aii, 1UL, get_qdmatrix_ij(a, i, i));
		set_qdmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rqd_mul(tmp, get_qdmatrix_ij(a, i, j), aii);
			set_qdmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rqd_mul(tmp, get_qdmatrix_ij(a, i, j), aii);
			set_qdmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rqd_neg(tmp, aii); /* tmp := -aii */
			rqd_mul(tmp, tmp, get_qdmatrix_ij(a, j, i));
			set_qdmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rqd_neg(tmp, aii); /* tmp := -aii */
			rqd_mul(tmp, tmp, get_qdmatrix_ij(a, j, i));
			set_qdmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_qdvec(MPFVector c, QDVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_qd(tmp, get_qdvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_qdvector_mpfvec(QDVector c, MPFVector a)
{
	long int i;
	double tmp[QDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_qd(tmp, get_mpfvector_i(a, i));
		set_qdvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_qdmat(MPFMatrix c, QDMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_qd(tmp, get_qdmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_qdmatrix_mpfmat(QDMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp[QDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_qd(tmp, get_mpfmatrix_ij(a, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_qdvector_mpfvec(double relerr[QDSIZE], QDVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	mpf_t mpf_relerr, norm_true_vec, norm_diff_vec;
	MPFVector diff_vec, mpf_approx_vec;

	prec = true_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_approx_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(mpf_relerr, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(norm_diff_vec, prec);

	// mpf_approx_vec := (mpf)approx_vec
	subst_mpfvector_qdvec(mpf_approx_vec, approx_vec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, mpf_approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	// mpf_relerr := ||approx_vec - true_vec||
	mpf_set(mpf_relerr, norm_diff_vec);

	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(mpf_relerr, norm_diff_vec, norm_true_vec);

	// relerr := (DD)mpf_relerr
	//mpf_get_dd(relerr, mpf_relerr);
	mpf_get_dd(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_qdvector_mpf(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	long int i;
	mpf_t abs_true_vec, abs_diff_vec, norm_diff_vec, norm_true_vec, mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr;
	MPFVector mpf_approx_vec, diff_vec;

	prec = true_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_approx_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(abs_true_vec, prec);
	mpf_init2(abs_diff_vec, prec);
	mpf_init2(norm_diff_vec, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(mpf_max_relerr, prec);
	mpf_init2(mpf_min_relerr, prec);
	mpf_init2(mpf_norm_relerr, prec);

	// mpf_approx_vec := (mpf)approx_vec
	subst_mpfvector_qdvec(mpf_approx_vec, approx_vec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, mpf_approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	mpf_set(mpf_norm_relerr, norm_diff_vec);
	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(mpf_norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	mpf_set_ui(mpf_max_relerr, 0UL);
	normi_mpfvector(mpf_min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		mpf_abs(abs_diff_vec, get_mpfvector_i(diff_vec, i));
		mpf_abs(abs_true_vec, get_mpfvector_i(true_vec, i));
		if(mpf_cmp_ui(abs_true_vec, 0UL) != 0)
			mpf_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(mpf_cmp(mpf_max_relerr, abs_diff_vec) < 0)
			mpf_set(mpf_max_relerr, abs_diff_vec);
		if(mpf_cmp(mpf_min_relerr, abs_diff_vec) > 0)
			mpf_set(mpf_min_relerr, abs_diff_vec);
	}

	// relerr := (DD)mpf_relerr
	mpf_get_qd(max_relerr, mpf_max_relerr);
	mpf_get_qd(min_relerr, mpf_min_relerr);
	mpf_get_qd(norm_relerr, mpf_norm_relerr);

	free_mpfvector(diff_vec);// Fix! 2012-06-03 by T.Kouya
	free_mpfvector(mpf_approx_vec);
	mpf_clear(abs_true_vec);
	mpf_clear(abs_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(mpf_max_relerr);
	mpf_clear(mpf_min_relerr);
	mpf_clear(mpf_norm_relerr);

	return;
}
#endif // USE_GMP

//---------------------------------/
// QD <-> double
//---------------------------------/
#if 0
/* c := (d)a */
void subst_qdvector_dvec(QDVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_qdvector_i_d(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_dvector_qdvec(DVector c, QDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rqd_get_d(get_qdvector_i(a, i));
	}
}


/* c := (qd)a */
void subst_qdmatrix_dmat(QDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qdmatrix_ij_d(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_dmatrix_qdmat(DMatrix c, QDMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_qdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (c->col_dim) + j;
			c->element[ij_index] = rqd_get_d(get_qdmatrix_ij(a, i, j));
		}
	}
}
#endif // 0

#ifdef USE_DDLINEAR
/* c := (dd)a */
void subst_qdvector_ddvec(QDVector c, DDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_qdvector_i_dd(c, i, get_ddvector_i(a, i));
	}
}

/* c := (dd)a */
void subst_ddvector_qdvec(DDVector c, QDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_ddvector_i(c, i, get_qdvector_i_dd(a, i));
	}
}


/* c := (dd)a */
void subst_qdmatrix_ddmat(QDMatrix c, DDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qdmatrix_ij_dd(c, i, j, get_ddmatrix_ij(a, i, j));
		}
	}
}

/* c := (dd)a */
void subst_ddmatrix_qdmat(DDMatrix c, QDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_qdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij(c, i, j, get_qdmatrix_ij_dd(a, i, j));
		}
	}
}
#endif // USE_DDLINEAR

#ifdef USE_TDLINEAR
/* c := (td)a */
void subst_tdvector_qdvec(TDVector c, QDVector a)
{
	long int i;
	double tmp[TDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rtd_set_qd(tmp, get_qdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
}

/* c := (qd)a */
void subst_qdvector_tdvec(QDVector c, TDVector a)
{
	long int i;
	double tmp[QDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rqd_set_td(tmp, get_tdvector_i(a, i));
		set_qdvector_i_td(c, i, tmp);
	}
}

/* c := (TD)a */
void subst_tdmatrix_qdmat(TDMatrix c, QDMatrix a)
{
	long int i, j, ij_index;
	double td_c_ij[TDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix_qdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (a->real_col_dim) + j;
//			rqd_sub(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			td_c_ij[0] = a->element[0][ij_index];
			td_c_ij[1] = a->element[1][ij_index];
			td_c_ij[2] = a->element[2][ij_index];
			set_tdmatrix_ij(c, i, j, td_c_ij);
		}
	}
}


/* c := a */
void subst_qdmatrix_tdmat(QDMatrix c, TDMatrix a)
{
	long int i, j, ij_index;
	double qd_c_ij[QDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix_tdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (a->real_col_dim) + j;
//			rqd_sub(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			qd_c_ij[0] = a->element[0][ij_index];
			qd_c_ij[1] = a->element[1][ij_index];
			qd_c_ij[2] = a->element[2][ij_index];
			qd_c_ij[3] = 0.0;
			set_qdmatrix_ij(c, i, j, qd_c_ij);
		}
	}
}

/* (QDMatrix)c := (TDMatrix)a - (TDMatrix)b */
void sub_qdmatrix_tdmat_tdmat(QDMatrix c, TDMatrix a, TDMatrix b)
{
	long int i, j, row_dim, col_dim, ij_index;
	double tmp[QDSIZE], qd_a_ij[QDSIZE], qd_b_ij[QDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_tdmat_tdmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_tdmat_tdmat\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			ij_index = i * (a->real_col_dim) + j;
//			rqd_sub(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			qd_a_ij[0] = a->element[0][ij_index];
			qd_a_ij[1] = a->element[1][ij_index];
			qd_a_ij[2] = a->element[2][ij_index];
			qd_a_ij[3] = 0.0;
			qd_b_ij[0] = b->element[0][ij_index];
			qd_b_ij[1] = b->element[1][ij_index];
			qd_b_ij[2] = b->element[2][ij_index];
			qd_b_ij[3] = 0.0;
			rqd_sub(tmp, qd_a_ij, qd_b_ij);
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
}
#endif // USE_TDLINEAR

/* Normwise relative error of vector */
void relerr_qdvector(double relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type)
{
	double norm_true_vec[QDSIZE], norm_diff_vec[QDSIZE];
	QDVector diff_vec;

	diff_vec = init_qdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qdvector(norm_diff_vec, diff_vec);
			normi_qdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(norm_diff_vec, diff_vec);
			norm1_qdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(norm_diff_vec, diff_vec);
			norm2_qdvector(norm_true_vec, true_vec);
			break;
	}

	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(relerr, norm_diff_vec, norm_true_vec);

	free_qdvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_qdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type)
{
	double abs_true_vec[QDSIZE], abs_diff_vec[QDSIZE], norm_diff_vec[QDSIZE], norm_true_vec[QDSIZE];
	long int i;
	QDVector diff_vec;

	diff_vec = init_qdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qdvector(norm_diff_vec, diff_vec);
			normi_qdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(norm_diff_vec, diff_vec);
			norm1_qdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(norm_diff_vec, diff_vec);
			norm2_qdvector(norm_true_vec, true_vec);
			break;
	}

	rqd_set(norm_relerr, norm_diff_vec);
	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rqd_set_ui(max_relerr, 0UL);
	normi_qdvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rqd_abs(abs_diff_vec, get_qdvector_i(diff_vec, i));
		rqd_abs(abs_true_vec, get_qdvector_i(true_vec, i));
		if(rqd_cmp_ui(abs_true_vec, 0UL) != 0)
			rqd_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rqd_cmp(max_relerr, abs_diff_vec) < 0)
			rqd_set(max_relerr, abs_diff_vec);
		if(rqd_cmp(min_relerr, abs_diff_vec) > 0)
			rqd_set(min_relerr, abs_diff_vec);
	}

	free_qdvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

#if 0
// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_qdmatrix(QDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[QDSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rqd_set(tmp, get_qdmatrix_ij(mat, row_index0, i));
		set_qdmatrix_ij(mat, row_index0, i, get_qdmatrix_ij(mat, row_index1, i));
		set_qdmatrix_ij(mat, row_index1, i, tmp);
	}
}

//#if 0

// QD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QDLUdecomp(QDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, i));
			set_qdmatrix_ij(a, j, i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQDLS(QDVector answer, QDMatrix lu, QDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu: LU decomposed Matrix (given by user)   */
/*      QDVector b: constant vector (given by user)         */
/*      QDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	subst_qdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, j, i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, i, j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, i, i));
		set_qdvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QDLUdecompP(QDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rqd_abs(dmaxii, get_qdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, ch[j], i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, ch[j], i), get_qdmatrix_ij(a, ch[i], i));
			set_qdmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, ch[j], i), get_qdmatrix_ij(a, ch[i], k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, ch[j], k), dtmp1);
				set_qdmatrix_ij(a, ch[j], k, dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQDLSP(QDVector answer, QDMatrix lu, QDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QDVector b[]: constant vector (given by user)       */
/*      QDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, i, get_qdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, ch[i], i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, ch[j], i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, ch[i], j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, ch[i], i));
		set_qdvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QDLUdecompC(QDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a[]: Matrix (given by user)               */
/*       long int row_ch[]: Row order                       */
/*       long int col_ch[]: Column order                    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*  row_ch[]: Row order                                     */
/*  col_ch[]: Column order                                  */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rqd_abs(dmaxii, get_qdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_abs(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rqd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rqd_set(dmaxii, dtmp);
				}
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QDLUdecompC)!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = row_ch[imax];
			row_ch[imax] = row_ch[i];
			row_ch[i] = itmp;
		}
		if(jmax != i)
		{
			itmp = col_ch[jmax];
			col_ch[jmax] = col_ch[i];
			col_ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[i]), get_qdmatrix_ij(a, row_ch[i], col_ch[i]));
			set_qdmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, row_ch[j], col_ch[i]), get_qdmatrix_ij(a, row_ch[i], col_ch[k]));
				rqd_sub(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
				set_qdmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQDLSC(QDVector answer, QDMatrix lu, QDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix lu: LU decomposed Matrix (given by user)  */
/*       QDVector b: constant vector (given by user)        */
/*       QDVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, col_ch[i], get_qdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, row_ch[j], col_ch[i]), get_qdvector_i(answer, col_ch[i]));
			rqd_sub(dtmp, get_qdvector_i(answer, col_ch[j]), dtmp1);
			set_qdvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, row_ch[i], col_ch[j]), get_qdvector_i(answer, col_ch[j]));
			rqd_sub(dtmp, get_qdvector_i(answer, col_ch[i]), dtmp1);
			set_qdvector_i(answer, col_ch[i], dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, col_ch[i]), get_qdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_qdvector_i(answer, col_ch[i], dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QDLUdecompPM(QDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[QDSIZE], aji256[QDSIZE], ajk256[QDSIZE], aik256[QDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
#endif // __AVX2__

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_qdmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, i));
			set_qdmatrix_ij(a, j, i, dtmp);
		}
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji256[0] = _mm256_set_pd(
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji]
            );
			aji256[1] = _mm256_set_pd(
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji]
            );
			aji256[2] = _mm256_set_pd(
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji]
            );
			aji256[3] = _mm256_set_pd(
                a->element[3][index_ji],
                a->element[3][index_ji],
                a->element[3][index_ji],
                a->element[3][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);
//#if 0
			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik256[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_pd(&(a->element[2][index_ik]));
				aik256[3] = _mm256_load_pd(&(a->element[3][index_ik]));				
				_bncavx2_rqd_mul(dtmp256, aji256, aik256);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				ajk256[3] = _mm256_load_pd(&(a->element[3][index_jk]));
				_bncavx2_rqd_sub(dtmp256, ajk256, dtmp256);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[2]);
				_mm256_store_pd(&(a->element[3][index_jk]), dtmp256[3]);
			}
			//printf(", %ld middle", k);
		}
#elif defined(__AVX512F__) // __AVX512F__
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji256[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji256[1] = vdupq_n_f64(a->element[1][index_ji]);
			aji256[2] = vdupq_n_f64(a->element[2][index_ji]);
			aji256[3] = vdupq_n_f64(a->element[3][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik256[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik256[1] = vld1q_f64(&(a->element[1][index_ik]));
				aik256[2] = vld1q_f64(&(a->element[2][index_ik]));
				aik256[3] = vld1q_f64(&(a->element[3][index_ik]));
				_bncneon_rqd_mul(dtmp256, aji256, aik256);
				index_jk = j * a->real_col_dim + k;
				ajk256[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk256[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk256[2] = vld1q_f64(&(a->element[2][index_jk]));
				ajk256[3] = vld1q_f64(&(a->element[3][index_jk]));
				_bncneon_rqd_sub(dtmp256, ajk256, dtmp256);
				vst1q_f64(&(a->element[0][index_jk]), dtmp256[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp256[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp256[2]);
				vst1q_f64(&(a->element[3][index_jk]), dtmp256[3]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji_0 = svdup_f64(a->element[0][index_ji]);
			aji_1 = svdup_f64(a->element[1][index_ji]);
			aji_2 = svdup_f64(a->element[2][index_ji]);
			aji_3 = svdup_f64(a->element[3][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)dim_end);
				index_ik = i * a->real_col_dim + k;
				aik_0 = svld1_f64(pg, &(a->element[0][index_ik]));
				aik_1 = svld1_f64(pg, &(a->element[1][index_ik]));
				aik_2 = svld1_f64(pg, &(a->element[2][index_ik]));
				aik_3 = svld1_f64(pg, &(a->element[3][index_ik]));
				_bncsve2_rqd_mul(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, &dtmp_3, aji_0, aji_1, aji_2, aji_3, aik_0, aik_1, aik_2, aik_3);
				index_jk = j * a->real_col_dim + k;
				ajk_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				ajk_3 = svld1_f64(pg, &(a->element[3][index_jk]));
				_bncsve2_rqd_neg(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, &dtmp_3, dtmp_0, dtmp_1, dtmp_2, dtmp_3);
				_bncsve2_rqd_add(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, &dtmp_3, ajk_0, ajk_1, ajk_2, ajk_3, dtmp_0, dtmp_1, dtmp_2, dtmp_3);
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_2);
				svst1_f64(pg, &(a->element[3][index_jk]), dtmp_3);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                         (Octuple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QDVector b[]: constant vector (given by user)       */
/*      QDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, i, get_qdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, j, i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, i, j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, i, i));
		set_qdvector_i(answer, i, dtmp);
	}

	return 0;
}
#endif // 0

#ifdef __cplusplus
} // extern "C"
#endif
