/* cf_simd.h - hand-written SIMD kernels for native complex-float (cf).
 *
 * Data layout: AoS interleaved 'float _Complex' i.e. [re0,im0,re1,im1,...],
 * reinterpreted as `float*` of 2*n elements. Complex (a+bi)(c+di) =
 * (ac-bd)+(ad+bc)i is evaluated with deinterleaving loads (NEON vld2 / SVE svld2)
 * or shuffle+addsub/fmaddsub (AVX2 / AVX-512).
 *
 * Two primitives cover every cf op in the bench:
 *   cf_axpy(z,y,sre,sim,x,n) : z = y + alpha*x         (alpha = sre+sim*i)
 *                              with z==y it becomes dst += alpha*x  (matmul i-k-j)
 *   cf_dot (a,b,n,&re,&im)   : (re,im) = sum_k a[k]*b[k] (matvec / spmv row)
 *
 * Backend gating uses BOTH the BNC_ENABLE_* opt-in AND the arch feature macro,
 * because on AArch64 __ARM_NEON is always set (a scalar build must stay scalar).
 *
 *   SVE2 : -DBNC_ENABLE_SVE2  (&& __ARM_FEATURE_SVE2)
 *   NEON : -DBNC_ENABLE_NEON  (&& __ARM_NEON)
 *   AVX512: -DBNC_ENABLE_AVX512 (&& __AVX512F__)        [x86 only]
 *   AVX2 : -DBNC_ENABLE_AVX2  (&& __AVX2__)             [x86 only]
 *   else : portable scalar
 */
#ifndef CF_SIMD_H
#define CF_SIMD_H
#include <complex.h>

typedef float _Complex cf_t;

/* ---- pick active backend ------------------------------------------------ */
#if defined(BNC_ENABLE_SVE2) && defined(__ARM_FEATURE_SVE2)
#  define CF_SIMD_SVE2 1
#  define CF_BACKEND "sve2"
#  include <arm_sve.h>
#elif defined(BNC_ENABLE_NEON) && defined(__ARM_NEON)
#  define CF_SIMD_NEON 1
#  define CF_BACKEND "neon"
#  include <arm_neon.h>
#elif defined(BNC_ENABLE_AVX512) && defined(__AVX512F__)
#  define CF_SIMD_AVX512 1
#  define CF_BACKEND "avx512"
#  include <immintrin.h>
#elif defined(BNC_ENABLE_AVX2) && defined(__AVX2__)
#  define CF_SIMD_AVX2 1
#  define CF_BACKEND "avx2"
#  include <immintrin.h>
#else
#  define CF_SIMD_SCALAR 1
#  define CF_BACKEND "scalar"
#endif

/* ---- portable scalar reference (always available, used by --check) ------ */
static inline void cf_axpy_scalar(cf_t *z, const cf_t *y, float sre, float sim,
                                  const cf_t *x, long n){
    cf_t a = sre + sim*(cf_t)I;
    for(long k=0;k<n;k++) z[k] = y[k] + a*x[k];
}
static inline void cf_dot_scalar(const cf_t *a, const cf_t *b, long n,
                                 float *out_re, float *out_im){
    cf_t s = 0.0f;
    for(long k=0;k<n;k++) s += a[k]*b[k];
    *out_re = crealf(s); *out_im = cimagf(s);
}

/* ======================================================================== */
#if defined(CF_SIMD_NEON)
static inline void cf_axpy(cf_t *z, const cf_t *y, float sre, float sim,
                           const cf_t *x, long n){
    const float *xp=(const float*)x, *yp=(const float*)y; float *zp=(float*)z;
    float32x4_t are=vdupq_n_f32(sre), aim=vdupq_n_f32(sim);
    long k=0;
    for(; k+4<=n; k+=4){
        float32x4x2_t X=vld2q_f32(xp+2*k), Y=vld2q_f32(yp+2*k), Z;
        float32x4_t pr=vsubq_f32(vmulq_f32(are,X.val[0]), vmulq_f32(aim,X.val[1]));
        float32x4_t pi=vaddq_f32(vmulq_f32(are,X.val[1]), vmulq_f32(aim,X.val[0]));
        Z.val[0]=vaddq_f32(Y.val[0],pr);
        Z.val[1]=vaddq_f32(Y.val[1],pi);
        vst2q_f32(zp+2*k,Z);
    }
    for(; k<n; k++) z[k]=y[k]+(sre+sim*(cf_t)I)*x[k];
}
static inline void cf_dot(const cf_t *a, const cf_t *b, long n,
                          float *out_re, float *out_im){
    const float *ap=(const float*)a, *bp=(const float*)b;
    float32x4_t accr=vdupq_n_f32(0.0f), acci=vdupq_n_f32(0.0f);
    long k=0;
    for(; k+4<=n; k+=4){
        float32x4x2_t A=vld2q_f32(ap+2*k), B=vld2q_f32(bp+2*k);
        accr=vaddq_f32(accr, vsubq_f32(vmulq_f32(A.val[0],B.val[0]), vmulq_f32(A.val[1],B.val[1])));
        acci=vaddq_f32(acci, vaddq_f32(vmulq_f32(A.val[0],B.val[1]), vmulq_f32(A.val[1],B.val[0])));
    }
    float re=vaddvq_f32(accr), im=vaddvq_f32(acci);
    for(; k<n; k++){ re+=crealf(a[k])*crealf(b[k])-cimagf(a[k])*cimagf(b[k]);
                     im+=crealf(a[k])*cimagf(b[k])+cimagf(a[k])*crealf(b[k]); }
    *out_re=re; *out_im=im;
}

/* ======================================================================== */
#elif defined(CF_SIMD_SVE2)
static inline void cf_axpy(cf_t *z, const cf_t *y, float sre, float sim,
                           const cf_t *x, long n){
    const float *xp=(const float*)x, *yp=(const float*)y; float *zp=(float*)z;
    svfloat32_t are=svdup_f32(sre), aim=svdup_f32(sim);
    long k=0; long vl=svcntw();   /* float lanes; complex per step = vl/... use whilelt on n */
    for(svbool_t pg=svwhilelt_b32((uint64_t)k,(uint64_t)n); svptest_any(svptrue_b32(),pg);
        k+=vl, pg=svwhilelt_b32((uint64_t)k,(uint64_t)n)){
        svfloat32x2_t X=svld2_f32(pg, xp+2*k), Y=svld2_f32(pg, yp+2*k);
        svfloat32_t xr=svget2_f32(X,0), xi=svget2_f32(X,1);
        svfloat32_t pr=svsub_f32_x(pg, svmul_f32_x(pg,are,xr), svmul_f32_x(pg,aim,xi));
        svfloat32_t pi=svadd_f32_x(pg, svmul_f32_x(pg,are,xi), svmul_f32_x(pg,aim,xr));
        svfloat32x2_t Z=svcreate2_f32(svadd_f32_x(pg, svget2_f32(Y,0), pr),
                                      svadd_f32_x(pg, svget2_f32(Y,1), pi));
        svst2_f32(pg, zp+2*k, Z);
    }
}
static inline void cf_dot(const cf_t *a, const cf_t *b, long n,
                          float *out_re, float *out_im){
    const float *ap=(const float*)a, *bp=(const float*)b;
    svfloat32_t accr=svdup_f32(0.0f), acci=svdup_f32(0.0f);
    long k=0; long vl=svcntw();
    for(svbool_t pg=svwhilelt_b32((uint64_t)k,(uint64_t)n); svptest_any(svptrue_b32(),pg);
        k+=vl, pg=svwhilelt_b32((uint64_t)k,(uint64_t)n)){
        svfloat32x2_t A=svld2_f32(pg, ap+2*k), B=svld2_f32(pg, bp+2*k);
        svfloat32_t ar=svget2_f32(A,0), ai=svget2_f32(A,1);
        svfloat32_t br=svget2_f32(B,0), bi=svget2_f32(B,1);
        accr=svadd_f32_x(pg, accr, svsub_f32_x(pg, svmul_f32_x(pg,ar,br), svmul_f32_x(pg,ai,bi)));
        acci=svadd_f32_x(pg, acci, svadd_f32_x(pg, svmul_f32_x(pg,ar,bi), svmul_f32_x(pg,ai,br)));
    }
    *out_re=svaddv_f32(svptrue_b32(), accr);
    *out_im=svaddv_f32(svptrue_b32(), acci);
}

/* ======================================================================== */
#elif defined(CF_SIMD_AVX512)
/* 16 floats = 8 complex per 512-bit register */
static inline void cf_axpy(cf_t *z, const cf_t *y, float sre, float sim,
                           const cf_t *x, long n){
    const float *xp=(const float*)x, *yp=(const float*)y; float *zp=(float*)z;
    __m512 br=_mm512_set1_ps(sre), bi=_mm512_set1_ps(sim);
    long k=0;
    for(; k+8<=n; k+=8){
        __m512 X=_mm512_loadu_ps(xp+2*k);
        __m512 Xsw=_mm512_permute_ps(X,0xB1);            /* swap re/im in each pair */
        __m512 t=_mm512_mul_ps(Xsw,bi);
        __m512 p=_mm512_fmaddsub_ps(X,br,t);             /* even:Xr*sre-Xi*sim, odd:Xi*sre+Xr*sim */
        __m512 Y=_mm512_loadu_ps(yp+2*k);
        _mm512_storeu_ps(zp+2*k, _mm512_add_ps(Y,p));
    }
    for(; k<n; k++) z[k]=y[k]+(sre+sim*(cf_t)I)*x[k];
}
static inline void cf_dot(const cf_t *a, const cf_t *b, long n,
                          float *out_re, float *out_im){
    const float *ap=(const float*)a, *bp=(const float*)b;
    __m512 acc=_mm512_setzero_ps();
    long k=0;
    for(; k+8<=n; k+=8){
        __m512 A=_mm512_loadu_ps(ap+2*k), B=_mm512_loadu_ps(bp+2*k);
        __m512 br=_mm512_moveldup_ps(B);                 /* (br,br,...) */
        __m512 bi=_mm512_movehdup_ps(B);                 /* (bi,bi,...) */
        __m512 Asw=_mm512_permute_ps(A,0xB1);            /* (ai,ar,...) */
        __m512 t=_mm512_mul_ps(Asw,bi);
        __m512 prod=_mm512_fmaddsub_ps(A,br,t);          /* even:ar*br-ai*bi, odd:ai*br+ar*bi */
        acc=_mm512_add_ps(acc,prod);
    }
    float buf[16]; _mm512_storeu_ps(buf,acc);
    float re=0.0f, im=0.0f;
    for(int j=0;j<16;j+=2){ re+=buf[j]; im+=buf[j+1]; }
    for(; k<n; k++){ re+=crealf(a[k])*crealf(b[k])-cimagf(a[k])*cimagf(b[k]);
                     im+=crealf(a[k])*cimagf(b[k])+cimagf(a[k])*crealf(b[k]); }
    *out_re=re; *out_im=im;
}

/* ======================================================================== */
#elif defined(CF_SIMD_AVX2)
/* 8 floats = 4 complex per 256-bit register */
static inline void cf_axpy(cf_t *z, const cf_t *y, float sre, float sim,
                           const cf_t *x, long n){
    const float *xp=(const float*)x, *yp=(const float*)y; float *zp=(float*)z;
    __m256 br=_mm256_set1_ps(sre), bi=_mm256_set1_ps(sim);
    long k=0;
    for(; k+4<=n; k+=4){
        __m256 X=_mm256_loadu_ps(xp+2*k);
        __m256 Xsw=_mm256_permute_ps(X,0xB1);
        __m256 t1=_mm256_mul_ps(X,br);
        __m256 t2=_mm256_mul_ps(Xsw,bi);
        __m256 p=_mm256_addsub_ps(t1,t2);                /* even:Xr*sre-Xi*sim, odd:Xi*sre+Xr*sim */
        __m256 Y=_mm256_loadu_ps(yp+2*k);
        _mm256_storeu_ps(zp+2*k, _mm256_add_ps(Y,p));
    }
    for(; k<n; k++) z[k]=y[k]+(sre+sim*(cf_t)I)*x[k];
}
static inline void cf_dot(const cf_t *a, const cf_t *b, long n,
                          float *out_re, float *out_im){
    const float *ap=(const float*)a, *bp=(const float*)b;
    __m256 acc=_mm256_setzero_ps();
    long k=0;
    for(; k+4<=n; k+=4){
        __m256 A=_mm256_loadu_ps(ap+2*k), B=_mm256_loadu_ps(bp+2*k);
        __m256 br=_mm256_moveldup_ps(B);
        __m256 bi=_mm256_movehdup_ps(B);
        __m256 Asw=_mm256_permute_ps(A,0xB1);
        __m256 t1=_mm256_mul_ps(A,br);
        __m256 t2=_mm256_mul_ps(Asw,bi);
        __m256 prod=_mm256_addsub_ps(t1,t2);             /* even:ar*br-ai*bi, odd:ai*br+ar*bi */
        acc=_mm256_add_ps(acc,prod);
    }
    float buf[8]; _mm256_storeu_ps(buf,acc);
    float re=buf[0]+buf[2]+buf[4]+buf[6];
    float im=buf[1]+buf[3]+buf[5]+buf[7];
    for(; k<n; k++){ re+=crealf(a[k])*crealf(b[k])-cimagf(a[k])*cimagf(b[k]);
                     im+=crealf(a[k])*cimagf(b[k])+cimagf(a[k])*crealf(b[k]); }
    *out_re=re; *out_im=im;
}

/* ======================================================================== */
#else  /* CF_SIMD_SCALAR */
static inline void cf_axpy(cf_t *z, const cf_t *y, float sre, float sim,
                           const cf_t *x, long n){ cf_axpy_scalar(z,y,sre,sim,x,n); }
static inline void cf_dot(const cf_t *a, const cf_t *b, long n,
                          float *out_re, float *out_im){ cf_dot_scalar(a,b,n,out_re,out_im); }
#endif

#endif /* CF_SIMD_H */
