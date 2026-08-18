// -------------------------------------------------------
// _bncsve2_d.h  --  SVE2 double vector routines
// 2025-08-05  Tomonori Kouya
//
// NEONとの対応表:
//   float64x2_t           -> svfloat64_t  (可変長)
//   vdupq_n_f64(x)        -> svdup_f64(x)
//   vld1q_f64(ptr)        -> svld1_f64(pg, ptr)
//   vst1q_f64(ptr, v)     -> svst1_f64(pg, ptr, v)
//   vaddq_f64(a,b)        -> svadd_f64_x(pg,a,b)
//   vsubq_f64(a,b)        -> svsub_f64_x(pg,a,b)
//   vmulq_f64(a,b)        -> svmul_f64_x(pg,a,b)
//   vdivq_f64(a,b)        -> svdiv_f64_x(pg,a,b)
//   vabsq_f64(a)          -> svabs_f64_x(pg,a)
//   vnegq_f64(a)          -> svneg_f64_x(pg,a)
//   vfmaq_f64(c,a,b)      -> svmad_f64_x(pg,a,b,c)  [a*b+c]
//   vmlaq_f64(acc,a,b)    -> svmla_f64_x(pg,acc,a,b) [acc+a*b]
//   vaddvq_f64(v)         -> svaddv_f64(pg,v)
// -------------------------------------------------------
#ifndef __BNCSVE2_D_H
#define __BNCSVE2_D_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

// ----------------------------------------------------------------
//  _bncsve2_fabs : ret[i] = |a[i]|
// ----------------------------------------------------------------
static inline svfloat64_t _bncsve2_fabs(svbool_t pg, svfloat64_t a)
{
    return svabs_f64_x(pg, a);
}

// ----------------------------------------------------------------
//  _bncsve2_dneg : ret[i] = -a[i]
// ----------------------------------------------------------------
static inline svfloat64_t _bncsve2_dneg(svbool_t pg, svfloat64_t a)
{
    return svneg_f64_x(pg, a);
}

// ----------------------------------------------------------------
//  _bncsve2_dsum_all : Σ a[i]  (全アクティブ要素の総和)
//  NEON版 _bncneon_dsum128 の可変長版
// ----------------------------------------------------------------
static inline double _bncsve2_dsum_all(svbool_t pg, svfloat64_t a)
{
    return svaddv_f64(pg, a);
}

// ----------------------------------------------------------------
//  _bncsve2_dfma : ret[i] = a[i]*b[i] + c[i]
//  svmad_f64_x(pg, a, b, c) = a*b + c
// ----------------------------------------------------------------
static void _bncsve2_dfma(
        double ret[], const double a[], const double b[], const double c[],
        int dim)
{
    svbool_t pg;
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);   // アクティブ述語生成
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        svfloat64_t vc = svld1_f64(pg, &c[index]);
        // svmad: a*b + c
        svfloat64_t vr = svmad_f64_x(pg, va, vb, vc);
        svst1_f64(pg, &ret[index], vr);
        index += (int)svcntd();   // ベクトル長だけ進める
    }
}

// ----------------------------------------------------------------
//  _bncsve2_dmul : ret[i] = a[i]*b[i]
// ----------------------------------------------------------------
static void _bncsve2_dmul(
        double ret[], const double a[], const double b[],
        int dim)
{
    svbool_t pg;
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        svst1_f64(pg, &ret[index], svmul_f64_x(pg, va, vb));
        index += (int)svcntd();
    }
}

// ----------------------------------------------------------------
//  _bncsve2_ddiv : ret[i] = a[i]/b[i]
// ----------------------------------------------------------------
static void _bncsve2_ddiv(
        double ret[], const double a[], const double b[],
        int dim)
{
    svbool_t pg;
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        svst1_f64(pg, &ret[index], svdiv_f64_x(pg, va, vb));
        index += (int)svcntd();
    }
}

// ----------------------------------------------------------------
//  _bncsve2_dadd : ret[i] = a[i]+b[i]
// ----------------------------------------------------------------
static void _bncsve2_dadd(
        double ret[], const double a[], const double b[],
        int dim)
{
    svbool_t pg;
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        svst1_f64(pg, &ret[index], svadd_f64_x(pg, va, vb));
        index += (int)svcntd();
    }
}

// ----------------------------------------------------------------
//  _bncsve2_dsub : ret[i] = a[i]-b[i]
// ----------------------------------------------------------------
static void _bncsve2_dsub(
        double ret[], const double a[], const double b[],
        int dim)
{
    svbool_t pg;
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        svst1_f64(pg, &ret[index], svsub_f64_x(pg, va, vb));
        index += (int)svcntd();
    }
}

// ----------------------------------------------------------------
//  _bncsve2_ddotp : Σ a[i]*b[i]  (内積, double精度)
//
//  SVE2では svmla_f64_x(pg, acc, a, b) = acc + a*b
//  を繰り返して総和を svaddv_f64 で水平加算する。
//  Kahan補正は入れていないが、SVE2環境ではFMA連鎖により
//  NEONより丸め誤差が抑制される。
// ----------------------------------------------------------------
static double _bncsve2_ddotp(const double a[], const double b[], int dim)
{
    if (dim <= 0) return 0.0;

    svbool_t pg;
    svfloat64_t acc = svdup_f64(0.0);
    int index = 0;

    while (index < dim)
    {
        pg = svwhilelt_b64_s32(index, dim);
        svfloat64_t va = svld1_f64(pg, &a[index]);
        svfloat64_t vb = svld1_f64(pg, &b[index]);
        // acc[i] += a[i]*b[i]
        acc = svmla_f64_x(pg, acc, va, vb);
        index += (int)svcntd();
    }

    // 全要素の総和 (アクティブ = すべて真)
    svbool_t pg_all = svptrue_b64();
    return svaddv_f64(pg_all, acc);
}

#endif // defined(__ARM_SVE2)
#endif // ifndef __BNCSVE2_D_H
