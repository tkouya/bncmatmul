
// ------------------------
// ----- QD (NEON128) -----
// ------------------------
#ifndef __BNCNEON128_QD_H
#define __BNCNEON128_QD_H

#if !defined(__ARM_NEON) || !defined(__aarch64__)
#  error "This header requires AArch64 with NEON (Armv8)."
#endif

#include <arm_neon.h>

#ifndef QDSIZE
#  define QDSIZE 4
#endif

// Notes:
// - This header is a 128-bit NEON port of the AVX2 QD routines in _bncavx_qd.h.
// - It processes 2 doubles per vector (float64x2_t).
// - It mirrors all _bncavx2_* APIs with _bncneon2_* names, keeping call sites consistent.
// - It assumes the existence of scalar helpers and qdfloat used by the AVX2 header:
//     void   rqd_set(double ret[QDSIZE], const double a[QDSIZE]);
//     void   rqd_add(double ret[QDSIZE], const double a[QDSIZE], const double b[QDSIZE]);
//     void   rqd_abs(double ret[QDSIZE], const double a[QDSIZE]);
//     void   rqd_mul(double ret[QDSIZE], const double a[QDSIZE], const double b[QDSIZE]);
//     int    rqd_cmp(const double a[QDSIZE], const double b[QDSIZE]);
//     void   rqd_sqrt(double ret[QDSIZE], const double a[QDSIZE]);
//     void   renorm(double *a0, double *a1, double *a2, double *a3);
//     void   renorm4(double *a0, double *a1, double *a2, double *a3, double *a4);
//     typedef struct { double val[QDSIZE]; } qdfloat;

// ---------- small helpers ----------
static inline float64x2_t _f64x2_zero() { return vdupq_n_f64(0.0); }
static inline float64x2_t _f64x2_set1(double x){ return vdupq_n_f64(x); }
static inline float64x2_t _f64x2_add(float64x2_t a, float64x2_t b){ return vaddq_f64(a,b); }
static inline float64x2_t _f64x2_sub(float64x2_t a, float64x2_t b){ return vsubq_f64(a,b); }
static inline float64x2_t _f64x2_mul(float64x2_t a, float64x2_t b){ return vmulq_f64(a,b); }
static inline float64x2_t _f64x2_div(float64x2_t a, float64x2_t b){ return vdivq_f64(a,b); }
static inline float64x2_t _f64x2_fma(float64x2_t a, float64x2_t b, float64x2_t c){ return vfmaq_f64(c,a,b); }
static inline float64x2_t _f64x2_neg(float64x2_t a){ return vnegq_f64(a); }

static inline double _lane_get(float64x2_t v, int i){ return vgetq_lane_f64(v, i); }
static inline void _lane_set(float64x2_t *v, int i, double x){ *v = vsetq_lane_f64(x, *v, i); }

// Knuth 2Sum: returns s = a+b, *e = rounding error
static inline float64x2_t _dtwo_sum(float64x2_t a, float64x2_t b, float64x2_t *e){
    float64x2_t s  = _f64x2_add(a,b);
    float64x2_t bb = _f64x2_sub(s,a);
    float64x2_t t1 = _f64x2_sub(a, _f64x2_sub(s,bb));
    float64x2_t t2 = _f64x2_sub(b, bb);
    *e = _f64x2_add(t1,t2);
    return s;
}

// Dekker/Veltkamp 2Prod using FMA: returns p = a*b, *e = fma(a,b,-p)
static inline float64x2_t _dtwo_prod(float64x2_t a, float64x2_t b, float64x2_t *e){
    float64x2_t p = _f64x2_mul(a,b);
    *e = _f64x2_fma(a,b,_f64x2_neg(p));
    return p;
}

// three_sum / three_sum2 (vectorized)
static inline void _three_sum(float64x2_t *a, float64x2_t *b, float64x2_t *c){
    float64x2_t t1,t2,t3;
    t1 = _dtwo_sum(*a, *b, &t2);
    *a = _dtwo_sum(*c, t1, &t3);
    *b = _dtwo_sum(t2, t3,  c);
}
static inline void _three_sum2(float64x2_t *a, float64x2_t *b, float64x2_t *c){
    float64x2_t t1,t2,t3;
    t1 = _dtwo_sum(*a, *b, &t2);
    *a = _dtwo_sum(*c, t1, &t3);
    *b = _f64x2_add(t2, t3);
}

// renorm / renorm4 (apply per-lane via scalar calls)
static inline void _renorm_v(float64x2_t *c0, float64x2_t *c1, float64x2_t *c2, float64x2_t *c3){
    for(int i=0;i<2;i++){
        double a0 = _lane_get(*c0,i), a1=_lane_get(*c1,i), a2=_lane_get(*c2,i), a3=_lane_get(*c3,i);
        renorm(&a0,&a1,&a2,&a3);
        _lane_set(c0,i,a0); _lane_set(c1,i,a1); _lane_set(c2,i,a2); _lane_set(c3,i,a3);
    }
}
static inline void _renorm4_v(float64x2_t *c0, float64x2_t *c1, float64x2_t *c2, float64x2_t *c3, float64x2_t *c4){
    for(int i=0;i<2;i++){
        double a0=_lane_get(*c0,i), a1=_lane_get(*c1,i), a2=_lane_get(*c2,i), a3=_lane_get(*c3,i), a4=_lane_get(*c4,i);
        renorm4(&a0,&a1,&a2,&a3,&a4);
        _lane_set(c0,i,a0); _lane_set(c1,i,a1); _lane_set(c2,i,a2); _lane_set(c3,i,a3); _lane_set(c4,i,a4);
    }
}

// ---------------------- API (mirrors AVX2) ----------------------

// ret := 0
static inline void _bncneon2_set0_qd(float64x2_t ret[QDSIZE]){
    ret[0]=_f64x2_zero(); ret[1]=_f64x2_zero(); ret[2]=_f64x2_zero(); ret[3]=_f64x2_zero();
}

// ret := val (QD)
static inline void _bncneon2_rqd_set(float64x2_t ret[QDSIZE], float64x2_t val[QDSIZE]){
    ret[0]=val[0]; ret[1]=val[1]; ret[2]=val[2]; ret[3]=val[3];
}

// ret := (double)val
static inline void _bncneon2_rqd_set_d(float64x2_t ret[QDSIZE], float64x2_t val){
    ret[0]=val; ret[1]=_f64x2_zero(); ret[2]=_f64x2_zero(); ret[3]=_f64x2_zero();
}

// ret := (DD)val
static inline void _bncneon2_rqd_set_dd(float64x2_t ret[QDSIZE], float64x2_t val[2]){
    ret[0]=val[0]; ret[1]=val[1]; ret[2]=_f64x2_zero(); ret[3]=_f64x2_zero();
}

// ret := (TD)val
static inline void _bncneon2_rqd_set_td(float64x2_t ret[QDSIZE], float64x2_t val[3]){
    ret[0]=val[0]; ret[1]=val[1]; ret[2]=val[2]; ret[3]=_f64x2_zero();
}

// ret := [(QD)val ...] (broadcast per component)
static inline void _bncneon2_rqd_set1_qd(float64x2_t ret[QDSIZE], const double val[QDSIZE]){
    ret[0]=_f64x2_set1(val[0]); ret[1]=_f64x2_set1(val[1]); ret[2]=_f64x2_set1(val[2]); ret[3]=_f64x2_set1(val[3]);
}

// ret := ret4[][neon_index]
static inline void _bncneon2_get_qd_f64x2_i(qdfloat *ret, float64x2_t ret4[QDSIZE], int neon_index){
    ret->val[0] = _lane_get(ret4[0], neon_index);
    ret->val[1] = _lane_get(ret4[1], neon_index);
    ret->val[2] = _lane_get(ret4[2], neon_index);
    ret->val[3] = _lane_get(ret4[3], neon_index);
}

// ret := mmval[0] + ... + mmval[1]  (lane-wise reduce into one QD)
static inline void _bncneon2_rqd_sum128d(double ret[QDSIZE], float64x2_t mmval[QDSIZE]){
    qdfloat r0, r1;
    _bncneon2_get_qd_f64x2_i(&r0, mmval, 0);
    _bncneon2_get_qd_f64x2_i(&r1, mmval, 1);
    rqd_set(ret, r0.val);
    rqd_add(ret, ret, r1.val);
}

// ret := |a| (sign from the leading component per-lane)
static inline void _bncneon2_rqd_abs(float64x2_t ret[QDSIZE], float64x2_t a[QDSIZE]){
    for(int i=0;i<2;i++){
        double s0 = _lane_get(a[0],i);
        for(int k=0;k<4;k++){
            double v = _lane_get(a[k],i);
            _lane_set(&ret[k], i, (s0<0.0)? -v : v);
        }
    }
}

// ret := |mmval[0]| + |mmval[1]|
static inline void _bncneon2_rqd_abssum128d(double ret[QDSIZE], float64x2_t mmval[QDSIZE]){
    qdfloat r; double tmp[QDSIZE];
    _bncneon2_get_qd_f64x2_i(&r, mmval, 0); rqd_abs(tmp, r.val); rqd_set(ret, tmp);
    _bncneon2_get_qd_f64x2_i(&r, mmval, 1); rqd_abs(tmp, r.val); rqd_add(ret, ret, tmp);
}

// ret := max(|mmval[0]|, |mmval[1]|)
static inline void _bncneon2_rqd_absmax128d(double ret[QDSIZE], float64x2_t mmval[QDSIZE]){
    qdfloat r; double tmp[QDSIZE];
    _bncneon2_get_qd_f64x2_i(&r, mmval, 0); rqd_abs(tmp, r.val); rqd_set(ret, tmp);
    _bncneon2_get_qd_f64x2_i(&r, mmval, 1); rqd_abs(tmp, r.val); if(rqd_cmp(ret,tmp)<0) rqd_set(ret,tmp);
}

// ret := sqrt( sum_i mmval[i]^2 )
static inline void _bncneon2_rqd_norm128d(double ret[QDSIZE], float64x2_t mmval[QDSIZE]){
    qdfloat r; double tmp[QDSIZE];
    _bncneon2_get_qd_f64x2_i(&r, mmval, 0); rqd_mul(tmp, r.val, r.val); rqd_set(ret,tmp);
    _bncneon2_get_qd_f64x2_i(&r, mmval, 1); rqd_mul(tmp, r.val, r.val); rqd_add(ret, ret, tmp);
    rqd_sqrt(tmp, ret); rqd_set(ret, tmp);
}

// c := a + b (QD)
static inline void _bncneon2_rqd_add(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE]){
    float64x2_t s0,s1,s2,s3, t0,t1,t2, v0,v1,v2,v3, u0,u1,u2,u3, w0,w1,w2,w3;
    s0=_f64x2_add(a[0],b[0]); s1=_f64x2_add(a[1],b[1]); s2=_f64x2_add(a[2],b[2]); s3=_f64x2_add(a[3],b[3]);
    v0=_f64x2_sub(s0,a[0]);   v1=_f64x2_sub(s1,a[1]);   v2=_f64x2_sub(s2,a[2]);   v3=_f64x2_sub(s3,a[3]);
    u0=_f64x2_sub(s0,v0);     u1=_f64x2_sub(s1,v1);     u2=_f64x2_sub(s2,v2);     u3=_f64x2_sub(s3,v3);
    w0=_f64x2_sub(a[0],u0);   w1=_f64x2_sub(a[1],u1);   w2=_f64x2_sub(a[2],u2);   w3=_f64x2_sub(a[3],u3);
    u0=_f64x2_sub(b[0],v0);   u1=_f64x2_sub(b[1],v1);   u2=_f64x2_sub(b[2],v2);   u3=_f64x2_sub(b[3],v3);
    t0=_f64x2_add(w0,u0);     t1=_f64x2_add(w1,u1);     t2=_f64x2_add(w2,u2);
    s1=_dtwo_sum(s1,t0,&t0);  _three_sum(&s2,&t0,&t1);  _three_sum2(&s3,&t0,&t2);
    t0 = _f64x2_add(_f64x2_add(t0,t1), t2);
    _renorm4_v(&s0,&s1,&s2,&s3,&t0);
    c[0]=s0; c[1]=s1; c[2]=s2; c[3]=s3;
}

// TD: ret := a + b (triple-double)
static inline void _bncneon2_rtd_addq(float64x2_t ret[3], float64x2_t a[3], float64x2_t b[3]){
    float64x2_t s0,s1,s2, t0,t1,t2, v0,v1,v2, u0,u1,u2, w0,w1,w2;
    s0=_f64x2_add(a[0],b[0]); s1=_f64x2_add(a[1],b[1]); s2=_f64x2_add(a[2],b[2]);
    v0=_f64x2_sub(s0,a[0]);   v1=_f64x2_sub(s1,a[1]);   v2=_f64x2_sub(s2,a[2]);
    u0=_f64x2_sub(s0,v0);     u1=_f64x2_sub(s1,v1);     u2=_f64x2_sub(s2,v2);
    w0=_f64x2_sub(a[0],u0);   w1=_f64x2_sub(a[1],u1);   w2=_f64x2_sub(a[2],u2);
    u0=_f64x2_sub(b[0],v0);   u1=_f64x2_sub(b[1],v1);   u2=_f64x2_sub(b[2],v2);
    t0=_f64x2_add(w0,u0);     t1=_f64x2_add(w1,u1);     t2=_f64x2_add(w2,u2);
    s1=_dtwo_sum(s1,t0,&t0);  _three_sum(&s2,&t0,&t1);
    t0 = _f64x2_add(_f64x2_add(t0,t1), t2);
    _renorm_v(&s0,&s1,&s2,&t0);
    ret[0]=s0; ret[1]=s1; ret[2]=s2;
}

// TD: ret := a * b (triple-double)
static inline void _bncneon2_rtd_mulq(float64x2_t ret[3], float64x2_t a[3], float64x2_t b[3]){
    float64x2_t p0,p1,p2,p3,p4,p5, q0,q1,q2,q3,q4,q5, t0,t1, s0,s1,s2;
    p0=_dtwo_prod(a[0],b[0],&q0);
    p1=_dtwo_prod(a[0],b[1],&q1);
    p2=_dtwo_prod(a[1],b[0],&q2);
    p3=_dtwo_prod(a[0],b[2],&q3);
    p4=_dtwo_prod(a[1],b[1],&q4);
    p5=_dtwo_prod(a[2],b[0],&q5);
    _three_sum(&p1,&p2,&q0);
    _three_sum(&p2,&q1,&q2);
    _three_sum(&p3,&p4,&p5);
    s0=_dtwo_sum(p2,p3,&t0);
    s1=_dtwo_sum(q1,p4,&t1);
    s1=_dtwo_sum(s1,t0,&t0);
    s2 = _f64x2_add(q2,p5);
    s2 = _f64x2_add(s2, _f64x2_add(t0,t1));
    _renorm_v(&p0,&p1,&s0,&s1);
    ret[0]=p0; ret[1]=p1; ret[2]=s0;
}

// c := a * b (QD)
static inline void _bncneon2_rqd_mul(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE]){
    float64x2_t p0,p1,p2,p3,p4,p5, q0,q1,q2,q3,q4,q5, t0,t1, s0,s1,s2;
    p0=_dtwo_prod(a[0],b[0],&q0);
    p1=_dtwo_prod(a[0],b[1],&q1);
    p2=_dtwo_prod(a[1],b[0],&q2);
    p3=_dtwo_prod(a[0],b[2],&q3);
    p4=_dtwo_prod(a[1],b[1],&q4);
    p5=_dtwo_prod(a[2],b[0],&q5);
    _three_sum(&p1,&p2,&q0);
    _three_sum(&p2,&q1,&q2);
    _three_sum(&p3,&p4,&p5);
    s0=_dtwo_sum(p2,p3,&t0);
    s1=_dtwo_sum(q1,p4,&t1);
    s2 = _f64x2_add(q2,p5);
    s1=_dtwo_sum(s1,t0,&t0);
    s2 = _f64x2_add(s2, _f64x2_add(t0,t1));
    // O(eps^3) terms
    s1 = _f64x2_add(s1, _f64x2_mul(a[0],b[3]));
    s1 = _f64x2_add(s1, _f64x2_mul(a[1],b[2]));
    s1 = _f64x2_add(s1, _f64x2_mul(a[2],b[1]));
    s1 = _f64x2_add(s1, _f64x2_mul(a[3],b[0]));
    s1 = _f64x2_add(s1, q0);
    s1 = _f64x2_add(s1, q3);
    s1 = _f64x2_add(s1, q4);
    s1 = _f64x2_add(s1, q5);
    _renorm4_v(&p0,&p1,&s0,&s1,&s2);
    c[0]=p0; c[1]=p1; c[2]=s0; c[3]=s1;
}

static inline void _bncneon2_rqd_neg(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE]){
    float64x2_t z=_f64x2_zero();
    c[0]=_f64x2_sub(z,a[0]); c[1]=_f64x2_sub(z,a[1]); c[2]=_f64x2_sub(z,a[2]); c[3]=_f64x2_sub(z,a[3]);
}

static inline void _bncneon2_rqd_sub(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE]){
    float64x2_t mb[QDSIZE]; _bncneon2_rqd_neg(mb,b); _bncneon2_rqd_add(c,a,mb);
}

// c := a + b (QD, b is double)
static inline void _bncneon2_rqd_add_d(float64x2_t c[QDSIZE], const float64x2_t a[QDSIZE], float64x2_t b){
    float64x2_t e0,e1, s0,s1,s2,s3;
    s0=_dtwo_sum(a[0], b,  &e0);
    s1=_dtwo_sum(a[1], e0, &e1);
    s2=_dtwo_sum(a[2], e1, &e0);
    s3=_dtwo_sum(a[3], e0, &e1);
    _renorm4_v(&s0,&s1,&s2,&s3,&e1);
    c[0]=s0; c[1]=s1; c[2]=s2; c[3]=s3;
}

// TD: ret := a + b (b is double)
static inline void _bncneon2_rtd_addq_d(float64x2_t ret[3], const float64x2_t a[3], float64x2_t b){
    float64x2_t e0,e1, s0,s1,s2;
    s0=_dtwo_sum(a[0], b,  &e0);
    s1=_dtwo_sum(a[1], e0, &e1);
    s2=_dtwo_sum(a[2], e1, &e0);
    _renorm_v(&s0,&s1,&s2,&e0);
    ret[0]=s0; ret[1]=s1; ret[2]=s2;
}

// c := a * b (QD, b is double)
static inline void _bncneon2_rqd_mul_d(float64x2_t c[QDSIZE], const float64x2_t a[QDSIZE], float64x2_t b){
    float64x2_t p0,p1,p2,p3, q0,q1,q2, s0,s1,s2,s3,s4;
    p0=_dtwo_prod(a[0],b,&q0);
    p1=_dtwo_prod(a[1],b,&q1);
    p2=_dtwo_prod(a[2],b,&q2);
    p3=_f64x2_mul(a[3],b);
    s0=p0;
    s1=_dtwo_sum(q0,p1,&s2);
    _three_sum(&s2,&q1,&p2);
    _three_sum2(&q1,&q2,&p3);
    s3=q1; s4=_f64x2_add(q2,p2);
    _renorm4_v(&s0,&s1,&s2,&s3,&s4);
    c[0]=s0; c[1]=s1; c[2]=s2; c[3]=s3;
}

// TD: ret := a * b (b is double)
static inline void _bncneon2_rtd_mulq_d(float64x2_t ret[3], const float64x2_t a[3], float64x2_t b){
    float64x2_t p0,p1,p2, q0,q1, s0,s1,s2,s3;
    p0=_dtwo_prod(a[0],b,&q0);
    p1=_dtwo_prod(a[1],b,&q1);
    p2=_f64x2_mul(a[2],b);
    s0=p0;
    s1=_dtwo_sum(q0,p1,&s2);
    _three_sum2(&s2,&q1,&p2);
    s3=_f64x2_add(q1,p2);
    _renorm_v(&s0,&s1,&s2,&s3);
    ret[0]=s0; ret[1]=s1; ret[2]=s2;
}

// TD: ret := a / b (QD / TD) — compute 3-term quotient
static inline void _bncneon2_rtd_divq(float64x2_t ret[3], float64x2_t a[QDSIZE], float64x2_t b[3]){
    float64x2_t q0,q1,q2, r[QDSIZE], tmp[QDSIZE];
    // q0 = a0 / b0
    q0 = _f64x2_div(a[0], b[0]);
    // r = a - q0*b
    _bncneon2_rtd_mulq(tmp, b, (float64x2_t[3]){q0,_f64x2_zero(),_f64x2_zero()});
    _bncneon2_rqd_sub(r, a, tmp);
    // q1 = r0 / b0
    q1 = _f64x2_div(r[0], b[0]);
    _bncneon2_rtd_mulq(tmp, b, (float64x2_t[3]){q1,_f64x2_zero(),_f64x2_zero()});
    _bncneon2_rqd_sub(r, r, tmp);
    // q2 = r0 / b0
    q2 = _f64x2_div(r[0], b[0]);
    _renorm_v(&q0,&q1,&q2,&r[0]);
    ret[0]=q0; ret[1]=q1; ret[2]=q2;
}

// c := a / b (QD / QD) — 4-term quotient
static inline void _bncneon2_rqd_div(float64x2_t c[QDSIZE], float64x2_t a[QDSIZE], float64x2_t b[QDSIZE]){
    float64x2_t q0,q1,q2,q3, r[QDSIZE], tmp[QDSIZE];
    q0 = _f64x2_div(a[0], b[0]);
    _bncneon2_rqd_mul_d(tmp, b, q0); _bncneon2_rqd_sub(r, a, tmp);
    q1 = _f64x2_div(r[0], b[0]);
    _bncneon2_rqd_mul_d(tmp, b, q1); _bncneon2_rqd_sub(r, r, tmp);
    q2 = _f64x2_div(r[0], b[0]);
    _bncneon2_rqd_mul_d(tmp, b, q2); _bncneon2_rqd_sub(r, r, tmp);
    q3 = _f64x2_div(r[0], b[0]);
    _renorm_v(&q0,&q1,&q2,&q3);
    c[0]=q0; c[1]=q1; c[2]=q2; c[3]=q3;
}

#endif // __BNCNEON128_QD_H
