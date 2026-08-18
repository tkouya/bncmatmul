// ------------------------
// ------ QD AVX-512 ------
// ------------------------
#ifndef __BNCAVX_QD_AVX512_H
#define __BNCAVX_QD_AVX512_H

#ifndef QDSIZE
    #define QDSIZE 4
#endif // QDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_set0_qd(__m512d ret[QDSIZE])
{
    ret[0] = _mm512_setzero_pd();
    ret[1] = _mm512_setzero_pd();
    ret[2] = _mm512_setzero_pd();
    ret[3] = _mm512_setzero_pd();
}
#define _bncavx512_rqd_set0(ret) _bncavx512_set0_qd((ret))

// ret := val
static inline void _bncavx512_rqd_set(__m512d ret[QDSIZE], __m512d val[QDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = val[3];
}

// ret := (double)val
static inline void _bncavx512_rqd_set_d(__m512d ret[QDSIZE], __m512d val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_pd(); // val[1];
    ret[2] = _mm512_setzero_pd(); // val[2];
    ret[3] = _mm512_setzero_pd(); // val[3];
}

// ret := (DD)val
static inline void _bncavx512_rqd_set_dd(__m512d ret[QDSIZE], __m512d val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = _mm512_setzero_pd(); // val[2];
    ret[3] = _mm512_setzero_pd(); // val[3];
}

// ret := (TD)val
static inline void _bncavx512_rqd_set_td(__m512d ret[QDSIZE], __m512d val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = _mm512_setzero_pd(); // val[3];
}

// ret := [val val val val]
static inline void _bncavx512_rqd_set1_qd(__m512d ret[QDSIZE], double val[QDSIZE])
{
    ret[0] = _mm512_set1_pd(val[0]);
    ret[1] = _mm512_set1_pd(val[1]);
    ret[2] = _mm512_set1_pd(val[2]);
    ret[3] = _mm512_set1_pd(val[3]);
}

// ret := ret8[][avx_index]
static inline void _bncavx512_get_qd_m512d_i(qdfloat *ret, __m512d ret8[QDSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];
    ret->val[2] = ret8[2][avx_index];
    ret->val[3] = ret8[3][avx_index];

    return;
}

// ret := mmval[0] + ... + mmval[7]
static void _bncavx512_rqd_sum512d(double ret[QDSIZE], __m512d ret8[QDSIZE])
{
    qdfloat ret8_i[8];

    // ret8_i := ret8
    _bncavx512_get_qd_m512d_i(&(ret8_i[0]), ret8, 0);
    _bncavx512_get_qd_m512d_i(&(ret8_i[1]), ret8, 1);
    _bncavx512_get_qd_m512d_i(&(ret8_i[2]), ret8, 2);
    _bncavx512_get_qd_m512d_i(&(ret8_i[3]), ret8, 3);
    _bncavx512_get_qd_m512d_i(&(ret8_i[4]), ret8, 4);
    _bncavx512_get_qd_m512d_i(&(ret8_i[5]), ret8, 5);
    _bncavx512_get_qd_m512d_i(&(ret8_i[6]), ret8, 6);
    _bncavx512_get_qd_m512d_i(&(ret8_i[7]), ret8, 7);

    rqd_set(ret, ret8_i[0].val);
    rqd_add(ret, ret, ret8_i[1].val);
    rqd_add(ret, ret, ret8_i[2].val);
    rqd_add(ret, ret, ret8_i[3].val);
    rqd_add(ret, ret, ret8_i[4].val);
    rqd_add(ret, ret, ret8_i[5].val);
    rqd_add(ret, ret, ret8_i[6].val);
    rqd_add(ret, ret, ret8_i[7].val);
}

// abs
static inline void _bncavx512_rqd_abs(__m512d ret[QDSIZE], __m512d a[QDSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 8; avx_index++)
    {
        if(a[0][avx_index] < 0.0)
        {
            ret[0][avx_index] = -a[0][avx_index];
            ret[1][avx_index] = -a[1][avx_index];
            ret[2][avx_index] = -a[2][avx_index];
            ret[3][avx_index] = -a[3][avx_index];
        }
        else
        {
            ret[0][avx_index] = a[0][avx_index];
            ret[1][avx_index] = a[1][avx_index];
            ret[2][avx_index] = a[2][avx_index];
            ret[3][avx_index] = a[3][avx_index];
        }
    }
}
// ret := |ret8[0]| + |ret8[1]| + |ret8[2]| + |ret8[3]|
static void _bncavx512_rqd_abssum512d(double ret[QDSIZE], __m512d ret8[QDSIZE])
{
    qdfloat ret8_i[8];
    double tmp[QDSIZE];

    // ret8_i := ret8
    _bncavx512_get_qd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_qd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_qd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_qd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_qd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_qd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_qd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_qd_m512d_i(&ret8_i[7], ret8, 7);

    rqd_abs(tmp, ret8_i[0].val);
    rqd_set(ret, tmp);

    rqd_abs(tmp, ret8_i[1].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[2].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[3].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[4].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[5].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[6].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret8_i[7].val);
    rqd_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, |ret8[2]|, |ret8[3]|)
static void _bncavx512_rqd_absmax512d(double ret[QDSIZE], __m512d ret8[QDSIZE])
{
    qdfloat ret8_i[8];
    double tmp[QDSIZE];

    // ret8_i := ret8
    _bncavx512_get_qd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_qd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_qd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_qd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_qd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_qd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_qd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_qd_m512d_i(&ret8_i[7], ret8, 7);

    rqd_abs(tmp, ret8_i[0].val);
    rqd_set(ret, tmp); // ret := |ret8_i[0]|

    rqd_abs(tmp, ret8_i[1].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[1]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[1]|

    rqd_abs(tmp, ret8_i[2].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[2]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[2]|

    rqd_abs(tmp, ret8_i[3].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[3]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[3]|

    rqd_abs(tmp, ret8_i[4].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[4]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[4]|

    rqd_abs(tmp, ret8_i[5].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[5]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[5]|

    rqd_abs(tmp, ret8_i[6].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[6]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[6]|

    rqd_abs(tmp, ret8_i[7].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[7]|)
        rqd_set(ret, tmp);    //   ret := |ret8_i[7]|

}

// ret := || ret8[0]^2 + ret8[1]^2 + ret8[2]^2 + ret8[3]^2 ||_2
static void _bncavx512_rqd_norm512d(double ret[QDSIZE], __m512d ret8[QDSIZE])
{
    qdfloat ret8_i[8];
    double tmp[QDSIZE];

    // ret8_i := ret8
    _bncavx512_get_qd_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_qd_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_qd_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_qd_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_qd_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_qd_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_qd_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_qd_m512d_i(&ret8_i[7], ret8, 7);

    rqd_mul(tmp, ret8_i[0].val, ret8_i[0].val);
    rqd_set(ret, tmp);

    rqd_mul(tmp, ret8_i[1].val, ret8_i[1].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret8_i[2].val, ret8_i[2].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret8_i[3].val, ret8_i[3].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret8_i[4].val, ret8_i[4].val);
    rqd_set(ret, tmp);

    rqd_mul(tmp, ret8_i[5].val, ret8_i[5].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret8_i[6].val, ret8_i[6].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret8_i[7].val, ret8_i[7].val);
    rqd_add(ret, ret, tmp);

    rqd_sqrt(tmp, ret);
    rqd_set(ret, tmp);
}

//inline void renorm(double *c0, double *c1, double *c2, double *c3)
static inline void _bncavx512_renorm(__m512d *c0, __m512d *c1, __m512d *c2, __m512d *c3)
{
// New codes on 2020-11-10 by T.Kouya
    double q0[4], q1[4], q2[4], q3[4], q4[4], q5[4], q6[4], q7[4];

    q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0];
    q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1];
    q2[0] = (* c0)[2]; q2[1] = (* c1)[2]; q2[2] = (* c2)[2]; q2[3] = (* c3)[2];
    q3[0] = (* c0)[3]; q3[1] = (* c1)[3]; q3[2] = (* c2)[3]; q3[3] = (* c3)[3];
    q4[0] = (* c0)[4]; q4[1] = (* c1)[4]; q4[2] = (* c2)[4]; q4[3] = (* c3)[4];
    q5[0] = (* c0)[5]; q5[1] = (* c1)[5]; q5[2] = (* c2)[5]; q5[3] = (* c3)[5];
    q6[0] = (* c0)[6]; q6[1] = (* c1)[6]; q6[2] = (* c2)[6]; q6[3] = (* c3)[6];
    q7[0] = (* c0)[7]; q7[1] = (* c1)[7]; q7[2] = (* c2)[7]; q7[3] = (* c3)[7];

    renorm(&q0[0], &q0[1], &q0[2], &q0[3]);
    renorm(&q1[0], &q1[1], &q1[2], &q1[3]);
    renorm(&q2[0], &q2[1], &q2[2], &q2[3]);
    renorm(&q3[0], &q3[1], &q3[2], &q3[3]);
    renorm(&q4[0], &q4[1], &q4[2], &q4[3]);
    renorm(&q5[0], &q5[1], &q5[2], &q5[3]);
    renorm(&q6[0], &q6[1], &q6[2], &q6[3]);
    renorm(&q7[0], &q7[1], &q7[2], &q7[3]);

    (* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3];
    (* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3];
    (* c0)[2] = q2[0]; (* c1)[2] = q2[1]; (* c2)[2] = q2[2]; (* c3)[2] = q2[3];
    (* c0)[3] = q3[0]; (* c1)[3] = q3[1]; (* c2)[3] = q3[2]; (* c3)[3] = q3[3];
    (* c0)[4] = q4[0]; (* c1)[4] = q4[1]; (* c2)[4] = q4[2]; (* c3)[4] = q4[3];
    (* c0)[5] = q5[0]; (* c1)[5] = q5[1]; (* c2)[5] = q5[2]; (* c3)[5] = q5[3];
    (* c0)[6] = q6[0]; (* c1)[6] = q6[1]; (* c2)[6] = q6[2]; (* c3)[6] = q6[3];
    (* c0)[7] = q7[0]; (* c1)[7] = q7[1]; (* c2)[7] = q7[2]; (* c3)[7] = q7[3];
}

//inline void renorm4(double *c0, double *c1, double *c2, double *c3, double *c4)
static inline void _bncavx512_renorm4(__m512d *c0, __m512d *c1, __m512d *c2, __m512d *c3, __m512d *c4)
{
    int avx_index;
	double s0, s1, s2 = 0.0, s3 = 0.0;

// New codes on 2020-11-10 by T.Kouya
    double q0[5], q1[5], q2[5], q3[5], q4[5], q5[5], q6[5], q7[5];

    q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0]; q0[4] = (* c4)[0];
    q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1]; q1[4] = (* c4)[1];
    q2[0] = (* c0)[2]; q2[1] = (* c1)[2]; q2[2] = (* c2)[2]; q2[3] = (* c3)[2]; q2[4] = (* c4)[2];
    q3[0] = (* c0)[3]; q3[1] = (* c1)[3]; q3[2] = (* c2)[3]; q3[3] = (* c3)[3]; q3[4] = (* c4)[3];
    q4[0] = (* c0)[4]; q4[1] = (* c1)[4]; q4[2] = (* c2)[4]; q4[3] = (* c3)[4]; q4[4] = (* c4)[4];
    q5[0] = (* c0)[5]; q5[1] = (* c1)[5]; q5[2] = (* c2)[5]; q5[3] = (* c3)[5]; q5[4] = (* c4)[5];
    q6[0] = (* c0)[6]; q6[1] = (* c1)[6]; q6[2] = (* c2)[6]; q6[3] = (* c3)[6]; q6[4] = (* c4)[6];
    q7[0] = (* c0)[7]; q7[1] = (* c1)[7]; q7[2] = (* c2)[7]; q7[3] = (* c3)[7]; q7[4] = (* c4)[7];

    renorm4(&q0[0], &q0[1], &q0[2], &q0[3], &q0[4]);
    renorm4(&q1[0], &q1[1], &q1[2], &q1[3], &q1[4]);
    renorm4(&q2[0], &q2[1], &q2[2], &q2[3], &q2[4]);
    renorm4(&q3[0], &q3[1], &q3[2], &q3[3], &q3[4]);
    renorm4(&q4[0], &q4[1], &q4[2], &q4[3], &q4[4]);
    renorm4(&q5[0], &q5[1], &q5[2], &q5[3], &q5[4]);
	renorm4(&q6[0], &q6[1], &q6[2], &q6[3], &q6[4]);
	renorm4(&q7[0], &q7[1], &q7[2], &q7[3], &q7[4]);

    (* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3]; (* c4)[0] = q0[4];
    (* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3]; (* c4)[1] = q1[4];
    (* c0)[2] = q2[0]; (* c1)[2] = q2[1]; (* c2)[2] = q2[2]; (* c3)[2] = q2[3]; (* c4)[2] = q2[4];
    (* c0)[3] = q3[0]; (* c1)[3] = q3[1]; (* c2)[3] = q3[2]; (* c3)[3] = q3[3]; (* c4)[3] = q3[4];
    (* c0)[4] = q4[0]; (* c1)[4] = q4[1]; (* c2)[4] = q4[2]; (* c3)[4] = q4[3]; (* c4)[4] = q4[4];
    (* c0)[5] = q5[0]; (* c1)[5] = q5[1]; (* c2)[5] = q5[2]; (* c3)[5] = q5[3]; (* c4)[5] = q5[4];
    (* c0)[6] = q6[0]; (* c1)[6] = q6[1]; (* c2)[6] = q6[2]; (* c3)[6] = q6[3]; (* c4)[6] = q6[4];
    (* c0)[7] = q7[0]; (* c1)[7] = q7[1]; (* c2)[7] = q7[2]; (* c3)[7] = q7[3]; (* c4)[7] = q7[4];
}


/********** Additions ************/
//inline void three_sum(double *a, double *b, double *c)
static inline void _bncavx512_three_sum(__m512d *a, __m512d *b, __m512d *c)
{
//	double t1, t2, t3;
    __m512d t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = two_sum(t2, t3, c);
	t1 = _bncavx512_dtwo_sum(*a, *b, &t2);
	*a = _bncavx512_dtwo_sum(*c, t1, &t3);
	*b = _bncavx512_dtwo_sum(t2, t3, c);

}

//inline void three_sum2(double *a, double *b, double *c)
static inline void _bncavx512_three_sum2(__m512d *a, __m512d *b, __m512d *c)
{
//	double t1, t2, t3;
    __m512d t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = t2 + t3;
	t1 = _bncavx512_dtwo_sum(*a, *b, &t2);
	*a = _bncavx512_dtwo_sum(*c, t1, &t3);
	*b = _mm512_add_pd(t2, t3);

}

#ifdef USE_QD_BF
    #define _bncavx512_rqd_add _bncavx512_rqd_add_bf
#else // USE_QD_BF
    #define _bncavx512_rqd_add _bncavx512_rqd_add_sloppy
#endif // USE_QD_BF

//static inline void _bncavx512_rqd_add(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
static inline void _bncavx512_rqd_add_sloppy(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
//#if 0
    double in_ret[8][QDSIZE], in_a[8][QDSIZE], in_b[8][QDSIZE];

    //in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
    //in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
    //in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
    //in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];
    in_a[4][0] = a[0][4]; in_a[4][1] = a[1][4]; in_a[4][2] = a[2][4]; in_a[4][3] = a[3][4];
    in_a[5][0] = a[0][5]; in_a[5][1] = a[1][5]; in_a[5][2] = a[2][5]; in_a[5][3] = a[3][5];
    in_a[6][0] = a[0][6]; in_a[6][1] = a[1][6]; in_a[6][2] = a[2][6]; in_a[6][3] = a[3][6];
    in_a[7][0] = a[0][7]; in_a[7][1] = a[1][7]; in_a[7][2] = a[2][7]; in_a[7][3] = a[3][7];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];
    in_b[4][0] = b[0][4]; in_b[4][1] = b[1][4]; in_b[4][2] = b[2][4]; in_b[4][3] = b[3][4];
    in_b[5][0] = b[0][5]; in_b[5][1] = b[1][5]; in_b[5][2] = b[2][5]; in_b[5][3] = b[3][5];
    in_b[6][0] = b[0][6]; in_b[6][1] = b[1][6]; in_b[6][2] = b[2][6]; in_b[6][3] = b[3][6];
    in_b[7][0] = b[0][7]; in_b[7][1] = b[1][7]; in_b[7][2] = b[2][7]; in_b[7][3] = b[3][7];

    rqd_add(in_ret[0], in_a[0], in_b[0]);
    rqd_add(in_ret[1], in_a[1], in_b[1]);
    rqd_add(in_ret[2], in_a[2], in_b[2]);
    rqd_add(in_ret[3], in_a[3], in_b[3]);
    rqd_add(in_ret[4], in_a[4], in_b[4]);
    rqd_add(in_ret[5], in_a[5], in_b[5]);
    rqd_add(in_ret[6], in_a[6], in_b[6]);
    rqd_add(in_ret[7], in_a[7], in_b[7]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
    ret[0][4] = in_ret[4][0]; ret[1][4] = in_ret[4][1]; ret[2][4] = in_ret[4][2]; ret[3][4] = in_ret[4][3];
    ret[0][5] = in_ret[5][0]; ret[1][5] = in_ret[5][1]; ret[2][5] = in_ret[5][2]; ret[3][5] = in_ret[5][3];
    ret[0][6] = in_ret[6][0]; ret[1][6] = in_ret[6][1]; ret[2][6] = in_ret[6][2]; ret[3][6] = in_ret[6][3];
    ret[0][7] = in_ret[7][0]; ret[1][7] = in_ret[7][1]; ret[2][7] = in_ret[7][2]; ret[3][7] = in_ret[7][3];

//#endif // 0
#if 0
    // c_qd_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	double s0, s1, s2, s3;
//	double t0, t1, t2, t3;
	__m512d s0, s1, s2, s3;
	__m512d t0, t1, t2, t3;

//	double v0, v1, v2, v3;
//	double u0, u1, u2, u3;
//	double w0, w1, w2, w3;
	__m512d v0, v1, v2, v3;
	__m512d u0, u1, u2, u3;
	__m512d w0, w1, w2, w3;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm512_add_pd(a[0], b[0]);
	s1 = _mm512_add_pd(a[1], b[1]);
	s2 = _mm512_add_pd(a[2], b[2]);
	s3 = _mm512_add_pd(a[3], b[3]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm512_sub_pd(s0, a[0]);
	v1 = _mm512_sub_pd(s1, a[1]);
	v2 = _mm512_sub_pd(s2, a[2]);
	v3 = _mm512_sub_pd(s3, a[3]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm512_sub_pd(s0, v0);
	u1 = _mm512_sub_pd(s1, v1);
	u2 = _mm512_sub_pd(s2, v2);
	u3 = _mm512_sub_pd(s3, v3);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm512_sub_pd(a[0], u0);
	w1 = _mm512_sub_pd(a[1], u1);
	w2 = _mm512_sub_pd(a[2], u2);
	w3 = _mm512_sub_pd(a[3], u3);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm512_sub_pd(b[0], v0);
	u1 = _mm512_sub_pd(b[1], v1);
	u2 = _mm512_sub_pd(b[2], v2);
	u3 = _mm512_sub_pd(b[3], v3);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm512_add_pd(w0, u0);
	t1 = _mm512_add_pd(w1, u1);
	t2 = _mm512_add_pd(w2, u2);
	t3 = _mm512_add_pd(w3, u3);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx512_dtwo_sum(s1, t0, &t0);
	_bncavx512_three_sum(&s2, &t0, &t1);
	_bncavx512_three_sum2(&s3, &t0, &t2);
	t0 = _mm512_add_pd(_mm512_add_pd(t0, t1), t3);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx512_renorm4(&s0, &s1, &s2, &s3, &t0);
//	return qd_real(s0, s1, s2, s3);
/*
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
*/
	ret[0] = s0;
	ret[1] = s1;
	ret[2] = s2;
	ret[3] = s3;
#endif // 0
}

// 2025-12-25(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Add4(const double x[4], const double y[4], double z[4]) {
static inline void _bncavx512_rqd_add_bf(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
	__m512d a0 , b0 , c0 , d0 , e0, f0, g0, h0;
	__m512d a1 , b1 , c1 , d1 , e1, f1, g1, h1;
	__m512d a2 , b2 , c2 , d2 , e2, f2, g2;
	__m512d a3 , b3 , c3 , d3 , e3, f3, g3;
	__m512d a4 , b4 , c4 , d4 , e4;
	__m512d a5 , b5 , c5 , d5 , e5;
	__m512d a6 , b6 , c6 , d6 , e6;
	__m512d a7 , b7 , c7 , d7 , e7;
	__m512d a8 , b8 , c8 , d8 , e8;
	__m512d a9 , b9 , c9 , d9 ;
	__m512d a10, b10, c10, d10;
	__m512d a11, b11, c11, d11;
	__m512d a12, b12, c12, d12;

	a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];
    a1 = _bncavx512_dtwo_sum(a0, b0, &b1);
    c1 = _bncavx512_dtwo_sum(c0, d0, &d1);
    e1 = _bncavx512_dtwo_sum(e0, f0, &f1);
    g1 = _bncavx512_dtwo_sum(g0, h0, &h1);
    a2 = _bncavx512_dquick_two_sum(a1, c1, &c2);
    b2 = _mm512_add_pd(b1, h1);
    d2 = _bncavx512_dtwo_sum(d1, e1, &e2);
    f2 = _bncavx512_dtwo_sum(f1, g1, &g2);
    b3 = _bncavx512_dtwo_sum(b2, g2, &g3);
    c3 = _bncavx512_dquick_two_sum(c2, d2, &d3);
    e3 = _bncavx512_dtwo_sum(e2, f2, &f3);
    a4 = _bncavx512_dquick_two_sum(a2, c3, &c4);
    d4 = _bncavx512_dquick_two_sum(d3, e3, &e4);
    b5 = _bncavx512_dtwo_sum(b3, d4, &d5);
    e5 = _mm512_add_pd(e4, f3);
    b6 = _bncavx512_dtwo_sum(b5, c4, &c6);
    d6 = _bncavx512_dtwo_sum(d5, e5, &e6);
    a7 = _bncavx512_dquick_two_sum(a4, b6, &b7);
    c7 = _bncavx512_dquick_two_sum(c6, d6, &d7);
    e8 = _mm512_add_pd(e6, g3);
    b8 = _bncavx512_dquick_two_sum(b7, c7, &c8);
    d9 = _mm512_add_pd(d7, e8);
    a10 = _bncavx512_dquick_two_sum(a7, b8, &b10);
    c10 = _bncavx512_dquick_two_sum(c8, d9, &d10);
    b11 = _bncavx512_dquick_two_sum(b10, c10, &c11);
    c12 = _bncavx512_dquick_two_sum(c11, d10, &d12);
    //return MultiFloat<T, 4>{a10, b11, c12, d12};
	ret[0] = a10;
	ret[1] = b11;
	ret[2] = c12;
	ret[3] = d12;
}


// rqd_add -> rtd_add
static inline void _bncavx512_rtd_addq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//#ifdef BNC_USE_ICC
#if 0
    double in_ret[8][QDSIZE], in_a[8][QDSIZE], in_b[8][QDSIZE];

    //in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
    //in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
    //in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
    //in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];
    in_a[4][0] = a[0][4]; in_a[4][1] = a[1][4]; in_a[4][2] = a[2][4]; in_a[4][3] = a[3][4];
    in_a[5][0] = a[0][5]; in_a[5][1] = a[1][5]; in_a[5][2] = a[2][5]; in_a[5][3] = a[3][5];
    in_a[6][0] = a[0][6]; in_a[6][1] = a[1][6]; in_a[6][2] = a[2][6]; in_a[6][3] = a[3][6];
    in_a[7][0] = a[0][7]; in_a[7][1] = a[1][7]; in_a[7][2] = a[2][7]; in_a[7][3] = a[3][7];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];
    in_b[4][0] = b[0][4]; in_b[4][1] = b[1][4]; in_b[4][2] = b[2][4]; in_b[4][3] = b[3][4];
    in_b[5][0] = b[0][5]; in_b[5][1] = b[1][5]; in_b[5][2] = b[2][5]; in_b[5][3] = b[3][5];
    in_b[6][0] = b[0][6]; in_b[6][1] = b[1][6]; in_b[6][2] = b[2][6]; in_b[6][3] = b[3][6];
    in_b[7][0] = b[0][7]; in_b[7][1] = b[1][7]; in_b[7][2] = b[2][7]; in_b[7][3] = b[3][7];

    rtd_add(in_ret[0], in_a[0], in_b[0]);
    rtd_add(in_ret[1], in_a[1], in_b[1]);
    rtd_add(in_ret[2], in_a[2], in_b[2]);
    rtd_add(in_ret[3], in_a[3], in_b[3]);
    rtd_add(in_ret[4], in_a[4], in_b[4]);
    rtd_add(in_ret[5], in_a[5], in_b[5]);
    rtd_add(in_ret[6], in_a[6], in_b[6]);
    rtd_add(in_ret[7], in_a[7], in_b[7]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
    ret[0][4] = in_ret[4][0]; ret[1][4] = in_ret[4][1]; ret[2][4] = in_ret[4][2]; ret[3][4] = in_ret[4][3];
    ret[0][5] = in_ret[5][0]; ret[1][5] = in_ret[5][1]; ret[2][5] = in_ret[5][2]; ret[3][5] = in_ret[5][3];
    ret[0][6] = in_ret[6][0]; ret[1][6] = in_ret[6][1]; ret[2][6] = in_ret[6][2]; ret[3][6] = in_ret[6][3];
    ret[0][7] = in_ret[7][0]; ret[1][7] = in_ret[7][1]; ret[2][7] = in_ret[7][2]; ret[3][7] = in_ret[7][3];

#endif // 0

//#if 0
    // c_qd_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	double s0, s1, s2, s3;
//	double t0, t1, t2, t3;
	__m512d s0, s1, s2;
	__m512d t0, t1, t2;

//	double v0, v1, v2;
//	double u0, u1, u2;
//	double w0, w1, w2;
	__m512d v0, v1, v2;
	__m512d u0, u1, u2;
	__m512d w0, w1, w2;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm512_add_pd(a[0], b[0]);
	s1 = _mm512_add_pd(a[1], b[1]);
	s2 = _mm512_add_pd(a[2], b[2]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm512_sub_pd(s0, a[0]);
	v1 = _mm512_sub_pd(s1, a[1]);
	v2 = _mm512_sub_pd(s2, a[2]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm512_sub_pd(s0, v0);
	u1 = _mm512_sub_pd(s1, v1);
	u2 = _mm512_sub_pd(s2, v2);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm512_sub_pd(a[0], u0);
	w1 = _mm512_sub_pd(a[1], u1);
	w2 = _mm512_sub_pd(a[2], u2);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm512_sub_pd(b[0], v0);
	u1 = _mm512_sub_pd(b[1], v1);
	u2 = _mm512_sub_pd(b[2], v2);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm512_add_pd(w0, u0);
	t1 = _mm512_add_pd(w1, u1);
	t2 = _mm512_add_pd(w2, u2);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx512_dtwo_sum(s1, t0, &t0);
	_bncavx512_three_sum(&s2, &t0, &t1);
//	_bncavx512_three_sum2(&s2, &t0, &t1);
//	t0 = _mm512_add_pd(_mm512_add_pd(t0, t1), t2);
	t0 = _mm512_add_pd(t0, t1);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx512_renorm(&s0, &s1, &s2, &t0);
//	return qd_real(s0, s1, s2, s3);
/*
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
*/
	ret[0] = s0;
	ret[1] = s1;
	ret[2] = s2;
//#endif // 0
}

// mul
static inline void _bncavx512_rtd_mulq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//#if 0
    // c_qd_mul_sloppy
/*
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;
*/
    __m512d p0, p1, p2, p3, p4, p5;
    __m512d q0, q1, q2, q3, q4, q5;
    __m512d t0, t1;
    __m512d s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx512_dtwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx512_dtwo_prod(a[0], b[1], &q1);
	p2 = _bncavx512_dtwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx512_dtwo_prod(a[0], b[2], &q3);
	p4 = _bncavx512_dtwo_prod(a[1], b[1], &q4);
	p5 = _bncavx512_dtwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx512_three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx512_three_sum2(&p2, &q1, &q2);
	_bncavx512_three_sum2(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx512_dtwo_sum(p2, p3, &t0);
	s1 = _bncavx512_dtwo_sum(q1, p4, &t1);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx512_dtwo_sum(s1, t0, &t0);

	/* O(eps^3) order terms */
	//s1 += a[1]*b[2] + a[2]*b[1] + q0 + q3 + q4;
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[1], b[2]));
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[2], b[1]));
    s1 = _mm512_add_pd(s1, q0);
    s1 = _mm512_add_pd(s1, q3);
    s1 = _mm512_add_pd(s1, q4);

	//renorm(p0, p1, s0, s1, s2);
//	renorm(&p0, &p1, &s0, &s1);
	_bncavx512_renorm(&p0, &p1, &s0, &s1);
//	return qd_real(p0, p1, s0, s1);
/*
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	c[3] = s1;
*/
	ret[0] = p0;
	ret[1] = p1;
	ret[2] = s0;

//#endif // 0
}

#ifdef USE_QD_BF
    #define _bncavx512_rqd_mul _bncavx512_rqd_mul_bf
#else // USE_QD_BF
    #define _bncavx512_rqd_mul _bncavx512_rqd_mul_sloppy
#endif // USE_QD_BF

// mul
//static inline void _bncavx512_rqd_mul(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
static inline void _bncavx512_rqd_mul_sloppy(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
//#if 0
    double in_ret[8][QDSIZE], in_a[8][QDSIZE], in_b[8][QDSIZE];

    //in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
    //in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
    //in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
    //in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];
    in_a[4][0] = a[0][4]; in_a[4][1] = a[1][4]; in_a[4][2] = a[2][4]; in_a[4][3] = a[3][4];
    in_a[5][0] = a[0][5]; in_a[5][1] = a[1][5]; in_a[5][2] = a[2][5]; in_a[5][3] = a[3][5];
    in_a[6][0] = a[0][6]; in_a[6][1] = a[1][6]; in_a[6][2] = a[2][6]; in_a[6][3] = a[3][6];
    in_a[7][0] = a[0][7]; in_a[7][1] = a[1][7]; in_a[7][2] = a[2][7]; in_a[7][3] = a[3][7];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];
    in_b[4][0] = b[0][4]; in_b[4][1] = b[1][4]; in_b[4][2] = b[2][4]; in_b[4][3] = b[3][4];
    in_b[5][0] = b[0][5]; in_b[5][1] = b[1][5]; in_b[5][2] = b[2][5]; in_b[5][3] = b[3][5];
    in_b[6][0] = b[0][6]; in_b[6][1] = b[1][6]; in_b[6][2] = b[2][6]; in_b[6][3] = b[3][6];
    in_b[7][0] = b[0][7]; in_b[7][1] = b[1][7]; in_b[7][2] = b[2][7]; in_b[7][3] = b[3][7];

    rqd_mul(in_ret[0], in_a[0], in_b[0]);
    rqd_mul(in_ret[1], in_a[1], in_b[1]);
    rqd_mul(in_ret[2], in_a[2], in_b[2]);
    rqd_mul(in_ret[3], in_a[3], in_b[3]);
    rqd_mul(in_ret[4], in_a[4], in_b[4]);
    rqd_mul(in_ret[5], in_a[5], in_b[5]);
    rqd_mul(in_ret[6], in_a[6], in_b[6]);
    rqd_mul(in_ret[7], in_a[7], in_b[7]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
    ret[0][4] = in_ret[4][0]; ret[1][4] = in_ret[4][1]; ret[2][4] = in_ret[4][2]; ret[3][4] = in_ret[4][3];
    ret[0][5] = in_ret[5][0]; ret[1][5] = in_ret[5][1]; ret[2][5] = in_ret[5][2]; ret[3][5] = in_ret[5][3];
    ret[0][6] = in_ret[6][0]; ret[1][6] = in_ret[6][1]; ret[2][6] = in_ret[6][2]; ret[3][6] = in_ret[6][3];
    ret[0][7] = in_ret[7][0]; ret[1][7] = in_ret[7][1]; ret[2][7] = in_ret[7][2]; ret[3][7] = in_ret[7][3];

//#endif // 0
#if 0
    // c_qd_mul_sloppy
/*
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;
*/
    __m512d p0, p1, p2, p3, p4, p5;
    __m512d q0, q1, q2, q3, q4, q5;
    __m512d t0, t1;
    __m512d s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx512_dtwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx512_dtwo_prod(a[0], b[1], &q1);
	p2 = _bncavx512_dtwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx512_dtwo_prod(a[0], b[2], &q3);
	p4 = _bncavx512_dtwo_prod(a[1], b[1], &q4);
	p5 = _bncavx512_dtwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx512_three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx512_three_sum(&p2, &q1, &q2);
	_bncavx512_three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx512_dtwo_sum(p2, p3, &t0);
	s1 = _bncavx512_dtwo_sum(q1, p4, &t1);
//	s2 = q2 + p5;
	s2 = _mm512_add_pd(q2, p5);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx512_dtwo_sum(s1, t0, &t0);
//	s2 += (t0 + t1);
	s2 = _mm512_add_pd(s2, _mm512_add_pd(t0, t1));

	/* O(eps^3) order terms */
	//s1 += a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5;
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[0], b[3]));
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[1], b[2]));
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[2], b[1]));
    s1 = _mm512_add_pd(s1, _mm512_mul_pd(a[3], b[0]));
    s1 = _mm512_add_pd(s1, q0);
    s1 = _mm512_add_pd(s1, q3);
    s1 = _mm512_add_pd(s1, q4);
    s1 = _mm512_add_pd(s1, q5);

	//renorm(p0, p1, s0, s1, s2);
//	renorm4(&p0, &p1, &s0, &s1, &s2);
	_bncavx512_renorm4(&p0, &p1, &s0, &s1, &s2);
//	return qd_real(p0, p1, s0, s1);
/*
	c[0] = p0;
	c[1] = p1;
	c[2] = s0;
	c[3] = s1;
*/
	ret[0] = p0;
	ret[1] = p1;
	ret[2] = s0;
	ret[3] = s1;

#endif // 0

}

//#endif //defined(__AVX2__)
// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Mul4(const double x[4], const double y[4], double z[4]) {
static inline void _bncavx512_rqd_mul_bf(__m512d ret[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
	__m512d a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
	__m512d a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1;
	__m512d a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2;
	__m512d a3, b3, c3, d3, e3, f3, g3, h3;
	__m512d a4, b4, c4, d4, e4, f4;
	__m512d a5, b5, c5, d5;
	__m512d a6, b6, c6, d6;
	__m512d a7, b7, c7, d7;
	__m512d a8, b8, c8, d8;
	__m512d a9, b9, c9, d9;
	__m512d a10, b10, c10, d10;

    a0 = _bncavx512_dtwo_prod(a[0], b[0], &b0);
    c0 = _bncavx512_dtwo_prod(a[0], b[1], &e0);
    d0 = _bncavx512_dtwo_prod(a[1], b[0], &f0);
    g0 = _bncavx512_dtwo_prod(a[0], b[2], &j0);
    h0 = _bncavx512_dtwo_prod(a[1], b[1], &k0);
    i0 = _bncavx512_dtwo_prod(a[2], b[0], &l0);
    m0 = _mm512_mul_pd(a[0], b[3]);
    n0 = _mm512_mul_pd(a[1], b[2]);
    o0 = _mm512_mul_pd(a[2], b[1]);
    p0 = _mm512_mul_pd(a[3], b[0]);
    c1 = _bncavx512_dtwo_sum(c0, d0, &d1);
    e1 = _bncavx512_dtwo_sum(e0, f0, &f1);
    g1 = _bncavx512_dtwo_sum(g0, i0, &i1);
    j1 = _mm512_add_pd(j0, l0);
    m1 = _mm512_add_pd(m0, p0);
    n1 = _mm512_add_pd(n0, o0);
    b2 = _bncavx512_dtwo_sum(b0, c1, &c2);
    e2 = _bncavx512_dtwo_sum(e1, h0, &h2);
    f2 = _mm512_add_pd(f1, j1);
    i2 = _mm512_add_pd(i1, k0);
    m2 = _mm512_add_pd(m1, n1);
    a3 = _bncavx512_dquick_two_sum(a0, b2, &b3);
    c3 = _bncavx512_dquick_two_sum(c2, d1, &d3);
    e3 = _bncavx512_dtwo_sum(e2, g1, &g3);
    f3 = _mm512_add_pd(f2, m2);
    h3 = _mm512_add_pd(h2, i2);
    c4 = _bncavx512_dtwo_sum(c3, e3, &e4);
    d4 = _mm512_add_pd(d3, h3);
    f4 = _mm512_add_pd(f3, g3);
    d5 = _mm512_add_pd(d4, e4);
    c6 = _bncavx512_dtwo_sum(c4, d5, &d6);
    b7 = _bncavx512_dtwo_sum(b3, c6, &c7);
    d7 = _mm512_add_pd(d6, f4);
    a8 = _bncavx512_dquick_two_sum(a3, b7, &b8);
    c8 = _bncavx512_dtwo_sum(c7, d7, &d8);
    b9 = _bncavx512_dtwo_sum(b8, c8, &c9);
    c10 = _bncavx512_dquick_two_sum(c9, d8, &d10);
    //return MultiFloat<T, 4>{a8, b9, c10, d10};
	ret[0] = a8;
	ret[1] = b9;
	ret[2] = c10;
	ret[3] = d10;
}

// c[3] := -a[3]
//static inline void c_qd_neg(const double *a, double *c)
static inline void _bncavx512_rqd_neg(__m512d c[QDSIZE], __m512d a[QDSIZE])
{
    __m512d zero8;

    zero8 = _mm512_setzero_pd();

	//c[0] = -a[0];
    c[0] = _mm512_sub_pd(zero8, a[0]);
	//c[1] = -a[1];
    c[1] = _mm512_sub_pd(zero8, a[1]);
    //c[2] = -a[2];
    c[2] = _mm512_sub_pd(zero8, a[2]);    
    //c[3] = -a[3];
    c[3] = _mm512_sub_pd(zero8, a[3]);    

}

/* sub */
// c := a - b
//static inline void c_qd_sub(const double *a, const double *b, double *c)
static inline void _bncavx512_rqd_sub(__m512d c[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
	__m512d mb[QDSIZE];

	// a + (-b)
	//c_qd_neg(b, mb);
    _bncavx512_rqd_neg(mb, b);
	//c_qd_add(a, mb, c);
    _bncavx512_rqd_add(c, a, mb);

#if 0
  qd_real cc;
  cc = qd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}

// c:= a + (double)b
/* quad-double + double */
// static inline void c_qd_add_qd_d(const double *a, double b, double *c)
// 2022-11-22(Tue) T.Kouya
static inline void _bncavx512_rqd_add_d(__m512d c[QDSIZE], const __m512d a[QDSIZE], __m512d b)
{
	//double e;
    __m512d e;

	c[0] = _bncavx512_dtwo_sum(a[0], b, &e);
	c[1] = _bncavx512_dtwo_sum(a[1], e, &e);
	c[2] = _bncavx512_dtwo_sum(a[2], e, &e);
	c[3] = _bncavx512_dtwo_sum(a[3], e, &e);

	//qd::renorm(c0, c1, c2, c3, e);
	_bncavx512_renorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);

//	return qd_real(c0, c1, c2, c3);
	return;
}

// c := a * (double)b
//static inline void c_qd_mul_qd_d(const double *a, double b, double *c)
static inline void _bncavx512_rqd_mul_d(__m512d c[QDSIZE], const __m512d a[QDSIZE], __m512d b)
{
	//double p0, p1, p2, p3;
	//double q0, q1, q2;
	//double s0, s1, s2, s3, s4;
	__m512d p0, p1, p2, p3;
	__m512d q0, q1, q2;
	__m512d s0, s1, s2, s3, s4;

	//p0 = two_prod(a[0], b, &q0);
	//p1 = two_prod(a[1], b, &q1);
	//p2 = two_prod(a[2], b, &q2);
	//p3 = a[3] * b;
	p0 = _bncavx512_dtwo_prod(a[0], b, &q0);
	p1 = _bncavx512_dtwo_prod(a[1], b, &q1);
	p2 = _bncavx512_dtwo_prod(a[2], b, &q2);
	p3 = _mm512_mul_pd(a[3], b);

	s0 = p0;

	//s1 = two_sum(q0, p1, &s2);
    s1 = _bncavx512_dtwo_sum(q0, p1, &s2);

	//three_sum(&s2, &q1, &p2);
	_bncavx512_three_sum(&s2, &q1, &p2);

	//three_sum2(&q1, &q2, &p3);
	_bncavx512_three_sum2(&q1, &q2, &p3);
	s3 = q1;

	//s4 = q2 + p2;
    s4 = _mm512_add_pd(q2, p2);

	//renorm4(&s0, &s1, &s2, &s3, &s4);
	_bncavx512_renorm4(&s0, &s1, &s2, &s3, &s4);
    //	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
}

// c := a / b
//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b)
static inline void _bncavx512_rtd_divq(__m512d c[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
	//double q0, q1, q2, q3;
	//double r[QDSIZE], tmp[QDSIZE];
	__m512d q0, q1, q2, q3;
	__m512d r[TDSIZE], tmp[TDSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm512_div_pd(a[0], b[0]);

	//r = a - (b * q0);
	//c_qd_mul_qd_d(b, q0, tmp);
	//c_qd_sub(a, tmp, r);
	_bncavx512_rtd_mul_d(tmp, b, q0); // q0, b);
	//_bncavx512_rtd_sub(r, a, tmp);
	_bncavx512_rtd_subq(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm512_div_pd(r[0], b[0]);

	//r -= (b * q1);
	_bncavx512_rtd_mul_d(tmp, b, q1); // , b);
	//_bncavx512_rtd_sub(r, tmp, r);
	_bncavx512_rtd_subq(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm512_div_pd(r[0], b[0]);
	//r -= (b * q2);
	//c_qd_mul_qd_d(b, q2, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx512_rtd_mul_d(tmp, b, q2); // , b);
	//_bncavx512_rtd_sub(r, tmp, r);
	_bncavx512_rtd_subq(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm512_div_pd(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx512_renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	//c[3] = q3;
}

/* div */
// c := a / b
//static inline void c_qd_div_sloppy(const double *a, const double *b, double *c)
//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b)
static inline void _bncavx512_rqd_div(__m512d c[QDSIZE], __m512d a[QDSIZE], __m512d b[QDSIZE])
{
	//double q0, q1, q2, q3;
	//double r[QDSIZE], tmp[QDSIZE];
	__m512d q0, q1, q2, q3;
	__m512d r[QDSIZE], tmp[QDSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm512_div_pd(a[0], b[0]);

	//r = a - (b * q0);
	//c_qd_mul_qd_d(b, q0, tmp);
	//c_qd_sub(a, tmp, r);
	_bncavx512_rqd_mul_d(tmp, b, q0);
	_bncavx512_rqd_sub(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm512_div_pd(r[0], b[0]);

	//r -= (b * q1);
	//c_qd_mul_qd_d(b, q1, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx512_rqd_mul_d(tmp, b, q1);
	_bncavx512_rqd_sub(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm512_div_pd(r[0], b[0]);
	//r -= (b * q2);
	//c_qd_mul_qd_d(b, q2, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx512_rqd_mul_d(tmp, b, q2);
	_bncavx512_rqd_sub(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm512_div_pd(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx512_renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}
#endif //defined(__AVX512F__)
#endif //ifndef __BNCAVX_QD_AVX512_H
