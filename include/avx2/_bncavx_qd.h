// ------------------------
// ---------- QD ----------
// ------------------------
#ifndef __BNCAVX_QD_H
#define __BNCAVX_QD_H

#ifndef QDSIZE
    #define QDSIZE 4
#endif // QDSIZE

// ret := 0
#if defined(__AVX2__)
static inline void _bncavx2_set0_qd(__m256d ret[QDSIZE])
{
    ret[0] = _mm256_setzero_pd();
    ret[1] = _mm256_setzero_pd();
    ret[2] = _mm256_setzero_pd();
    ret[3] = _mm256_setzero_pd();
}
#define _bncavx2_rqd_set0(ret) _bncavx2_set0_qd((ret))

// ret := val
static inline void _bncavx2_rqd_set(__m256d ret[QDSIZE], __m256d val[QDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = val[3];
}

// ret := (double)val
static inline void _bncavx2_rqd_set_d(__m256d ret[QDSIZE], __m256d val)
{
    ret[0] = val;
    ret[1] = _mm256_setzero_pd(); // val[1];
    ret[2] = _mm256_setzero_pd(); // val[2];
    ret[3] = _mm256_setzero_pd(); // val[3];
}

// ret := (DD)val
static inline void _bncavx2_rqd_set_dd(__m256d ret[QDSIZE], __m256d val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = _mm256_setzero_pd(); // val[2];
    ret[3] = _mm256_setzero_pd(); // val[3];
}

// ret := (TD)val
static inline void _bncavx2_rqd_set_td(__m256d ret[QDSIZE], __m256d val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
    ret[3] = _mm256_setzero_pd(); // val[3];
}

// ret := [(QD)val (QD)val (QD)val (QD)val]
static inline void _bncavx2_rqd_set1_qd(__m256d ret[QDSIZE], double val[QDSIZE])
{
    ret[0] = _mm256_set1_pd(val[0]);
    ret[1] = _mm256_set1_pd(val[1]);
    ret[2] = _mm256_set1_pd(val[2]);
    ret[3] = _mm256_set1_pd(val[3]);
}

// ret := ret4[][avx_index]
static inline void _bncavx2_get_qd_m256d_i(qdfloat *ret, __m256d ret4[QDSIZE], int avx_index)
{
    ret->val[0] = ret4[0][avx_index];
    ret->val[1] = ret4[1][avx_index];
    ret->val[2] = ret4[2][avx_index];
    ret->val[3] = ret4[3][avx_index];

    return;
}

// ret := mmval[0] + ... + mmval[7]
static void _bncavx2_rqd_sum256d(double ret[QDSIZE], __m256d ret4[QDSIZE])
{
    qdfloat ret4_i[4];

    // ret4_i := ret4
    _bncavx2_get_qd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qd_m256d_i(&ret4_i[3], ret4, 3);

    rqd_set(ret, ret4_i[0].val);
    rqd_add(ret, ret, ret4_i[1].val);
    rqd_add(ret, ret, ret4_i[2].val);
    rqd_add(ret, ret, ret4_i[3].val);
}

// abs
static inline void _bncavx2_rqd_abs(__m256d ret[QDSIZE], __m256d a[QDSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 4; avx_index++)
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
// ret := |ret4[0]| + |ret4[1]| + |ret4[2]| + |ret4[3]|
static void _bncavx2_rqd_abssum256d(double ret[QDSIZE], __m256d ret4[QDSIZE])
{
    qdfloat ret4_i[4];
    double tmp[QDSIZE];

    // ret4_i := ret4
    _bncavx2_get_qd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qd_m256d_i(&ret4_i[3], ret4, 3);

    rqd_abs(tmp, ret4_i[0].val);
    rqd_set(ret, tmp);

    rqd_abs(tmp, ret4_i[1].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret4_i[2].val);
    rqd_add(ret, ret, tmp);

    rqd_abs(tmp, ret4_i[3].val);
    rqd_add(ret, ret, tmp);
}

// ret := max(|ret4[0]|, |ret4[1]|, |ret4[2]|, |ret4[3]|)
static void _bncavx2_rqd_absmax256d(double ret[QDSIZE], __m256d ret4[QDSIZE])
{
    qdfloat ret4_i[4];
    double tmp[QDSIZE];

    // ret4_i := ret4
    _bncavx2_get_qd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qd_m256d_i(&ret4_i[3], ret4, 3);

    rqd_abs(tmp, ret4_i[0].val);
    rqd_set(ret, tmp); // ret := |ret4_i[0]|

    rqd_abs(tmp, ret4_i[1].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[1]|)
        rqd_set(ret, tmp);    //   ret := |ret4_i[1]|

    rqd_abs(tmp, ret4_i[2].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[2]|)
        rqd_set(ret, tmp);    //   ret := |ret4_i[2]|

    rqd_abs(tmp, ret4_i[3].val);
    if(rqd_cmp(ret, tmp) < 0) // if(ret < |ret4_i[3]|)
        rqd_set(ret, tmp);    //   ret := |ret4_i[3]|
}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx2_rqd_norm256d(double ret[QDSIZE], __m256d ret4[QDSIZE])
{
    qdfloat ret4_i[4];
    double tmp[QDSIZE];

    // ret4_i := ret4
    _bncavx2_get_qd_m256d_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qd_m256d_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qd_m256d_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qd_m256d_i(&ret4_i[3], ret4, 3);

    rqd_mul(tmp, ret4_i[0].val, ret4_i[0].val);
    rqd_set(ret, tmp);

    rqd_mul(tmp, ret4_i[1].val, ret4_i[1].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret4_i[2].val, ret4_i[2].val);
    rqd_add(ret, ret, tmp);

    rqd_mul(tmp, ret4_i[3].val, ret4_i[3].val);
    rqd_add(ret, ret, tmp);

    rqd_sqrt(tmp, ret);
    rqd_set(ret, tmp);
}
#endif // __AVX2__

#if defined(__AVX2__)
//inline void renorm(double *c0, double *c1, double *c2, double *c3)
static inline void _bncavx2_renorm(__m256d *c0, __m256d *c1, __m256d *c2, __m256d *c3)
{
// New codes on 2020-11-10 by T.Kouya
    double q0[4], q1[4], q2[4], q3[4];

    q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0];
    q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1];
    q2[0] = (* c0)[2]; q2[1] = (* c1)[2]; q2[2] = (* c2)[2]; q2[3] = (* c3)[2];
    q3[0] = (* c0)[3]; q3[1] = (* c1)[3]; q3[2] = (* c2)[3]; q3[3] = (* c3)[3];

    renorm(&q0[0], &q0[1], &q0[2], &q0[3]);
    renorm(&q1[0], &q1[1], &q1[2], &q1[3]);
    renorm(&q2[0], &q2[1], &q2[2], &q2[3]);
    renorm(&q3[0], &q3[1], &q3[2], &q3[3]);

    (* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3];
    (* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3];
    (* c0)[2] = q2[0]; (* c1)[2] = q2[1]; (* c2)[2] = q2[2]; (* c3)[2] = q2[3];
    (* c0)[3] = q3[0]; (* c1)[3] = q3[1]; (* c2)[3] = q3[2]; (* c3)[3] = q3[3];

#if 0
    int avx_index;
	double s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QD_ISINF(c0)) return;
    for(avx_index = 0; avx_index < 4; avx_index++)
    {
    	//if (isinf(*c0)) return;
        if (isinf((*c0)[avx_index])) continue; //return;

//        s0  = quick_two_sum((*c2)[avx_index], (*c3)[avx_index], &((*c3)[avx_index]));
        s0  = quick_two_sum((*c2)[avx_index], (*c3)[avx_index], (double *)(c3 + avx_index));
//        s0  = quick_two_sum((*c1)[avx_index],  s0, &((*c2)[avx_index]));
        s0  = quick_two_sum((*c1)[avx_index],  s0, (double *)(c2 + avx_index));
        (*c0)[avx_index] = quick_two_sum((*c0)[avx_index],  s0, &((*c1)[avx_index]));

        s0 = (*c0)[avx_index];
        s1 = (*c1)[avx_index];
        if (s1 != 0.0)
        {
            s1 = quick_two_sum(s1, (*c2)[avx_index], &s2);
            if (s2 != 0.0)
                s2 = quick_two_sum(s2, (*c3)[avx_index], &s3);
            else
                s1 = quick_two_sum(s1, (*c3)[avx_index], &s2);
        }
        else
        {
            s0 = quick_two_sum(s0, (*c2)[avx_index], &s1);
            if (s1 != 0.0)
                s1 = quick_two_sum(s1, (*c3)[avx_index], &s2);
            else
                s0 = quick_two_sum(s0, (*c3)[avx_index], &s1);
        }

        //*c0 = s0;
        //*c1 = s1;
        //*c2 = s2;
        //*c3 = s3;
        (*c0)[avx_index] = s0;
        (*c1)[avx_index] = s1;
        (*c2)[avx_index] = s2;
        (*c3)[avx_index] = s3;
    }
#endif // 0
}

//inline void renorm4(double *c0, double *c1, double *c2, double *c3, double *c4)
static inline void _bncavx2_renorm4(__m256d *c0, __m256d *c1, __m256d *c2, __m256d *c3, __m256d *c4)
{
    int avx_index;
	double s0, s1, s2 = 0.0, s3 = 0.0;

// New codes on 2020-11-10 by T.Kouya
    double q0[5], q1[5], q2[5], q3[5];

    q0[0] = (* c0)[0]; q0[1] = (* c1)[0]; q0[2] = (* c2)[0]; q0[3] = (* c3)[0]; q0[4] = (* c4)[0];
    q1[0] = (* c0)[1]; q1[1] = (* c1)[1]; q1[2] = (* c2)[1]; q1[3] = (* c3)[1]; q1[4] = (* c4)[1];
    q2[0] = (* c0)[2]; q2[1] = (* c1)[2]; q2[2] = (* c2)[2]; q2[3] = (* c3)[2]; q2[4] = (* c4)[2];
    q3[0] = (* c0)[3]; q3[1] = (* c1)[3]; q3[2] = (* c2)[3]; q3[3] = (* c3)[3]; q3[4] = (* c4)[3];

    renorm4(&q0[0], &q0[1], &q0[2], &q0[3], &q0[4]);
    renorm4(&q1[0], &q1[1], &q1[2], &q1[3], &q1[4]);
    renorm4(&q2[0], &q2[1], &q2[2], &q2[3], &q2[4]);
    renorm4(&q3[0], &q3[1], &q3[2], &q3[3], &q3[4]);

    (* c0)[0] = q0[0]; (* c1)[0] = q0[1]; (* c2)[0] = q0[2]; (* c3)[0] = q0[3]; (* c4)[0] = q0[4];
    (* c0)[1] = q1[0]; (* c1)[1] = q1[1]; (* c2)[1] = q1[2]; (* c3)[1] = q1[3]; (* c4)[1] = q1[4];
    (* c0)[2] = q2[0]; (* c1)[2] = q2[1]; (* c2)[2] = q2[2]; (* c3)[2] = q2[3]; (* c4)[2] = q2[4];
    (* c0)[3] = q3[0]; (* c1)[3] = q3[1]; (* c2)[3] = q3[2]; (* c3)[3] = q3[3]; (* c4)[3] = q3[4];

#if 0
    double in_c0[4], in_c1[4], in_c2[4], in_c3[4], in_c4[4];
/*
    in_c0[0] = (*c0)[0]; in_c0[1] = (*c0)[1]; in_c0[2] = (*c0)[2]; in_c0[3] = (*c0)[3]; 
    in_c1[0] = (*c1)[0]; in_c1[1] = (*c1)[1]; in_c1[2] = (*c1)[2]; in_c1[3] = (*c1)[3]; 
    in_c2[0] = (*c2)[0]; in_c2[1] = (*c2)[1]; in_c2[2] = (*c2)[2]; in_c2[3] = (*c2)[3]; 
    in_c3[0] = (*c3)[0]; in_c3[1] = (*c3)[1]; in_c3[2] = (*c3)[2]; in_c3[3] = (*c3)[3]; 
    in_c4[0] = (*c4)[0]; in_c4[1] = (*c4)[1]; in_c4[2] = (*c4)[2]; in_c4[3] = (*c4)[3]; 
*/
    in_c0[3] = (*c0)[0]; in_c0[2] = (*c0)[1]; in_c0[1] = (*c0)[2]; in_c0[0] = (*c0)[3]; 
    in_c1[3] = (*c1)[0]; in_c1[2] = (*c1)[1]; in_c1[1] = (*c1)[2]; in_c1[0] = (*c1)[3]; 
    in_c2[3] = (*c2)[0]; in_c2[2] = (*c2)[1]; in_c2[1] = (*c2)[2]; in_c2[0] = (*c2)[3]; 
    in_c3[3] = (*c3)[0]; in_c3[2] = (*c3)[1]; in_c3[1] = (*c3)[2]; in_c3[0] = (*c3)[3]; 
    in_c4[3] = (*c4)[0]; in_c4[2] = (*c4)[1]; in_c4[1] = (*c4)[2]; in_c4[0] = (*c4)[3]; 

//	if (QD_ISINF(c0)) return;
    for(avx_index = 0; avx_index < 4; avx_index++)
    {
        if (isinf(in_c0[avx_index])) continue; //return;

	    // s0  = quick_two_sum(*c3, *c4, c4);
	    // s0  = quick_two_sum(*c2, s0 , c3);
	    // s0  = quick_two_sum(*c1, s0 , c2);
	    // *c0 = quick_two_sum(*c0, s0 , c1);
        s0  = quick_two_sum(in_c3[avx_index], in_c4[avx_index], &in_c4[avx_index]);
        s0  = quick_two_sum(in_c2[avx_index],              s0 , &in_c3[avx_index]);
        s0  = quick_two_sum(in_c1[avx_index],              s0 , &in_c2[avx_index]);
        in_c0[avx_index] = quick_two_sum(in_c0[avx_index], s0 , &in_c1[avx_index]);

    	// s0 = *c0;
	    // s1 = *c1;
        s0 = in_c0[avx_index];
        s1 = in_c1[avx_index];

    //	s0 = quick_two_sum(c0[avx_index], c1[avx_index], &s1);
        if (s1 != 0.0)
        {
		    //s1 = quick_two_sum(s1, *c2, &s2);
            s1 = quick_two_sum(s1, in_c2[avx_index], &s2);
            if (s2 != 0.0)
            {
			    //s2 = quick_two_sum(s2, *c3, &s3);
                s2 = quick_two_sum(s2, in_c3[avx_index], &s3);
                if (s3 != 0.0)
                    s3 += in_c4[avx_index];
                else
                {
				    // s2 = quick_two_sum(s2, *c4, &s3); // fix!: 2020-11-06 by T.Kouya // s2 += *c4;
				    s2 = quick_two_sum(s2, in_c4[avx_index], &s3); // fix!: 2020-11-06 by T.Kouya //  s2 += (*c4)[avx_index];
                }
            }
            else
            {
			    //s1 = quick_two_sum(s1, *c3, &s2);
                s1 = quick_two_sum(s1, in_c3[avx_index], &s2);
                if (s2 != 0.0)
                {
    				//s2 = quick_two_sum(s2, *c4, &s3);
                    s2 = quick_two_sum(s2, in_c4[avx_index], &s3);
                }
                else
                {
				    //s1 = quick_two_sum(s1, *c4, &s2);
                    s1 = quick_two_sum(s1, in_c4[avx_index], &s2);
                }
            }
        }
        else
        {
		    // s0 = quick_two_sum(s0, *c2, &s1);
            s0 = quick_two_sum(s0, in_c2[avx_index], &s1);
            if (s1 != 0.0)
            {
			    //s1 = quick_two_sum(s1, *c3, &s2);
                s1 = quick_two_sum(s1, in_c3[avx_index], &s2);
                if (s2 != 0.0)
                {
				    //s2 = quick_two_sum(s2, *c4, &s3);
                    s2 = quick_two_sum(s2, in_c4[avx_index], &s3);
                }
                else
                {
				    //s1 = quick_two_sum(s1, *c4, &s2);
                    s1 = quick_two_sum(s1, in_c4[avx_index], &s2);
                }
            }
            else
            {
			    //s0 = quick_two_sum(s0, *c3, &s1);
                s0 = quick_two_sum(s0, in_c3[avx_index], &s1);
                if (s1 != 0.0)
                {
				    //s1 = quick_two_sum(s1, *c4, &s2);
                    s1 = quick_two_sum(s1, in_c4[avx_index], &s2);
                }
                else
                {
				    //s0 = quick_two_sum(s0, *c4, &s1);
                    s0 = quick_two_sum(s0, in_c4[avx_index], &s1);
                }
            }
        }

        in_c0[avx_index] = s0;
        in_c1[avx_index] = s1;
        in_c2[avx_index] = s2;
        in_c3[avx_index] = s3;
    }


    *c0 = _mm256_set_pd(in_c0[0], in_c0[1], in_c0[2], in_c0[3]);
    *c1 = _mm256_set_pd(in_c1[0], in_c1[1], in_c1[2], in_c1[3]);
    *c2 = _mm256_set_pd(in_c2[0], in_c2[1], in_c2[2], in_c2[3]);
    *c3 = _mm256_set_pd(in_c3[0], in_c3[1], in_c3[2], in_c3[3]);
    *c4 = _mm256_set_pd(in_c4[0], in_c4[1], in_c4[2], in_c4[3]);

/*
    *c0 = _mm256_set_pd(in_c0[3], in_c0[2], in_c0[1], in_c0[0]);
    *c1 = _mm256_set_pd(in_c1[3], in_c1[2], in_c1[1], in_c1[0]);
    *c2 = _mm256_set_pd(in_c2[3], in_c2[2], in_c2[1], in_c2[0]);
    *c3 = _mm256_set_pd(in_c3[3], in_c3[2], in_c3[1], in_c3[0]);
    *c4 = _mm256_set_pd(in_c4[3], in_c4[2], in_c4[1], in_c4[0]);
*/
#endif // 0
#if 0 //
//	if (QD_ISINF(c0)) return;
    for(avx_index = 0; avx_index < 4; avx_index++)
    {
        if (isinf((*c0)[avx_index])) continue; //return;

//        s0  = quick_two_sum((*c3)[avx_index], (*c4)[avx_index], &((*c4)[avx_index]));
//        s0  = quick_two_sum((*c2)[avx_index], s0 , &((*c3)[avx_index]));
//        s0  = quick_two_sum((*c1)[avx_index], s0 , &((*c2)[avx_index]));
        s0  = quick_two_sum((*c3)[avx_index], (*c4)[avx_index], (double *)&((*c4)[avx_index]));
        s0  = quick_two_sum((*c2)[avx_index], s0 , (double *)&((*c3)[avx_index]));
        s0  = quick_two_sum((*c1)[avx_index], s0 , (double *)&((*c2)[avx_index]));
        (*c0)[avx_index] = quick_two_sum((*c0)[avx_index], s0 , (double *)&((*c1)[avx_index]));

        s0 = (*c0)[avx_index];
        s1 = (*c1)[avx_index];

    //	s0 = quick_two_sum(c0[avx_index], c1[avx_index], &s1);
        if (s1 != 0.0)
        {
            s1 = quick_two_sum(s1, (*c2)[avx_index], &s2);
            if (s2 != 0.0)
            {
                s2 = quick_two_sum(s2, (*c3)[avx_index], &s3);
                if (s3 != 0.0)
                    s3 += (*c4)[avx_index];
                else
				    s2 = quick_two_sum(s2, (*c4)[avx_index], &s3); // fix!: 2020-11-06 by T.Kouya //  s2 += (*c4)[avx_index];
            }
            else
            {
                s1 = quick_two_sum(s1, (*c3)[avx_index], &s2);
                if (s2 != 0.0)
                    s2 = quick_two_sum(s2, (*c4)[avx_index], &s3);
                else
                    s1 = quick_two_sum(s1, (*c4)[avx_index], &s2);
            }
        }
        else
        {
            s0 = quick_two_sum(s0, (*c2)[avx_index], &s1);
            if (s1 != 0.0)
            {
                s1 = quick_two_sum(s1, (*c3)[avx_index], &s2);
                if (s2 != 0.0)
                    s2 = quick_two_sum(s2, (*c4)[avx_index], &s3);
                else
                    s1 = quick_two_sum(s1, (*c4)[avx_index], &s2);
            }
            else
            {
                s0 = quick_two_sum(s0, (*c3)[avx_index], &s1);
                if (s1 != 0.0)
                    s1 = quick_two_sum(s1, (*c4)[avx_index], &s2);
                else
                    s0 = quick_two_sum(s0, (*c4)[avx_index], &s1);
            }
        }

        (*c0)[avx_index] = s0;
        (*c1)[avx_index] = s1;
        (*c2)[avx_index] = s2;
        (*c3)[avx_index] = s3;
    }
#endif // 0
}


/********** Additions ************/
//inline void three_sum(double *a, double *b, double *c)
static inline void _bncavx2_three_sum(__m256d *a, __m256d *b, __m256d *c)
{
//	double t1, t2, t3;
    __m256d t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = two_sum(t2, t3, c);
	t1 = _bncavx2_dtwo_sum(*a, *b, &t2);
	*a = _bncavx2_dtwo_sum(*c, t1, &t3);
	*b = _bncavx2_dtwo_sum(t2, t3, c);

}

//inline void three_sum2(double *a, double *b, double *c)
static inline void _bncavx2_three_sum2(__m256d *a, __m256d *b, __m256d *c)
{
//	double t1, t2, t3;
    __m256d t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = t2 + t3;
	t1 = _bncavx2_dtwo_sum(*a, *b, &t2);
	*a = _bncavx2_dtwo_sum(*c, t1, &t3);
	*b = _mm256_add_pd(t2, t3);

}

#ifdef USE_QD_BF
    #define _bncavx2_rqd_add _bncavx2_rqd_add_bf
#else // USE_QD_BF
    #define _bncavx2_rqd_add _bncavx2_rqd_add_sloppy
#endif // USE_QD_BF

//static inline void _bncavx2_rqd_add(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
static inline void _bncavx2_rqd_add_sloppy(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
//#ifdef BNC_USE_ICC
#if 0
    double in_ret[4][QDSIZE], in_a[4][QDSIZE], in_b[4][QDSIZE];

    //in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
    //in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
    //in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
    //in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];

    rqd_add(in_ret[0], in_a[0], in_b[0]);
    rqd_add(in_ret[1], in_a[1], in_b[1]);
    rqd_add(in_ret[2], in_a[2], in_b[2]);
    rqd_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
//#else // ifdef BNC_USE_ICC
    // c_qd_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	double s0, s1, s2, s3;
//	double t0, t1, t2, t3;
	__m256d s0, s1, s2, s3;
	__m256d t0, t1, t2, t3;

//	double v0, v1, v2, v3;
//	double u0, u1, u2, u3;
//	double w0, w1, w2, w3;
	__m256d v0, v1, v2, v3;
	__m256d u0, u1, u2, u3;
	__m256d w0, w1, w2, w3;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm256_add_pd(a[0], b[0]);
	s1 = _mm256_add_pd(a[1], b[1]);
	s2 = _mm256_add_pd(a[2], b[2]);
	s3 = _mm256_add_pd(a[3], b[3]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm256_sub_pd(s0, a[0]);
	v1 = _mm256_sub_pd(s1, a[1]);
	v2 = _mm256_sub_pd(s2, a[2]);
	v3 = _mm256_sub_pd(s3, a[3]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm256_sub_pd(s0, v0);
	u1 = _mm256_sub_pd(s1, v1);
	u2 = _mm256_sub_pd(s2, v2);
	u3 = _mm256_sub_pd(s3, v3);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm256_sub_pd(a[0], u0);
	w1 = _mm256_sub_pd(a[1], u1);
	w2 = _mm256_sub_pd(a[2], u2);
	w3 = _mm256_sub_pd(a[3], u3);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm256_sub_pd(b[0], v0);
	u1 = _mm256_sub_pd(b[1], v1);
	u2 = _mm256_sub_pd(b[2], v2);
	u3 = _mm256_sub_pd(b[3], v3);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm256_add_pd(w0, u0);
	t1 = _mm256_add_pd(w1, u1);
	t2 = _mm256_add_pd(w2, u2);
	t3 = _mm256_add_pd(w3, u3);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx2_dtwo_sum(s1, t0, &t0);
	_bncavx2_three_sum(&s2, &t0, &t1);
	_bncavx2_three_sum2(&s3, &t0, &t2);
	t0 = _mm256_add_pd(_mm256_add_pd(t0, t1), t3);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx2_renorm4(&s0, &s1, &s2, &s3, &t0);
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
//#endif // 0
//#endif //def BNC_USE_ICC
}

// 2025-12-25(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Add4(const double x[4], const double y[4], double z[4]) {
static inline void _bncavx2_rqd_add_bf(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
	__m256d a0 , b0 , c0 , d0 , e0, f0, g0, h0;
	__m256d a1 , b1 , c1 , d1 , e1, f1, g1, h1;
	__m256d a2 , b2 , c2 , d2 , e2, f2, g2;
	__m256d a3 , b3 , c3 , d3 , e3, f3, g3;
	__m256d a4 , b4 , c4 , d4 , e4;
	__m256d a5 , b5 , c5 , d5 , e5;
	__m256d a6 , b6 , c6 , d6 , e6;
	__m256d a7 , b7 , c7 , d7 , e7;
	__m256d a8 , b8 , c8 , d8 , e8;
	__m256d a9 , b9 , c9 , d9 ;
	__m256d a10, b10, c10, d10;
	__m256d a11, b11, c11, d11;
	__m256d a12, b12, c12, d12;

	a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    g0 = a[3];
    h0 = b[3];
    a1 = _bncavx2_dtwo_sum(a0, b0, &b1);
    c1 = _bncavx2_dtwo_sum(c0, d0, &d1);
    e1 = _bncavx2_dtwo_sum(e0, f0, &f1);
    g1 = _bncavx2_dtwo_sum(g0, h0, &h1);
    a2 = _bncavx2_dquick_two_sum(a1, c1, &c2);
    b2 = _mm256_add_pd(b1, h1);
    d2 = _bncavx2_dtwo_sum(d1, e1, &e2);
    f2 = _bncavx2_dtwo_sum(f1, g1, &g2);
    b3 = _bncavx2_dtwo_sum(b2, g2, &g3);
    c3 = _bncavx2_dquick_two_sum(c2, d2, &d3);
    e3 = _bncavx2_dtwo_sum(e2, f2, &f3);
    a4 = _bncavx2_dquick_two_sum(a2, c3, &c4);
    d4 = _bncavx2_dquick_two_sum(d3, e3, &e4);
    b5 = _bncavx2_dtwo_sum(b3, d4, &d5);
    e5 = _mm256_add_pd(e4, f3);
    b6 = _bncavx2_dtwo_sum(b5, c4, &c6);
    d6 = _bncavx2_dtwo_sum(d5, e5, &e6);
    a7 = _bncavx2_dquick_two_sum(a4, b6, &b7);
    c7 = _bncavx2_dquick_two_sum(c6, d6, &d7);
    e8 = _mm256_add_pd(e6, g3);
    b8 = _bncavx2_dquick_two_sum(b7, c7, &c8);
    d9 = _mm256_add_pd(d7, e8);
    a10 = _bncavx2_dquick_two_sum(a7, b8, &b10);
    c10 = _bncavx2_dquick_two_sum(c8, d9, &d10);
    b11 = _bncavx2_dquick_two_sum(b10, c10, &c11);
    c12 = _bncavx2_dquick_two_sum(c11, d10, &d12);
    //return MultiFloat<T, 4>{a10, b11, c12, d12};
	ret[0] = a10;
	ret[1] = b11;
	ret[2] = c12;
	ret[3] = d12;
}

// rqd_add -> rtd_add
static inline void _bncavx2_rtd_addq(__m256d ret[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
#if 0
    double in_ret[4][QDSIZE], in_a[4][QDSIZE], in_b[4][QDSIZE];

    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];

    rqd_add(in_ret[0], in_a[0], in_b[0]);
    rqd_add(in_ret[1], in_a[1], in_b[1]);
    rqd_add(in_ret[2], in_a[2], in_b[2]);
    rqd_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
    // c_qd_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	double s0, s1, s2, s3;
//	double t0, t1, t2, t3;
	__m256d s0, s1, s2;
	__m256d t0, t1, t2;

//	double v0, v1, v2;
//	double u0, u1, u2;
//	double w0, w1, w2;
	__m256d v0, v1, v2;
	__m256d u0, u1, u2;
	__m256d w0, w1, w2;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm256_add_pd(a[0], b[0]);
	s1 = _mm256_add_pd(a[1], b[1]);
	s2 = _mm256_add_pd(a[2], b[2]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm256_sub_pd(s0, a[0]);
	v1 = _mm256_sub_pd(s1, a[1]);
	v2 = _mm256_sub_pd(s2, a[2]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm256_sub_pd(s0, v0);
	u1 = _mm256_sub_pd(s1, v1);
	u2 = _mm256_sub_pd(s2, v2);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm256_sub_pd(a[0], u0);
	w1 = _mm256_sub_pd(a[1], u1);
	w2 = _mm256_sub_pd(a[2], u2);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm256_sub_pd(b[0], v0);
	u1 = _mm256_sub_pd(b[1], v1);
	u2 = _mm256_sub_pd(b[2], v2);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm256_add_pd(w0, u0);
	t1 = _mm256_add_pd(w1, u1);
	t2 = _mm256_add_pd(w2, u2);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx2_dtwo_sum(s1, t0, &t0);
	_bncavx2_three_sum(&s2, &t0, &t1);
//	_bncavx2_three_sum2(&s2, &t0, &t1);
	t0 = _mm256_add_pd(_mm256_add_pd(t0, t1), t2);
//	t0 = _mm256_add_pd(t0, t1);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx2_renorm(&s0, &s1, &s2, &t0);
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
static inline void _bncavx2_rtd_mulq(__m256d ret[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
//#if 0
    // c_qd_mul_sloppy
/*
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;
*/
    __m256d p0, p1, p2, p3, p4, p5;
    __m256d q0, q1, q2, q3, q4, q5;
    __m256d t0, t1;
    __m256d s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx2_dtwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx2_dtwo_prod(a[0], b[1], &q1);
	p2 = _bncavx2_dtwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx2_dtwo_prod(a[0], b[2], &q3);
	p4 = _bncavx2_dtwo_prod(a[1], b[1], &q4);
	p5 = _bncavx2_dtwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx2_three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx2_three_sum2(&p2, &q1, &q2);
	_bncavx2_three_sum2(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx2_dtwo_sum(p2, p3, &t0);
	s1 = _bncavx2_dtwo_sum(q1, p4, &t1);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx2_dtwo_sum(s1, t0, &t0);

	/* O(eps^3) order terms */
	//s1 += a[1]*b[2] + a[2]*b[1] + q0 + q3 + q4;
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[1], b[2]));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[2], b[1]));
    s1 = _mm256_add_pd(s1, q0);
    s1 = _mm256_add_pd(s1, q3);
    s1 = _mm256_add_pd(s1, q4);

	//renorm(p0, p1, s0, s1, s2);
//	renorm(&p0, &p1, &s0, &s1);
	_bncavx2_renorm(&p0, &p1, &s0, &s1);
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
    #define _bncavx2_rqd_mul _bncavx2_rqd_mul_bf
#else // USE_QD_BF
    #define _bncavx2_rqd_mul _bncavx2_rqd_mul_sloppy
#endif // USE_QD_BF

// mul
//static inline void _bncavx2_rqd_mul(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
static inline void _bncavx2_rqd_mul_sloppy(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
//#ifdef BNC_USE_ICC
#if 0
    double in_ret[4][QDSIZE], in_a[4][QDSIZE], in_b[4][QDSIZE];

//    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0]; in_ret[0][3] = ret[3][0];
//    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1]; in_ret[1][3] = ret[3][1];
//    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2]; in_ret[2][3] = ret[3][2];
//    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3]; in_ret[3][3] = ret[3][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0]; in_b[0][3] = b[3][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1]; in_b[1][3] = b[3][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2]; in_b[2][3] = b[3][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3]; in_b[3][3] = b[3][3];

    rqd_mul(in_ret[0], in_a[0], in_b[0]);
    rqd_mul(in_ret[1], in_a[1], in_b[1]);
    rqd_mul(in_ret[2], in_a[2], in_b[2]);
    rqd_mul(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
//#else //ifdef BNC_USE_ICC
    // c_qd_mul_sloppy
/*
	double p0, p1, p2, p3, p4, p5;
	double q0, q1, q2, q3, q4, q5;
	double t0, t1;
	double s0, s1, s2;
*/
    __m256d p0, p1, p2, p3, p4, p5;
    __m256d q0, q1, q2, q3, q4, q5;
    __m256d t0, t1;
    __m256d s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx2_dtwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx2_dtwo_prod(a[0], b[1], &q1);
	p2 = _bncavx2_dtwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx2_dtwo_prod(a[0], b[2], &q3);
	p4 = _bncavx2_dtwo_prod(a[1], b[1], &q4);
	p5 = _bncavx2_dtwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx2_three_sum(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx2_three_sum(&p2, &q1, &q2);
	_bncavx2_three_sum(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx2_dtwo_sum(p2, p3, &t0);
	s1 = _bncavx2_dtwo_sum(q1, p4, &t1);
//	s2 = q2 + p5;
	s2 = _mm256_add_pd(q2, p5);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx2_dtwo_sum(s1, t0, &t0);
//	s2 += (t0 + t1);
	s2 = _mm256_add_pd(s2, _mm256_add_pd(t0, t1));

	/* O(eps^3) order terms */
	//s1 += a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5;
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[0], b[3]));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[1], b[2]));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[2], b[1]));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(a[3], b[0]));
    s1 = _mm256_add_pd(s1, q0);
    s1 = _mm256_add_pd(s1, q3);
    s1 = _mm256_add_pd(s1, q4);
    s1 = _mm256_add_pd(s1, q5);

	//renorm(p0, p1, s0, s1, s2);
//	renorm4(&p0, &p1, &s0, &s1, &s2);
	_bncavx2_renorm4(&p0, &p1, &s0, &s1, &s2);
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

//#endif // 0
//#endif //def BNC_USE_ICC
}
//#endif //defined(__AVX2__)

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Mul4(const double x[4], const double y[4], double z[4]) {
static inline void _bncavx2_rqd_mul_bf(__m256d ret[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
	__m256d a0, b0, c0, d0, e0, f0, g0, h0, i0, j0, k0, l0, m0, n0, o0, p0;
	__m256d a1, b1, c1, d1, e1, f1, g1, h1, i1, j1, k1, l1, m1, n1;
	__m256d a2, b2, c2, d2, e2, f2, g2, h2, i2, j2, k2, l2, m2;
	__m256d a3, b3, c3, d3, e3, f3, g3, h3;
	__m256d a4, b4, c4, d4, e4, f4;
	__m256d a5, b5, c5, d5;
	__m256d a6, b6, c6, d6;
	__m256d a7, b7, c7, d7;
	__m256d a8, b8, c8, d8;
	__m256d a9, b9, c9, d9;
	__m256d a10, b10, c10, d10;

    a0 = _bncavx2_dtwo_prod(a[0], b[0], &b0);
    c0 = _bncavx2_dtwo_prod(a[0], b[1], &e0);
    d0 = _bncavx2_dtwo_prod(a[1], b[0], &f0);
    g0 = _bncavx2_dtwo_prod(a[0], b[2], &j0);
    h0 = _bncavx2_dtwo_prod(a[1], b[1], &k0);
    i0 = _bncavx2_dtwo_prod(a[2], b[0], &l0);
    m0 = _mm256_mul_pd(a[0], b[3]);
    n0 = _mm256_mul_pd(a[1], b[2]);
    o0 = _mm256_mul_pd(a[2], b[1]);
    p0 = _mm256_mul_pd(a[3], b[0]);
    c1 = _bncavx2_dtwo_sum(c0, d0, &d1);
    e1 = _bncavx2_dtwo_sum(e0, f0, &f1);
    g1 = _bncavx2_dtwo_sum(g0, i0, &i1);
    j1 = _mm256_add_pd(j0, l0);
    m1 = _mm256_add_pd(m0, p0);
    n1 = _mm256_add_pd(n0, o0);
    b2 = _bncavx2_dtwo_sum(b0, c1, &c2);
    e2 = _bncavx2_dtwo_sum(e1, h0, &h2);
    f2 = _mm256_add_pd(f1, j1);
    i2 = _mm256_add_pd(i1, k0);
    m2 = _mm256_add_pd(m1, n1);
    a3 = _bncavx2_dquick_two_sum(a0, b2, &b3);
    c3 = _bncavx2_dquick_two_sum(c2, d1, &d3);
    e3 = _bncavx2_dtwo_sum(e2, g1, &g3);
    f3 = _mm256_add_pd(f2, m2);
    h3 = _mm256_add_pd(h2, i2);
    c4 = _bncavx2_dtwo_sum(c3, e3, &e4);
    d4 = _mm256_add_pd(d3, h3);
    f4 = _mm256_add_pd(f3, g3);
    d5 = _mm256_add_pd(d4, e4);
    c6 = _bncavx2_dtwo_sum(c4, d5, &d6);
    b7 = _bncavx2_dtwo_sum(b3, c6, &c7);
    d7 = _mm256_add_pd(d6, f4);
    a8 = _bncavx2_dquick_two_sum(a3, b7, &b8);
    c8 = _bncavx2_dtwo_sum(c7, d7, &d8);
    b9 = _bncavx2_dtwo_sum(b8, c8, &c9);
    c10 = _bncavx2_dquick_two_sum(c9, d8, &d10);
    //return MultiFloat<T, 4>{a8, b9, c10, d10};
	ret[0] = a8;
	ret[1] = b9;
	ret[2] = c10;
	ret[3] = d10;
}

// c[3] := -a[3]
//static inline void c_qd_neg(const double *a, double *c)
static inline void _bncavx2_rqd_neg(__m256d c[QDSIZE], __m256d a[QDSIZE])
{
    __m256d zero4;

    zero4 = _mm256_setzero_pd();

	//c[0] = -a[0];
    c[0] = _mm256_sub_pd(zero4, a[0]);
	//c[1] = -a[1];
    c[1] = _mm256_sub_pd(zero4, a[1]);
    //c[2] = -a[2];
    c[2] = _mm256_sub_pd(zero4, a[2]);    
    //c[3] = -a[3];
    c[3] = _mm256_sub_pd(zero4, a[3]);    

}
/* sub */
// c := a - b
//static inline void c_qd_sub(const double *a, const double *b, double *c)
static inline void _bncavx2_rqd_sub(__m256d c[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
	__m256d mb[QDSIZE];

	// a + (-b)
	//c_qd_neg(b, mb);
    _bncavx2_rqd_neg(mb, b);
	//c_qd_add(a, mb, c);
    _bncavx2_rqd_add(c, a, mb);

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
static inline void _bncavx2_rqd_add_d(__m256d c[QDSIZE], const __m256d a[QDSIZE], __m256d b)
{

#if 0
    double in_ret[4][QDSIZE], in_a[4][QDSIZE], in_b[4];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0]; in_a[0][3] = a[3][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1]; in_a[1][3] = a[3][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2]; in_a[2][3] = a[3][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3]; in_a[3][3] = a[3][3];

    rqd_add_d(in_ret[0], in_a[0], b[0]);
    rqd_add_d(in_ret[1], in_a[1], b[1]);
    rqd_add_d(in_ret[2], in_a[2], b[2]);
    rqd_add_d(in_ret[3], in_a[3], b[3]);

    c[0][0] = in_ret[0][0]; c[1][0] = in_ret[0][1]; c[2][0] = in_ret[0][2]; c[3][0] = in_ret[0][3];
    c[0][1] = in_ret[1][0]; c[1][1] = in_ret[1][1]; c[2][1] = in_ret[1][2]; c[3][1] = in_ret[1][3];
    c[0][2] = in_ret[2][0]; c[1][2] = in_ret[2][1]; c[2][2] = in_ret[2][2]; c[3][2] = in_ret[2][3];
    c[0][3] = in_ret[3][0]; c[1][3] = in_ret[3][1]; c[2][3] = in_ret[3][2]; c[3][3] = in_ret[3][3];
#endif // 0
//#if 0
	//double e;
    __m256d e0, e1, s0, s1, s2, s3;

	//c[0] = _bncavx2_dtwo_sum(a[0], b, &e);
	//c[1] = _bncavx2_dtwo_sum(a[1], e, &e);
	//c[2] = _bncavx2_dtwo_sum(a[2], e, &e);
	//c[3] = _bncavx2_dtwo_sum(a[3], e, &e);
	s0 = _bncavx2_dtwo_sum(a[0], b,  &e0);
	s1 = _bncavx2_dtwo_sum(a[1], e0, &e1);
	s2 = _bncavx2_dtwo_sum(a[2], e1, &e0);
	s3 = _bncavx2_dtwo_sum(a[3], e0, &e1);

	//qd::renorm(c0, c1, c2, c3, e);
	//_bncavx2_renorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);
	_bncavx2_renorm4(&s0, &s1, &s2, &s3, &e1);

//	return qd_real(c0, c1, c2, c3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
//#endif // 0

	return;
}

// c:= a + (double)b
/* triple-double + double */
// static inline void c_qd_add_qd_d(const double *a, double b, double *c)
// 2024-09-04(Wed) T.Kouya
static inline void _bncavx2_rtd_addq_d(__m256d c[TDSIZE], const __m256d a[TDSIZE], __m256d b)
{
	//double e;
    __m256d e0, e1, s0, s1, s2, s3;

	//c[0] = _bncavx2_dtwo_sum(a[0], b, &e);
	//c[1] = _bncavx2_dtwo_sum(a[1], e, &e);
	//c[2] = _bncavx2_dtwo_sum(a[2], e, &e);
	//c[3] = _bncavx2_dtwo_sum(a[3], e, &e);
	s0 = _bncavx2_dtwo_sum(a[0], b,  &e0);
	s1 = _bncavx2_dtwo_sum(a[1], e0, &e1);
	s2 = _bncavx2_dtwo_sum(a[2], e1, &e0);
	//s3 = _bncavx2_dtwo_sum(a[3], e0, &e1);

	//qd::renorm(c0, c1, c2, c3, e);
	//_bncavx2_renorm4(&(c[0]), &(c[1]), &(c[2]), &(c[3]), &e);
	//_bncavx2_renorm4(&s0, &s1, &s2, &s3, &e1);
	_bncavx2_renorm(&s0, &s1, &s2, &e0);

//	return qd_real(c0, c1, c2, c3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	//c[3] = s3;
//#endif // 0

	return;
}

// c := a * (double)b
//static inline void c_qd_mul_qd_d(const double *a, double b, double *c)
static inline void _bncavx2_rqd_mul_d(__m256d c[QDSIZE], const __m256d a[QDSIZE], __m256d b)
{
	//double p0, p1, p2, p3;
	//double q0, q1, q2;
	//double s0, s1, s2, s3, s4;
	__m256d p0, p1, p2, p3;
	__m256d q0, q1, q2;
	__m256d s0, s1, s2, s3, s4;

	//p0 = two_prod(a[0], b, &q0);
	//p1 = two_prod(a[1], b, &q1);
	//p2 = two_prod(a[2], b, &q2);
	//p3 = a[3] * b;
	p0 = _bncavx2_dtwo_prod(a[0], b, &q0);
	p1 = _bncavx2_dtwo_prod(a[1], b, &q1);
	p2 = _bncavx2_dtwo_prod(a[2], b, &q2);
	p3 = _mm256_mul_pd(a[3], b);

	s0 = p0;

	//s1 = two_sum(q0, p1, &s2);
    s1 = _bncavx2_dtwo_sum(q0, p1, &s2);

	//three_sum(&s2, &q1, &p2);
	_bncavx2_three_sum(&s2, &q1, &p2);

	//three_sum2(&q1, &q2, &p3);
	_bncavx2_three_sum2(&q1, &q2, &p3);
	s3 = q1;

	//s4 = q2 + p2;
    s4 = _mm256_add_pd(q2, p2);

	//renorm4(&s0, &s1, &s2, &s3, &s4);
	_bncavx2_renorm4(&s0, &s1, &s2, &s3, &s4);
    //	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
}

// 2024-09-04(Wed)
// c := a * (double)b
//static inline void c_qd_mul_qd_d(const double *a, double b, double *c)
static inline void _bncavx2_rtd_mulq_d(__m256d c[TDSIZE], const __m256d a[TDSIZE], __m256d b)
{
	//double p0, p1, p2, p3;
	//double q0, q1, q2;
	//double s0, s1, s2, s3, s4;
	__m256d p0, p1, p2, p3;
	__m256d q0, q1, q2;
	__m256d s0, s1, s2, s3, s4;

	//p0 = two_prod(a[0], b, &q0);
	//p1 = two_prod(a[1], b, &q1);
	//p2 = two_prod(a[2], b, &q2);
	//p3 = a[3] * b;
	p0 = _bncavx2_dtwo_prod(a[0], b, &q0);
	p1 = _bncavx2_dtwo_prod(a[1], b, &q1);
	p2 = _bncavx2_dtwo_prod(a[2], b, &q2);
	//p3 = _mm256_mul_pd(a[3], b);

	s0 = p0;

	//s1 = two_sum(q0, p1, &s2);
    s1 = _bncavx2_dtwo_sum(q0, p1, &s2);

	//three_sum(&s2, &q1, &p2);
	_bncavx2_three_sum(&s2, &q1, &p2);

	//three_sum2(&q1, &q2, &p3);
	//_bncavx2_three_sum2(&q1, &q2, &p3);
	//s3 = q1;

	//s4 = q2 + p2;
    s4 = _mm256_add_pd(q2, p2);

	//renorm4(&s0, &s1, &s2, &s3, &s4);
	//_bncavx2_renorm4(&s0, &s1, &s2, &s3, &s4);
	_bncavx2_renorm(&s0, &s1, &s2, &s4);
    //	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	//c[3] = s3;
}


// c := a / b
//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b)
static inline void _bncavx2_rtd_divq(__m256d c[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
	//double q0, q1, q2, q3;
	//double r[QDSIZE], tmp[QDSIZE];
	__m256d q0, q1, q2, q3;
	__m256d r[TDSIZE], tmp[TDSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm256_div_pd(a[0], b[0]);

	//r = a - (b * q0);
	//c_qd_mul_qd_d(b, q0, tmp);
	//c_qd_sub(a, tmp, r);
	_bncavx2_rtd_mul_d(tmp, q0, b);
	//_bncavx2_rtd_sub(r, a, tmp);
	_bncavx2_rtd_subq(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm256_div_pd(r[0], b[0]);

	//r -= (b * q1);
	_bncavx2_rtd_mul_d(tmp, q1, b);
	//_bncavx2_rtd_sub(r, tmp, r);
	_bncavx2_rtd_subq(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm256_div_pd(r[0], b[0]);
	//r -= (b * q2);
	//c_qd_mul_qd_d(b, q2, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx2_rtd_mul_d(tmp, q2, b);
	//_bncavx2_rtd_sub(r, tmp, r);
	_bncavx2_rtd_subq(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm256_div_pd(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx2_renorm(&q0, &q1, &q2, &q3);

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
static inline void _bncavx2_rqd_div(__m256d c[QDSIZE], __m256d a[QDSIZE], __m256d b[QDSIZE])
{
	//double q0, q1, q2, q3;
	//double r[QDSIZE], tmp[QDSIZE];
	__m256d q0, q1, q2, q3;
	__m256d r[QDSIZE], tmp[QDSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm256_div_pd(a[0], b[0]);

	//r = a - (b * q0);
	//c_qd_mul_qd_d(b, q0, tmp);
	//c_qd_sub(a, tmp, r);
	_bncavx2_rqd_mul_d(tmp, b, q0);
	_bncavx2_rqd_sub(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm256_div_pd(r[0], b[0]);

	//r -= (b * q1);
	//c_qd_mul_qd_d(b, q1, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx2_rqd_mul_d(tmp, b, q1);
	_bncavx2_rqd_sub(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm256_div_pd(r[0], b[0]);
	//r -= (b * q2);
	//c_qd_mul_qd_d(b, q2, tmp);
	//c_qd_selfsub(tmp, r);
	_bncavx2_rqd_mul_d(tmp, b, q2);
	_bncavx2_rqd_sub(r, tmp, r);
	//c_qd_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm256_div_pd(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx2_renorm(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}
#endif //defined(__AVX2__)

//#ifdef __BNC_QDLINEAR_H__
//#include "qdv_addmul.c"
//#endif //__BNC_QDLINEAR_H__

#ifdef __BNC_DDLINEAR_H__
#ifdef USE_MPFR
// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
static void set_test_ddmatrix(ddfloat mat[], int sqrt_seed, int row_dim, int col_dim)
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

    for(i = 0; i < row_dim; i++)
    {
        //printf("%5d: ", i);
        for(j = 0; j < col_dim; j++)
        {
            rdd_set_ui(mat[i * col_dim + j].val, i + j + 1);
            rdd_mul(mat[i * col_dim + j].val, mat[i * col_dim + j].val, ddsqrt.val);
            //rdd_out_str(mat[i * col_dim + j].val); printf(" ");
        }
        //printf("\n");
    }
}
#endif // USE_MPFR

// ddmatmul
static void ddmatmul(ddfloat ret[], ddfloat mat_a[], ddfloat mat_b[], int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    ddfloat tmp_add, tmp_mul;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rdd_set_ui(tmp_add.val, 0UL);
            for(k = 0; k < mid_dim; k++)
            {
                rdd_mul(tmp_mul.val, mat_a[i * mid_dim + k].val, mat_b[k * col_dim + j].val);
                rdd_add(tmp_add.val, tmp_add.val, tmp_mul.val);
            }
            rdd_set(ret[i * col_dim + j].val, tmp_add.val);
        }
    }
}

#endif //__BNC_DDLINEAR_H__

#ifdef __BNC_TDLINEAR_H__
#if 0
// tdrel_diff
static inline tdfloat tdrel_diff(tdfloat a, tdfloat b)
{
    tdfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rtd_sub(rel_diff.val, a.val, b.val);
    rtd_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rtd_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rtd_abs(abs_a.val, a.val);
        rtd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
#endif // 0

// qdrel_diff
static inline qdfloat qdrel_diff(qdfloat a, qdfloat b)
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
#endif //def __BNC_TDLINEAR_H__

#if defined(__BNC_QDLINEAR_H__) && defined(_DEF_BNC_QDVECTOR)
// qdmatmul_qdvec
static void qdmatmul_qdvec(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qdfloat tmp_mul, aik, bkj, cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqd_set_ui(cij.val, 0UL);
            for(k = 0; k < mid_dim; k++)
            {
                aik.val[0] = mat_a->element[0][i * mid_dim + k];
                aik.val[1] = mat_a->element[1][i * mid_dim + k];
                aik.val[2] = mat_a->element[2][i * mid_dim + k];
                aik.val[3] = mat_a->element[3][i * mid_dim + k];

                bkj.val[0] = mat_b->element[0][k * col_dim + j];
                bkj.val[1] = mat_b->element[1][k * col_dim + j];
                bkj.val[2] = mat_b->element[2][k * col_dim + j];
                bkj.val[3] = mat_b->element[3][k * col_dim + j];

                rqd_mul(tmp_mul.val, aik.val, bkj.val);
                rqd_add(cij.val, cij.val, tmp_mul.val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];        }
    }
}

// qdmatmul_qdvec_ur4
static void qdmatmul_qdvec_ur4(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qdfloat tmp_mul[4], aik[4], bkj[4], cij;
    //double cijval[4][QDSIZE];
    //__m256d tmp_mul[QDSIZE], aik[QDSIZE], bkj[QDSIZE], cij[QDSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqd_set_ui(cij.val, 0UL);
            //cij[0] = _mm256_setzero_pd();
            //cij[1] = _mm256_setzero_pd();
            for(k = 0; k < mid_dim; k += 4)
            {
                aik[0].val[0] = mat_a->element[0][i * mid_dim + k];
                aik[1].val[0] = mat_a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = mat_a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = mat_a->element[0][i * mid_dim + k + 3];

                aik[0].val[1] = mat_a->element[1][i * mid_dim + k];
                aik[1].val[1] = mat_a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = mat_a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = mat_a->element[1][i * mid_dim + k + 3];

                aik[0].val[2] = mat_a->element[2][i * mid_dim + k];
                aik[1].val[2] = mat_a->element[2][i * mid_dim + k + 1];
                aik[2].val[2] = mat_a->element[2][i * mid_dim + k + 2];
                aik[3].val[2] = mat_a->element[2][i * mid_dim + k + 3];

                aik[0].val[3] = mat_a->element[3][i * mid_dim + k];
                aik[1].val[3] = mat_a->element[3][i * mid_dim + k + 1];
                aik[2].val[3] = mat_a->element[3][i * mid_dim + k + 2];
                aik[3].val[3] = mat_a->element[3][i * mid_dim + k + 3];

                bkj[0].val[0] = mat_b->element[0][k * col_dim + j];
                bkj[1].val[0] = mat_b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = mat_b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = mat_b->element[0][(k + 3) * col_dim + j];

                bkj[0].val[1] = mat_b->element[1][k * col_dim + j];
                bkj[1].val[1] = mat_b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = mat_b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = mat_b->element[1][(k + 3) * col_dim + j];

                bkj[0].val[2] = mat_b->element[2][k * col_dim + j];
                bkj[1].val[2] = mat_b->element[2][(k + 1) * col_dim + j];
                bkj[2].val[2] = mat_b->element[2][(k + 2) * col_dim + j];
                bkj[3].val[2] = mat_b->element[2][(k + 3) * col_dim + j];

                bkj[0].val[3] = mat_b->element[3][k * col_dim + j];
                bkj[1].val[3] = mat_b->element[3][(k + 1) * col_dim + j];
                bkj[2].val[3] = mat_b->element[3][(k + 2) * col_dim + j];
                bkj[3].val[3] = mat_b->element[3][(k + 3) * col_dim + j];

                rqd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);

                rqd_add(cij.val, cij.val, tmp_mul[0].val);
                rqd_add(cij.val, cij.val, tmp_mul[1].val);
                rqd_add(cij.val, cij.val, tmp_mul[2].val);
                rqd_add(cij.val, cij.val, tmp_mul[3].val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];        }
    }
}

#if defined(__AVX2__)
// qdmatmul_qdvec_avx2
static void qdmatmul_qdvec_avx2(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
//    qdfloat tmp_mul[4], aik[4], bkj[4], cij;
    double cijval[4][QDSIZE];
    __m256d cij[QDSIZE], aik[QDSIZE], bkj[QDSIZE], tmp_mul[QDSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            //rqd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
            cij[2] = _mm256_setzero_pd();
            cij[3] = _mm256_setzero_pd();

            for(k = 0; k < mid_dim; k += 4)
            {
            /*
                aik[0].val[0] = mat_a->element[0][i * mid_dim + k];
                aik[1].val[0] = mat_a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = mat_a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = mat_a->element[0][i * mid_dim + k + 3];
            */
                //aik[0] = _mm256_loadu_pd(&(mat_a->element[0][i * mid_dim + k]));
                aik[0] = _mm256_set_pd(
                    mat_a->element[0][i * mid_dim + k],
                    mat_a->element[0][i * mid_dim + k + 1],
                    mat_a->element[0][i * mid_dim + k + 2],
                    mat_a->element[0][i * mid_dim + k + 3]
                );

            /*
                aik[0].val[1] = mat_a->element[1][i * mid_dim + k];
                aik[1].val[1] = mat_a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = mat_a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = mat_a->element[1][i * mid_dim + k + 3];
            */
                //aik[1] = _mm256_loadu_pd(&(mat_a->element[1][i * mid_dim + k]));
                aik[1] = _mm256_set_pd(
                    mat_a->element[1][i * mid_dim + k],
                    mat_a->element[1][i * mid_dim + k + 1],
                    mat_a->element[1][i * mid_dim + k + 2],
                    mat_a->element[1][i * mid_dim + k + 3]
                );

            /*
                aik[0].val[2] = mat_a->element[2][i * mid_dim + k];
                aik[1].val[2] = mat_a->element[2][i * mid_dim + k + 1];
                aik[2].val[2] = mat_a->element[2][i * mid_dim + k + 2];
                aik[3].val[2] = mat_a->element[2][i * mid_dim + k + 3];
            */
                //aik[2] = _mm256_loadu_pd(&(mat_a->element[2][i * mid_dim + k]));
                aik[2] = _mm256_set_pd(
                    mat_a->element[2][i * mid_dim + k],
                    mat_a->element[2][i * mid_dim + k + 1],
                    mat_a->element[2][i * mid_dim + k + 2],
                    mat_a->element[2][i * mid_dim + k + 3]
                );

            /*
                aik[0].val[3] = mat_a->element[3][i * mid_dim + k];
                aik[1].val[3] = mat_a->element[3][i * mid_dim + k + 1];
                aik[2].val[3] = mat_a->element[3][i * mid_dim + k + 2];
                aik[3].val[3] = mat_a->element[3][i * mid_dim + k + 3];
            */
                //aik[3] = _mm256_loadu_pd(&(mat_a->element[3][i * mid_dim + k]));
                aik[3] = _mm256_set_pd(
                    mat_a->element[3][i * mid_dim + k],
                    mat_a->element[3][i * mid_dim + k + 1],
                    mat_a->element[3][i * mid_dim + k + 2],
                    mat_a->element[3][i * mid_dim + k + 3]
                );

            /*
                bkj[0].val[0] = mat_b->element[0][k * col_dim + j];
                bkj[1].val[0] = mat_b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = mat_b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = mat_b->element[0][(k + 3) * col_dim + j];
           */
                bkj[0] = _mm256_set_pd(
                    mat_b->element[0][k * col_dim + j],
                    mat_b->element[0][(k + 1) * col_dim + j],
                    mat_b->element[0][(k + 2) * col_dim + j],
                    mat_b->element[0][(k + 3) * col_dim + j]
                );

            /*
                bkj[0].val[1] = mat_b->element[1][k * col_dim + j];
                bkj[1].val[1] = mat_b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = mat_b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = mat_b->element[1][(k + 3) * col_dim + j];
           */
                bkj[1] = _mm256_set_pd(
                    mat_b->element[1][k * col_dim + j],
                    mat_b->element[1][(k + 1) * col_dim + j],
                    mat_b->element[1][(k + 2) * col_dim + j],
                    mat_b->element[1][(k + 3) * col_dim + j]
                );

            /*
                bkj[0].val[2] = mat_b->element[2][k * col_dim + j];
                bkj[1].val[2] = mat_b->element[2][(k + 1) * col_dim + j];
                bkj[2].val[2] = mat_b->element[2][(k + 2) * col_dim + j];
                bkj[3].val[2] = mat_b->element[2][(k + 3) * col_dim + j];
           */
                bkj[2] = _mm256_set_pd(
                    mat_b->element[2][k * col_dim + j],
                    mat_b->element[2][(k + 1) * col_dim + j],
                    mat_b->element[2][(k + 2) * col_dim + j],
                    mat_b->element[2][(k + 3) * col_dim + j]
                );

            /*
                bkj[0].val[3] = mat_b->element[3][k * col_dim + j];
                bkj[1].val[3] = mat_b->element[3][(k + 1) * col_dim + j];
                bkj[2].val[3] = mat_b->element[3][(k + 2) * col_dim + j];
                bkj[3].val[3] = mat_b->element[3][(k + 3) * col_dim + j];
           */
                bkj[3] = _mm256_set_pd(
                    mat_b->element[3][k * col_dim + j],
                    mat_b->element[3][(k + 1) * col_dim + j],
                    mat_b->element[3][(k + 2) * col_dim + j],
                    mat_b->element[3][(k + 3) * col_dim + j]
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

            ret->element[0][i * col_dim + j] = cijval[0][0];
            ret->element[1][i * col_dim + j] = cijval[0][1];
            ret->element[2][i * col_dim + j] = cijval[0][2];
            ret->element[3][i * col_dim + j] = cijval[0][3];
        }
    }
}
#endif // defined(__AVX2__) __AVX2__
#endif //#if defined(__BNC_QDLINEAR_H__) && defined(_DEF_BNC_QDVECTOR)
#endif // ifndef __BNCAVX_QD_H