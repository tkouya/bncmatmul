// ------------------------
// ---------- QS ----------
// ------------------------
#ifndef QSSIZE
    #define QSSIZE 4
#endif // QSSIZE

// ret := 0
#if defined(__AVX2__)
static inline void _bncavx2_set0_qs(__m256 ret[QSSIZE])
{
    ret[0] = _mm256_setzero_ps();
    ret[1] = _mm256_setzero_ps();
    ret[2] = _mm256_setzero_ps();
    ret[3] = _mm256_setzero_ps();
}
#endif // __AVX2__

// ret := 0
#if defined(__AVX512F__)
static inline void _bncavx512_set0_qs(__m512 ret[QSSIZE])
{
    ret[0] = _mm512_setzero_ps();
    ret[1] = _mm512_setzero_ps();
    ret[2] = _mm512_setzero_ps();
    ret[3] = _mm512_setzero_ps();
}
#endif // __AVX512F__

#if defined(__AVX2__)
// ret := ret4[][avx_index]
static inline void _bncavx2_get_qs_m256_i(qsfloat *ret, __m256 ret4[QSSIZE], int avx_index)
{
    ret->val[0] = ret4[0][avx_index];
    ret->val[1] = ret4[1][avx_index];
    ret->val[2] = ret4[2][avx_index];
    ret->val[3] = ret4[3][avx_index];

    return;
}

// ret := mmval[0] + ... + mmval[7]
static void _bncavx2_rqs_sum256(float ret[QSSIZE], __m256 ret4[QSSIZE])
{
    qsfloat ret4_i[8]; /* __m256 holds 8 float lanes */
    int _l;

    for(_l = 0; _l < 8; _l++) _bncavx2_get_qs_m256_i(&ret4_i[_l], ret4, _l);

    rqs_set(ret, ret4_i[0].val);
    for(_l = 1; _l < 8; _l++) rqs_add(ret, ret, ret4_i[_l].val);
}

// abs
static inline void _bncavx2_rqs_abs(__m256 ret[QSSIZE], __m256 a[QSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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
static void _bncavx2_rqs_abssum256(float ret[QSSIZE], __m256 ret4[QSSIZE])
{
    qsfloat ret4_i[4];
    float tmp[QSSIZE];

    // ret4_i := ret4
    _bncavx2_get_qs_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qs_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qs_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qs_m256_i(&ret4_i[3], ret4, 3);

    rqs_abs(tmp, ret4_i[0].val);
    rqs_set(ret, tmp);

    rqs_abs(tmp, ret4_i[1].val);
    rqs_add(ret, ret, tmp);

    rqs_abs(tmp, ret4_i[2].val);
    rqs_add(ret, ret, tmp);

    rqs_abs(tmp, ret4_i[3].val);
    rqs_add(ret, ret, tmp);
}

// ret := max(|ret4[0]|, |ret4[1]|, |ret4[2]|, |ret4[3]|)
static void _bncavx2_rqs_absmax256(float ret[QSSIZE], __m256 ret4[QSSIZE])
{
    qsfloat ret4_i[4];
    float tmp[QSSIZE];

    // ret4_i := ret4
    _bncavx2_get_qs_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qs_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qs_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qs_m256_i(&ret4_i[3], ret4, 3);

    rqs_abs(tmp, ret4_i[0].val);
    rqs_set(ret, tmp); // ret := |ret4_i[0]|

    rqs_abs(tmp, ret4_i[1].val);
    if(rqs_cmp(ret, tmp) < 0) // if(ret < |ret4_i[1]|)
        rqs_set(ret, tmp);    //   ret := |ret4_i[1]|

    rqs_abs(tmp, ret4_i[2].val);
    if(rqs_cmp(ret, tmp) < 0) // if(ret < |ret4_i[2]|)
        rqs_set(ret, tmp);    //   ret := |ret4_i[2]|

    rqs_abs(tmp, ret4_i[3].val);
    if(rqs_cmp(ret, tmp) < 0) // if(ret < |ret4_i[3]|)
        rqs_set(ret, tmp);    //   ret := |ret4_i[3]|
}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx2_rqs_norm256(float ret[QSSIZE], __m256 ret4[QSSIZE])
{
    qsfloat ret4_i[4];
    float tmp[QSSIZE];

    // ret4_i := ret4
    _bncavx2_get_qs_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_qs_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_qs_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_qs_m256_i(&ret4_i[3], ret4, 3);

    rqs_mul(tmp, ret4_i[0].val, ret4_i[0].val);
    rqs_set(ret, tmp);

    rqs_mul(tmp, ret4_i[1].val, ret4_i[1].val);
    rqs_add(ret, ret, tmp);

    rqs_mul(tmp, ret4_i[2].val, ret4_i[2].val);
    rqs_add(ret, ret, tmp);

    rqs_mul(tmp, ret4_i[3].val, ret4_i[3].val);
    rqs_add(ret, ret, tmp);

    rqs_sqrt(tmp, ret);
    rqs_set(ret, tmp);
}
#endif // __AVX2__

#if defined(__AVX2__)
//inline void renorm(float *c0, float *c1, float *c2, float *c3)
static inline void _bncavx2_renormf(__m256 *c0, __m256 *c1, __m256 *c2, __m256 *c3)
{
// New codes on 2020-11-10 by T.Kouya
    float q[_BNC_S_WIDTH][QSSIZE];

    q[0][0] = (* c0)[0]; q[0][1] = (* c1)[0]; q[0][2] = (* c2)[0]; q[0][3] = (* c3)[0];
    q[1][0] = (* c0)[1]; q[1][1] = (* c1)[1]; q[1][2] = (* c2)[1]; q[1][3] = (* c3)[1];
    q[2][0] = (* c0)[2]; q[2][1] = (* c1)[2]; q[2][2] = (* c2)[2]; q[2][3] = (* c3)[2];
    q[3][0] = (* c0)[3]; q[3][1] = (* c1)[3]; q[3][2] = (* c2)[3]; q[3][3] = (* c3)[3];
    q[4][0] = (* c0)[4]; q[4][1] = (* c1)[4]; q[4][2] = (* c2)[4]; q[4][3] = (* c3)[4];
    q[5][0] = (* c0)[5]; q[5][1] = (* c1)[5]; q[5][2] = (* c2)[5]; q[5][3] = (* c3)[5];
    q[6][0] = (* c0)[6]; q[6][1] = (* c1)[6]; q[6][2] = (* c2)[6]; q[6][3] = (* c3)[6];
    q[7][0] = (* c0)[7]; q[7][1] = (* c1)[7]; q[7][2] = (* c2)[7]; q[7][3] = (* c3)[7];

    //renorm(&q0[0], &q0[1], &q0[2], &q0[3]);
    //renorm(&q1[0], &q1[1], &q1[2], &q1[3]);
    //renorm(&q2[0], &q2[1], &q2[2], &q2[3]);
    //renorm(&q3[0], &q3[1], &q3[2], &q3[3]);
    frenorm(&q[0][0], &q[0][1], &q[0][2], &q[0][3]);
    frenorm(&q[1][0], &q[1][1], &q[1][2], &q[1][3]);
    frenorm(&q[2][0], &q[2][1], &q[2][2], &q[2][3]);
    frenorm(&q[3][0], &q[3][1], &q[3][2], &q[3][3]);
    frenorm(&q[4][0], &q[4][1], &q[4][2], &q[4][3]);
    frenorm(&q[5][0], &q[5][1], &q[5][2], &q[5][3]);
    frenorm(&q[6][0], &q[6][1], &q[6][2], &q[6][3]);
    frenorm(&q[7][0], &q[7][1], &q[7][2], &q[7][3]);

    (* c0)[0] = q[0][0]; (* c1)[0] = q[0][1]; (* c2)[0] = q[0][2]; (* c3)[0] = q[0][3];
    (* c0)[1] = q[1][0]; (* c1)[1] = q[1][1]; (* c2)[1] = q[1][2]; (* c3)[1] = q[1][3];
    (* c0)[2] = q[2][0]; (* c1)[2] = q[2][1]; (* c2)[2] = q[2][2]; (* c3)[2] = q[2][3];
    (* c0)[3] = q[3][0]; (* c1)[3] = q[3][1]; (* c2)[3] = q[3][2]; (* c3)[3] = q[3][3];
    (* c0)[4] = q[4][0]; (* c1)[4] = q[4][1]; (* c2)[4] = q[4][2]; (* c3)[4] = q[4][3];
    (* c0)[5] = q[5][0]; (* c1)[5] = q[5][1]; (* c2)[5] = q[5][2]; (* c3)[5] = q[5][3];
    (* c0)[6] = q[6][0]; (* c1)[6] = q[6][1]; (* c2)[6] = q[6][2]; (* c3)[6] = q[6][3];
    (* c0)[7] = q[7][0]; (* c1)[7] = q[7][1]; (* c2)[7] = q[7][2]; (* c3)[7] = q[7][3];

#if 0
    int avx_index;
	float s0, s1, s2 = 0.0, s3 = 0.0;

//	if (QD_ISINF(c0)) return;
    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
    {
    	//if (isinf(*c0)) return;
        if (isinf((*c0)[avx_index])) continue; //return;

//        s0  = quick_two_sum((*c2)[avx_index], (*c3)[avx_index], &((*c3)[avx_index]));
        s0  = quick_two_sum((*c2)[avx_index], (*c3)[avx_index], (float *)(c3 + avx_index));
//        s0  = quick_two_sum((*c1)[avx_index],  s0, &((*c2)[avx_index]));
        s0  = quick_two_sum((*c1)[avx_index],  s0, (float *)(c2 + avx_index));
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

//inline void renorm4(float *c0, float *c1, float *c2, float *c3, float *c4)
static inline void _bncavx2_renorm4f(__m256 *c0, __m256 *c1, __m256 *c2, __m256 *c3, __m256 *c4)
{
    int avx_index;
	float s0, s1, s2 = 0.0, s3 = 0.0;

// New codes on 2020-11-10 by T.Kouya
    float q[8][5]; //, q1[5], q2[5], q3[5];

    q[0][0] = (* c0)[0]; q[0][1] = (* c1)[0]; q[0][2] = (* c2)[0]; q[0][3] = (* c3)[0]; q[0][4] = (* c4)[0];
    q[1][0] = (* c0)[1]; q[1][1] = (* c1)[1]; q[1][2] = (* c2)[1]; q[1][3] = (* c3)[1]; q[1][4] = (* c4)[1];
    q[2][0] = (* c0)[2]; q[2][1] = (* c1)[2]; q[2][2] = (* c2)[2]; q[2][3] = (* c3)[2]; q[2][4] = (* c4)[2];
    q[3][0] = (* c0)[3]; q[3][1] = (* c1)[3]; q[3][2] = (* c2)[3]; q[3][3] = (* c3)[3]; q[3][4] = (* c4)[3];
    q[4][0] = (* c0)[4]; q[4][1] = (* c1)[4]; q[4][2] = (* c2)[4]; q[4][3] = (* c3)[4]; q[4][4] = (* c4)[4];
    q[5][0] = (* c0)[5]; q[5][1] = (* c1)[5]; q[5][2] = (* c2)[5]; q[5][3] = (* c3)[5]; q[5][4] = (* c4)[5];
    q[6][0] = (* c0)[6]; q[6][1] = (* c1)[6]; q[6][2] = (* c2)[6]; q[6][3] = (* c3)[6]; q[6][4] = (* c4)[6];
    q[7][0] = (* c0)[7]; q[7][1] = (* c1)[7]; q[7][2] = (* c2)[7]; q[7][3] = (* c3)[7]; q[7][4] = (* c4)[7];

    frenorm4(&q[0][0], &q[0][1], &q[0][2], &q[0][3], &q[0][4]);
    frenorm4(&q[1][0], &q[1][1], &q[1][2], &q[1][3], &q[1][4]);
    frenorm4(&q[2][0], &q[2][1], &q[2][2], &q[2][3], &q[2][4]);
    frenorm4(&q[3][0], &q[3][1], &q[3][2], &q[3][3], &q[3][4]);
    frenorm4(&q[4][0], &q[4][1], &q[4][2], &q[4][3], &q[4][4]);
    frenorm4(&q[5][0], &q[5][1], &q[5][2], &q[5][3], &q[5][4]);
    frenorm4(&q[6][0], &q[6][1], &q[6][2], &q[6][3], &q[6][4]);
    frenorm4(&q[7][0], &q[7][1], &q[7][2], &q[7][3], &q[7][4]);

    (* c0)[0] = q[0][0]; (* c1)[0] = q[0][1]; (* c2)[0] = q[0][2]; (* c3)[0] = q[0][3]; (* c4)[0] = q[0][4];
    (* c0)[1] = q[1][0]; (* c1)[1] = q[1][1]; (* c2)[1] = q[1][2]; (* c3)[1] = q[1][3]; (* c4)[1] = q[1][4];
    (* c0)[2] = q[2][0]; (* c1)[2] = q[2][1]; (* c2)[2] = q[2][2]; (* c3)[2] = q[2][3]; (* c4)[2] = q[2][4];
    (* c0)[3] = q[3][0]; (* c1)[3] = q[3][1]; (* c2)[3] = q[3][2]; (* c3)[3] = q[3][3]; (* c4)[3] = q[3][4];
    (* c0)[4] = q[4][0]; (* c1)[4] = q[4][1]; (* c2)[4] = q[4][2]; (* c3)[4] = q[4][3]; (* c4)[4] = q[4][4];
    (* c0)[5] = q[5][0]; (* c1)[5] = q[5][1]; (* c2)[5] = q[5][2]; (* c3)[5] = q[5][3]; (* c4)[5] = q[5][4];
    (* c0)[6] = q[6][0]; (* c1)[6] = q[6][1]; (* c2)[6] = q[6][2]; (* c3)[6] = q[6][3]; (* c4)[6] = q[6][4];
    (* c0)[7] = q[7][0]; (* c1)[7] = q[7][1]; (* c2)[7] = q[7][2]; (* c3)[7] = q[7][3]; (* c4)[7] = q[7][4];

#if 0
    float in_c0[4], in_c1[4], in_c2[4], in_c3[4], in_c4[4];
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
    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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


    *c0 = _mm256_set_ps(in_c0[0], in_c0[1], in_c0[2], in_c0[3]);
    *c1 = _mm256_set_ps(in_c1[0], in_c1[1], in_c1[2], in_c1[3]);
    *c2 = _mm256_set_ps(in_c2[0], in_c2[1], in_c2[2], in_c2[3]);
    *c3 = _mm256_set_ps(in_c3[0], in_c3[1], in_c3[2], in_c3[3]);
    *c4 = _mm256_set_ps(in_c4[0], in_c4[1], in_c4[2], in_c4[3]);

/*
    *c0 = _mm256_set_ps(in_c0[3], in_c0[2], in_c0[1], in_c0[0]);
    *c1 = _mm256_set_ps(in_c1[3], in_c1[2], in_c1[1], in_c1[0]);
    *c2 = _mm256_set_ps(in_c2[3], in_c2[2], in_c2[1], in_c2[0]);
    *c3 = _mm256_set_ps(in_c3[3], in_c3[2], in_c3[1], in_c3[0]);
    *c4 = _mm256_set_ps(in_c4[3], in_c4[2], in_c4[1], in_c4[0]);
*/
#endif // 0
#if 0 //
//	if (QD_ISINF(c0)) return;
    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
    {
        if (isinf((*c0)[avx_index])) continue; //return;

//        s0  = quick_two_sum((*c3)[avx_index], (*c4)[avx_index], &((*c4)[avx_index]));
//        s0  = quick_two_sum((*c2)[avx_index], s0 , &((*c3)[avx_index]));
//        s0  = quick_two_sum((*c1)[avx_index], s0 , &((*c2)[avx_index]));
        s0  = quick_two_sum((*c3)[avx_index], (*c4)[avx_index], (float *)&((*c4)[avx_index]));
        s0  = quick_two_sum((*c2)[avx_index], s0 , (float *)&((*c3)[avx_index]));
        s0  = quick_two_sum((*c1)[avx_index], s0 , (float *)&((*c2)[avx_index]));
        (*c0)[avx_index] = quick_two_sum((*c0)[avx_index], s0 , (float *)&((*c1)[avx_index]));

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
//inline void three_sum(float *a, float *b, float *c)
static inline void _bncavx2_three_sumf(__m256 *a, __m256 *b, __m256 *c)
{
//	float t1, t2, t3;
    __m256 t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = two_sum(t2, t3, c);
	t1 = _bncavx2_ftwo_sum(*a, *b, &t2);
	*a = _bncavx2_ftwo_sum(*c, t1, &t3);
	*b = _bncavx2_ftwo_sum(t2, t3, c);

}

//inline void three_sum2(float *a, float *b, float *c)
static inline void _bncavx2_three_sum2f(__m256 *a, __m256 *b, __m256 *c)
{
//	float t1, t2, t3;
    __m256 t1, t2, t3;

//	t1 = two_sum(*a, *b, &t2);
//	*a = two_sum(*c, t1, &t3);
//	*b = t2 + t3;
	t1 = _bncavx2_ftwo_sum(*a, *b, &t2);
	*a = _bncavx2_ftwo_sum(*c, t1, &t3);
	*b = _mm256_add_ps(t2, t3);

}
static inline void _bncavx2_rqs_add(__m256 ret[QSSIZE], __m256 a[QSSIZE], __m256 b[QSSIZE])
{
#if 0
    float in_ret[4][QSSIZE], in_a[4][QSSIZE], in_b[4][QSSIZE];

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

    rqs_add(in_ret[0], in_a[0], in_b[0]);
    rqs_add(in_ret[1], in_a[1], in_b[1]);
    rqs_add(in_ret[2], in_a[2], in_b[2]);
    rqs_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
    // c_qs_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	float s0, s1, s2, s3;
//	float t0, t1, t2, t3;
	__m256 s0, s1, s2, s3;
	__m256 t0, t1, t2, t3;

//	float v0, v1, v2, v3;
//	float u0, u1, u2, u3;
//	float w0, w1, w2, w3;
	__m256 v0, v1, v2, v3;
	__m256 u0, u1, u2, u3;
	__m256 w0, w1, w2, w3;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm256_add_ps(a[0], b[0]);
	s1 = _mm256_add_ps(a[1], b[1]);
	s2 = _mm256_add_ps(a[2], b[2]);
	s3 = _mm256_add_ps(a[3], b[3]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm256_sub_ps(s0, a[0]);
	v1 = _mm256_sub_ps(s1, a[1]);
	v2 = _mm256_sub_ps(s2, a[2]);
	v3 = _mm256_sub_ps(s3, a[3]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm256_sub_ps(s0, v0);
	u1 = _mm256_sub_ps(s1, v1);
	u2 = _mm256_sub_ps(s2, v2);
	u3 = _mm256_sub_ps(s3, v3);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm256_sub_ps(a[0], u0);
	w1 = _mm256_sub_ps(a[1], u1);
	w2 = _mm256_sub_ps(a[2], u2);
	w3 = _mm256_sub_ps(a[3], u3);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm256_sub_ps(b[0], v0);
	u1 = _mm256_sub_ps(b[1], v1);
	u2 = _mm256_sub_ps(b[2], v2);
	u3 = _mm256_sub_ps(b[3], v3);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm256_add_ps(w0, u0);
	t1 = _mm256_add_ps(w1, u1);
	t2 = _mm256_add_ps(w2, u2);
	t3 = _mm256_add_ps(w3, u3);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx2_ftwo_sum(s1, t0, &t0);
	_bncavx2_three_sumf(&s2, &t0, &t1);
	_bncavx2_three_sum2f(&s3, &t0, &t2);
	t0 = _mm256_add_ps(_mm256_add_ps(t0, t1), t3);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx2_renorm4f(&s0, &s1, &s2, &s3, &t0);
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
}

// rqs_add -> rts_add
static inline void _bncavx2_rts_addq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
#if 0
    float in_ret[4][QSSIZE], in_a[4][QSSIZE], in_b[4][QSSIZE];

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

    rqs_add(in_ret[0], in_a[0], in_b[0]);
    rqs_add(in_ret[1], in_a[1], in_b[1]);
    rqs_add(in_ret[2], in_a[2], in_b[2]);
    rqs_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
    // c_qs_add_sloppy
	/* Same as above, but addition re-organized to minimize
	   data dependency ... unfortunately some compilers are
	   not very smart to do this automatically */
//	float s0, s1, s2, s3;
//	float t0, t1, t2, t3;
	__m256 s0, s1, s2;
	__m256 t0, t1, t2;

//	float v0, v1, v2;
//	float u0, u1, u2;
//	float w0, w1, w2;
	__m256 v0, v1, v2;
	__m256 u0, u1, u2;
	__m256 w0, w1, w2;

/*
	s0 = a[0] + b[0];
	s1 = a[1] + b[1];
	s2 = a[2] + b[2];
	s3 = a[3] + b[3];
*/
	s0 = _mm256_add_ps(a[0], b[0]);
	s1 = _mm256_add_ps(a[1], b[1]);
	s2 = _mm256_add_ps(a[2], b[2]);

/*
	v0 = s0 - a[0];
	v1 = s1 - a[1];
	v2 = s2 - a[2];
	v3 = s3 - a[3];
*/
	v0 = _mm256_sub_ps(s0, a[0]);
	v1 = _mm256_sub_ps(s1, a[1]);
	v2 = _mm256_sub_ps(s2, a[2]);
/*
	u0 = s0 - v0;
	u1 = s1 - v1;
	u2 = s2 - v2;
	u3 = s3 - v3;
*/
	u0 = _mm256_sub_ps(s0, v0);
	u1 = _mm256_sub_ps(s1, v1);
	u2 = _mm256_sub_ps(s2, v2);

/*
	w0 = a[0] - u0;
	w1 = a[1] - u1;
	w2 = a[2] - u2;
	w3 = a[3] - u3;
*/
	w0 = _mm256_sub_ps(a[0], u0);
	w1 = _mm256_sub_ps(a[1], u1);
	w2 = _mm256_sub_ps(a[2], u2);

/*
	u0 = b[0] - v0;
	u1 = b[1] - v1;
	u2 = b[2] - v2;
	u3 = b[3] - v3;
*/
	u0 = _mm256_sub_ps(b[0], v0);
	u1 = _mm256_sub_ps(b[1], v1);
	u2 = _mm256_sub_ps(b[2], v2);

/*
	t0 = w0 + u0;
	t1 = w1 + u1;
	t2 = w2 + u2;
	t3 = w3 + u3;
*/
	t0 = _mm256_add_ps(w0, u0);
	t1 = _mm256_add_ps(w1, u1);
	t2 = _mm256_add_ps(w2, u2);

/*	s1 = two_sum(s1, t0, &t0);
	three_sum(&s2, &t0, &t1);
	three_sum2(&s3, &t0, &t2);
	t0 = t0 + t1 + t3;
*/
	s1 = _bncavx2_ftwo_sum(s1, t0, &t0);
	_bncavx2_three_sumf(&s2, &t0, &t1);
//	_bncavx2_three_sum2f(&s2, &t0, &t1);
//	t0 = _mm256_add_ps(_mm256_add_ps(t0, t1), t2);
	t0 = _mm256_add_ps(t0, t1);

	/* renormalize */
//	renorm4(&s0, &s1, &s2, &s3, &t0);
	_bncavx2_renormf(&s0, &s1, &s2, &t0);
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
static inline void _bncavx2_rts_mulq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
//#if 0
    // c_qs_mul_sloppy
/*
	float p0, p1, p2, p3, p4, p5;
	float q0, q1, q2, q3, q4, q5;
	float t0, t1;
	float s0, s1, s2;
*/
    __m256 p0, p1, p2, p3, p4, p5;
    __m256 q0, q1, q2, q3, q4, q5;
    __m256 t0, t1;
    __m256 s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx2_ftwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx2_ftwo_prod(a[0], b[1], &q1);
	p2 = _bncavx2_ftwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx2_ftwo_prod(a[0], b[2], &q3);
	p4 = _bncavx2_ftwo_prod(a[1], b[1], &q4);
	p5 = _bncavx2_ftwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx2_three_sumf(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx2_three_sum2f(&p2, &q1, &q2);
	_bncavx2_three_sum2f(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx2_ftwo_sum(p2, p3, &t0);
	s1 = _bncavx2_ftwo_sum(q1, p4, &t1);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx2_ftwo_sum(s1, t0, &t0);

	/* O(eps^3) order terms */
	//s1 += a[1]*b[2] + a[2]*b[1] + q0 + q3 + q4;
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[1], b[2]));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[2], b[1]));
    s1 = _mm256_add_ps(s1, q0);
    s1 = _mm256_add_ps(s1, q3);
    s1 = _mm256_add_ps(s1, q4);

	//renorm(p0, p1, s0, s1, s2);
//	renorm(&p0, &p1, &s0, &s1);
	_bncavx2_renormf(&p0, &p1, &s0, &s1);
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

// mul
static inline void _bncavx2_rqs_mul(__m256 ret[QSSIZE], __m256 a[QSSIZE], __m256 b[QSSIZE])
{
#if 0
    float in_ret[4][QSSIZE], in_a[4][QSSIZE], in_b[4][QSSIZE];

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

    rqs_mul(in_ret[0], in_a[0], in_b[0]);
    rqs_mul(in_ret[1], in_a[1], in_b[1]);
    rqs_mul(in_ret[2], in_a[2], in_b[2]);
    rqs_mul(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2]; ret[3][0] = in_ret[0][3];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2]; ret[3][1] = in_ret[1][3];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2]; ret[3][2] = in_ret[2][3];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2]; ret[3][3] = in_ret[3][3];
#endif // 0
//#if 0
    // c_qs_mul_sloppy
/*
	float p0, p1, p2, p3, p4, p5;
	float q0, q1, q2, q3, q4, q5;
	float t0, t1;
	float s0, s1, s2;
*/
    __m256 p0, p1, p2, p3, p4, p5;
    __m256 q0, q1, q2, q3, q4, q5;
    __m256 t0, t1;
    __m256 s0, s1, s2;

//	p0 = two_prod(a[0], b[0], &q0);
	p0 = _bncavx2_ftwo_prod(a[0], b[0], &q0);

//	p1 = two_prod(a[0], b[1], &q1);
//	p2 = two_prod(a[1], b[0], &q2);
	p1 = _bncavx2_ftwo_prod(a[0], b[1], &q1);
	p2 = _bncavx2_ftwo_prod(a[1], b[0], &q2);

//	p3 = two_prod(a[0], b[2], &q3);
//	p4 = two_prod(a[1], b[1], &q4);
//	p5 = two_prod(a[2], b[0], &q5);
	p3 = _bncavx2_ftwo_prod(a[0], b[2], &q3);
	p4 = _bncavx2_ftwo_prod(a[1], b[1], &q4);
	p5 = _bncavx2_ftwo_prod(a[2], b[0], &q5);

	/* Start Accumulation */
//	three_sum(&p1, &p2, &q0);
	_bncavx2_three_sumf(&p1, &p2, &q0);

	/* Six-Three Sum  of p2, q1, q2, p3, p4, p5. */
	//three_sum(&p2, &q1, &q2);
	//three_sum(&p3, &p4, &p5);
	_bncavx2_three_sumf(&p2, &q1, &q2);
	_bncavx2_three_sumf(&p3, &p4, &p5);

	/* compute (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5). */
//	s0 = two_sum(p2, p3, &t0);
//	s1 = two_sum(q1, p4, &t1);
	s0 = _bncavx2_ftwo_sum(p2, p3, &t0);
	s1 = _bncavx2_ftwo_sum(q1, p4, &t1);
//	s2 = q2 + p5;
	s2 = _mm256_add_ps(q2, p5);
//	s1 = two_sum(s1, t0, &t0);
	s1 = _bncavx2_ftwo_sum(s1, t0, &t0);
//	s2 += (t0 + t1);
	s2 = _mm256_add_ps(s2, _mm256_add_ps(t0, t1));

	/* O(eps^3) order terms */
	//s1 += a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + q0 + q3 + q4 + q5;
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[0], b[3]));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[1], b[2]));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[2], b[1]));
    s1 = _mm256_add_ps(s1, _mm256_mul_ps(a[3], b[0]));
    s1 = _mm256_add_ps(s1, q0);
    s1 = _mm256_add_ps(s1, q3);
    s1 = _mm256_add_ps(s1, q4);
    s1 = _mm256_add_ps(s1, q5);

	//renorm(p0, p1, s0, s1, s2);
//	renorm4(&p0, &p1, &s0, &s1, &s2);
	_bncavx2_renorm4f(&p0, &p1, &s0, &s1, &s2);
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

}
//#endif //defined(__AVX2__)

// c[3] := -a[3]
//static inline void c_qs_neg(const float *a, float *c)
static inline void _bncavx2_rqs_neg(__m256 c[QSSIZE], __m256 a[QSSIZE])
{
    __m256 zero4;

    zero4 = _mm256_setzero_ps();

	//c[0] = -a[0];
    c[0] = _mm256_sub_ps(zero4, a[0]);
	//c[1] = -a[1];
    c[1] = _mm256_sub_ps(zero4, a[1]);
    //c[2] = -a[2];
    c[2] = _mm256_sub_ps(zero4, a[2]);    
    //c[3] = -a[3];
    c[3] = _mm256_sub_ps(zero4, a[3]);    

}
/* sub */
// c := a - b
//static inline void c_qs_sub(const float *a, const float *b, float *c)
static inline void _bncavx2_rqs_sub(__m256 c[QSSIZE], __m256 a[QSSIZE], __m256 b[QSSIZE])
{
	__m256 mb[QSSIZE];

	// a + (-b)
	//c_qs_neg(b, mb);
    _bncavx2_rqs_neg(mb, b);
	//c_qs_add(a, mb, c);
    _bncavx2_rqs_add(c, a, mb);

#if 0
  qd_real cc;
  cc = qd_real(a) - qd_real(b);
  TO_DOUBLE_PTR(cc, c);
#endif // 0
}
// c := a * (float)b
//static inline void c_qs_mul_qs_d(const float *a, float b, float *c)
static inline void _bncavx2_rqs_mul_f(__m256 c[QSSIZE], const __m256 a[QSSIZE], __m256 b)
{
	//float p0, p1, p2, p3;
	//float q0, q1, q2;
	//float s0, s1, s2, s3, s4;
	__m256 p0, p1, p2, p3;
	__m256 q0, q1, q2;
	__m256 s0, s1, s2, s3, s4;

	//p0 = two_prod(a[0], b, &q0);
	//p1 = two_prod(a[1], b, &q1);
	//p2 = two_prod(a[2], b, &q2);
	//p3 = a[3] * b;
	p0 = _bncavx2_ftwo_prod(a[0], b, &q0);
	p1 = _bncavx2_ftwo_prod(a[1], b, &q1);
	p2 = _bncavx2_ftwo_prod(a[2], b, &q2);
	p3 = _mm256_mul_ps(a[3], b);

	s0 = p0;

	//s1 = two_sum(q0, p1, &s2);
    s1 = _bncavx2_ftwo_sum(q0, p1, &s2);

	//three_sum(&s2, &q1, &p2);
	_bncavx2_three_sumf(&s2, &q1, &p2);

	//three_sum2(&q1, &q2, &p3);
	_bncavx2_three_sum2f(&q1, &q2, &p3);
	s3 = q1;

	//s4 = q2 + p2;
    s4 = _mm256_add_ps(q2, p2);

	//renorm4(&s0, &s1, &s2, &s3, &s4);
	_bncavx2_renorm4f(&s0, &s1, &s2, &s3, &s4);
    //	return qd_real(s0, s1, s2, s3);
	c[0] = s0;
	c[1] = s1;
	c[2] = s2;
	c[3] = s3;
}

// c := a / b
//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b)
static inline void _bncavx2_rts_divq(__m256 c[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
	//float q0, q1, q2, q3;
	//float r[QSSIZE], tmp[QSSIZE];
	__m256 q0, q1, q2, q3;
	__m256 r[TSSIZE], tmp[TSSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm256_div_ps(a[0], b[0]);

	//r = a - (b * q0);
	//c_qs_mul_qs_d(b, q0, tmp);
	//c_qs_sub(a, tmp, r);
	_bncavx2_rts_mul_f(tmp, q0, b);
	//_bncavx2_rts_sub(r, a, tmp);
	_bncavx2_rts_subq(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm256_div_ps(r[0], b[0]);

	//r -= (b * q1);
	_bncavx2_rts_mul_f(tmp, q1, b);
	//_bncavx2_rts_sub(r, tmp, r);
	_bncavx2_rts_subq(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm256_div_ps(r[0], b[0]);
	//r -= (b * q2);
	//c_qs_mul_qs_d(b, q2, tmp);
	//c_qs_selfsub(tmp, r);
	_bncavx2_rts_mul_f(tmp, q2, b);
	//_bncavx2_rts_sub(r, tmp, r);
	_bncavx2_rts_subq(r, tmp, r);
	//c_qs_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm256_div_ps(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx2_renormf(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	//c[3] = q3;
}

/* div */
// c := a / b
//static inline void c_qs_div_sloppy(const float *a, const float *b, float *c)
//qd_real qd_real::sloppy_div(const qd_real &a, const qd_real &b)
static inline void _bncavx2_rqs_div(__m256 c[QSSIZE], __m256 a[QSSIZE], __m256 b[QSSIZE])
{
	//float q0, q1, q2, q3;
	//float r[QSSIZE], tmp[QSSIZE];
	__m256 q0, q1, q2, q3;
	__m256 r[QSSIZE], tmp[QSSIZE];
	//qd_real r;

	//q0 = a[0] / b[0];
    q0 = _mm256_div_ps(a[0], b[0]);

	//r = a - (b * q0);
	//c_qs_mul_qs_d(b, q0, tmp);
	//c_qs_sub(a, tmp, r);
	_bncavx2_rqs_mul_f(tmp, b, q0);
	_bncavx2_rqs_sub(r, a, tmp);

	//q1 = r[0] / b[0];
    q1 = _mm256_div_ps(r[0], b[0]);

	//r -= (b * q1);
	//c_qs_mul_qs_d(b, q1, tmp);
	//c_qs_selfsub(tmp, r);
	_bncavx2_rqs_mul_f(tmp, b, q1);
	_bncavx2_rqs_sub(r, tmp, r);
	//c_qs_sub(r, tmp, r);

	//q2 = r[0] / b[0];
    q2 = _mm256_div_ps(r[0], b[0]);
	//r -= (b * q2);
	//c_qs_mul_qs_d(b, q2, tmp);
	//c_qs_selfsub(tmp, r);
	_bncavx2_rqs_mul_f(tmp, b, q2);
	_bncavx2_rqs_sub(r, tmp, r);
	//c_qs_sub(r, tmp, r);

	//q3 = r[0] / b[0];
    q3 = _mm256_div_ps(r[0], b[0]);

	//renorm(&q0, &q1, &q2, &q3);
    _bncavx2_renormf(&q0, &q1, &q2, &q3);

	//return qd_real(q0, q1, q2, q3);
	c[0] = q0;
	c[1] = q1;
	c[2] = q2;
	c[3] = q3;
}
#endif //defined(__AVX2__)

//#ifdef __BNC_QSLINEAR_H__
//#include "qdv_addmul.c"
//#endif //__BNC_QSLINEAR_H__

#ifdef __BNC_DSLINEAR_H__
#ifdef USE_MPFR
// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
static void set_test_dsmatrix(dsfloat mat[], int sqrt_seed, int row_dim, int col_dim)
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

    for(i = 0; i < row_dim; i++)
    {
        //printf("%5d: ", i);
        for(j = 0; j < col_dim; j++)
        {
            rds_set_ui(mat[i * col_dim + j].val, i + j + 1);
            rds_mul(mat[i * col_dim + j].val, mat[i * col_dim + j].val, dssqrt.val);
            //rdd_out_str(mat[i * col_dim + j].val); printf(" ");
        }
        //printf("\n");
    }
}
#endif // USE_MPFR

// dSmatmul
static void dsmatmul(dsfloat ret[], dsfloat mat_a[], dsfloat mat_b[], int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    dsfloat tmp_add, tmp_mul;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rds_set_ui(tmp_add.val, 0UL);
            for(k = 0; k < mid_dim; k++)
            {
                rds_mul(tmp_mul.val, mat_a[i * mid_dim + k].val, mat_b[k * col_dim + j].val);
                rds_add(tmp_add.val, tmp_add.val, tmp_mul.val);
            }
            rds_set(ret[i * col_dim + j].val, tmp_add.val);
        }
    }
}

#endif //__BNC_DSLINEAR_H__

#if 0
// qdrel_diff
static inline qsfloat qsrel_diff(qsfloat a, qsfloat b)
{
    qsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rqs_sub(rel_diff.val, a.val, b.val);
    rqs_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rqs_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rqs_abs(abs_a.val, a.val);
        rqs_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
#endif // 0

#if defined(__BNC_QSLINEAR_H__) && defined(_DEF_BNC_QSVECTOR)
// qdmatmul_qsvec
static void qsmatmul_qsvec(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qsfloat tmp_mul, aik, bkj, cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqs_set_ui(cij.val, 0UL);
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

                rqs_mul(tmp_mul.val, aik.val, bkj.val);
                rqs_add(cij.val, cij.val, tmp_mul.val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];        }
    }
}

// qdmatmul_qsvec_ur4
static void qsmatmul_qsvec_ur4(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    qsfloat tmp_mul[4], aik[4], bkj[4], cij;
    //float cijval[4][QSSIZE];
    //__m256 tmp_mul[QSSIZE], aik[QSSIZE], bkj[QSSIZE], cij[QSSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rqs_set_ui(cij.val, 0UL);
            //cij[0] = _mm256_setzero_ps();
            //cij[1] = _mm256_setzero_ps();
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

                rqs_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqs_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqs_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqs_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);

                rqs_add(cij.val, cij.val, tmp_mul[0].val);
                rqs_add(cij.val, cij.val, tmp_mul[1].val);
                rqs_add(cij.val, cij.val, tmp_mul[2].val);
                rqs_add(cij.val, cij.val, tmp_mul[3].val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
            ret->element[2][i * col_dim + j] = cij.val[2];
            ret->element[3][i * col_dim + j] = cij.val[3];        }
    }
}

#if defined(__AVX2__)
// qdmatmul_qsvec_avx2
static void qsmatmul_qsvec_avx2(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
//    qsfloat tmp_mul[4], aik[4], bkj[4], cij;
    float cijval[8][QSSIZE];
    __m256 cij[QSSIZE], aik[QSSIZE], bkj[QSSIZE], tmp_mul[QSSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            //rqs_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_ps();
            cij[1] = _mm256_setzero_ps();
            cij[2] = _mm256_setzero_ps();
            cij[3] = _mm256_setzero_ps();

            for(k = 0; k < mid_dim; k += 4)
            {
            /*
                aik[0].val[0] = mat_a->element[0][i * mid_dim + k];
                aik[1].val[0] = mat_a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = mat_a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = mat_a->element[0][i * mid_dim + k + 3];
            */
                //aik[0] = _mm256_loadu_ps(&(mat_a->element[0][i * mid_dim + k]));
                aik[0] = _mm256_set_ps(
                    mat_a->element[0][i * mid_dim + k],
                    mat_a->element[0][i * mid_dim + k + 1],
                    mat_a->element[0][i * mid_dim + k + 2],
                    mat_a->element[0][i * mid_dim + k + 3],
                    mat_a->element[0][i * mid_dim + k + 4],
                    mat_a->element[0][i * mid_dim + k + 5],
                    mat_a->element[0][i * mid_dim + k + 6],
                    mat_a->element[0][i * mid_dim + k + 7]
                );

            /*
                aik[0].val[1] = mat_a->element[1][i * mid_dim + k];
                aik[1].val[1] = mat_a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = mat_a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = mat_a->element[1][i * mid_dim + k + 3];
            */
                //aik[1] = _mm256_loadu_ps(&(mat_a->element[1][i * mid_dim + k]));
                aik[1] = _mm256_set_ps(
                    mat_a->element[1][i * mid_dim + k],
                    mat_a->element[1][i * mid_dim + k + 1],
                    mat_a->element[1][i * mid_dim + k + 2],
                    mat_a->element[1][i * mid_dim + k + 3],
                    mat_a->element[1][i * mid_dim + k + 4],
                    mat_a->element[1][i * mid_dim + k + 5],
                    mat_a->element[1][i * mid_dim + k + 6],
                    mat_a->element[1][i * mid_dim + k + 7]
                );

            /*
                aik[0].val[2] = mat_a->element[2][i * mid_dim + k];
                aik[1].val[2] = mat_a->element[2][i * mid_dim + k + 1];
                aik[2].val[2] = mat_a->element[2][i * mid_dim + k + 2];
                aik[3].val[2] = mat_a->element[2][i * mid_dim + k + 3];
            */
                //aik[2] = _mm256_loadu_ps(&(mat_a->element[2][i * mid_dim + k]));
                aik[2] = _mm256_set_ps(
                    mat_a->element[2][i * mid_dim + k],
                    mat_a->element[2][i * mid_dim + k + 1],
                    mat_a->element[2][i * mid_dim + k + 2],
                    mat_a->element[2][i * mid_dim + k + 3],
                    mat_a->element[2][i * mid_dim + k + 4],
                    mat_a->element[2][i * mid_dim + k + 5],
                    mat_a->element[2][i * mid_dim + k + 6],
                    mat_a->element[2][i * mid_dim + k + 7]
                );

            /*
                aik[0].val[3] = mat_a->element[3][i * mid_dim + k];
                aik[1].val[3] = mat_a->element[3][i * mid_dim + k + 1];
                aik[2].val[3] = mat_a->element[3][i * mid_dim + k + 2];
                aik[3].val[3] = mat_a->element[3][i * mid_dim + k + 3];
            */
                //aik[3] = _mm256_loadu_ps(&(mat_a->element[3][i * mid_dim + k]));
                aik[3] = _mm256_set_ps(
                    mat_a->element[3][i * mid_dim + k],
                    mat_a->element[3][i * mid_dim + k + 1],
                    mat_a->element[3][i * mid_dim + k + 2],
                    mat_a->element[3][i * mid_dim + k + 3],
                    mat_a->element[3][i * mid_dim + k + 4],
                    mat_a->element[3][i * mid_dim + k + 5],
                    mat_a->element[3][i * mid_dim + k + 6],
                    mat_a->element[3][i * mid_dim + k + 7]
                );

            /*
                bkj[0].val[0] = mat_b->element[0][k * col_dim + j];
                bkj[1].val[0] = mat_b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = mat_b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = mat_b->element[0][(k + 3) * col_dim + j];
           */
                bkj[0] = _mm256_set_ps(
                    mat_b->element[0][k * col_dim + j],
                    mat_b->element[0][(k + 1) * col_dim + j],
                    mat_b->element[0][(k + 2) * col_dim + j],
                    mat_b->element[0][(k + 3) * col_dim + j],
                    mat_b->element[0][(k + 4) * col_dim + j],
                    mat_b->element[0][(k + 5) * col_dim + j],
                    mat_b->element[0][(k + 6) * col_dim + j],
                    mat_b->element[0][(k + 7) * col_dim + j]
                );

            /*
                bkj[0].val[1] = mat_b->element[1][k * col_dim + j];
                bkj[1].val[1] = mat_b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = mat_b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = mat_b->element[1][(k + 3) * col_dim + j];
           */
                bkj[1] = _mm256_set_ps(
                    mat_b->element[1][k * col_dim + j],
                    mat_b->element[1][(k + 1) * col_dim + j],
                    mat_b->element[1][(k + 2) * col_dim + j],
                    mat_b->element[1][(k + 3) * col_dim + j],
                    mat_b->element[1][(k + 4) * col_dim + j],
                    mat_b->element[1][(k + 5) * col_dim + j],
                    mat_b->element[1][(k + 6) * col_dim + j],
                    mat_b->element[1][(k + 7) * col_dim + j]
                );

            /*
                bkj[0].val[2] = mat_b->element[2][k * col_dim + j];
                bkj[1].val[2] = mat_b->element[2][(k + 1) * col_dim + j];
                bkj[2].val[2] = mat_b->element[2][(k + 2) * col_dim + j];
                bkj[3].val[2] = mat_b->element[2][(k + 3) * col_dim + j];
           */
                bkj[2] = _mm256_set_ps(
                    mat_b->element[2][k * col_dim + j],
                    mat_b->element[2][(k + 1) * col_dim + j],
                    mat_b->element[2][(k + 2) * col_dim + j],
                    mat_b->element[2][(k + 3) * col_dim + j],
                    mat_b->element[2][(k + 4) * col_dim + j],
                    mat_b->element[2][(k + 5) * col_dim + j],
                    mat_b->element[2][(k + 6) * col_dim + j],
                    mat_b->element[2][(k + 7) * col_dim + j]
                );

            /*
                bkj[0].val[3] = mat_b->element[3][k * col_dim + j];
                bkj[1].val[3] = mat_b->element[3][(k + 1) * col_dim + j];
                bkj[2].val[3] = mat_b->element[3][(k + 2) * col_dim + j];
                bkj[3].val[3] = mat_b->element[3][(k + 3) * col_dim + j];
           */
                bkj[3] = _mm256_set_ps(
                    mat_b->element[3][k * col_dim + j],
                    mat_b->element[3][(k + 1) * col_dim + j],
                    mat_b->element[3][(k + 2) * col_dim + j],
                    mat_b->element[3][(k + 3) * col_dim + j],
                    mat_b->element[3][(k + 4) * col_dim + j],
                    mat_b->element[3][(k + 5) * col_dim + j],
                    mat_b->element[3][(k + 6) * col_dim + j],
                    mat_b->element[3][(k + 7) * col_dim + j]
                );

            /*
                rqs_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqs_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqs_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqs_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rqs_mul(tmp_mul, aik, bkj);

            /*
                rqs_add(cij.val, cij.val, tmp_mul[0].val);
                rqs_add(cij.val, cij.val, tmp_mul[1].val);
                rqs_add(cij.val, cij.val, tmp_mul[2].val);
                rqs_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rqs_add(cij, cij, tmp_mul);

            }
            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0]; cijval[0][3] = cij[3][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1]; cijval[1][3] = cij[3][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2]; cijval[2][3] = cij[3][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3]; cijval[3][3] = cij[3][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4]; cijval[4][3] = cij[3][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5]; cijval[5][3] = cij[3][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6]; cijval[6][3] = cij[3][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7]; cijval[7][3] = cij[3][7];

            rqs_add(cijval[0], cijval[0], cijval[1]);
            rqs_add(cijval[0], cijval[0], cijval[2]);
            rqs_add(cijval[0], cijval[0], cijval[3]);
            rqs_add(cijval[0], cijval[0], cijval[4]);
            rqs_add(cijval[0], cijval[0], cijval[5]);
            rqs_add(cijval[0], cijval[0], cijval[6]);
            rqs_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * col_dim + j] = cijval[0][0];
            ret->element[1][i * col_dim + j] = cijval[0][1];
            ret->element[2][i * col_dim + j] = cijval[0][2];
            ret->element[3][i * col_dim + j] = cijval[0][3];
            ret->element[4][i * col_dim + j] = cijval[0][4];
            ret->element[5][i * col_dim + j] = cijval[0][5];
            ret->element[6][i * col_dim + j] = cijval[0][6];
            ret->element[7][i * col_dim + j] = cijval[0][7];
        }
    }
}
#endif // defined(__AVX2__) __AVX2__
#endif //#if defined(__BNC_QSLINEAR_H__) && defined(_DEF_BNC_QSVECTOR)
