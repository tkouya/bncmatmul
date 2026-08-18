// ------------------------
// ------ TD AVX-512 ------
// ------------------------
#ifndef __BNCAVX_TD_AVX512_H
#define __BNCAVX_TD_AVX512_H

#ifndef TDSIZE
    #define TDSIZE 3
#endif // TDSIZE

#if defined(__AVX512F__)

// ret := 0
static inline void _bncavx512_set0_td(__m512d ret[TDSIZE])
{
    ret[0] = _mm512_setzero_pd();
    ret[1] = _mm512_setzero_pd();
    ret[2] = _mm512_setzero_pd();
}
#define _bncavx512_rtd_set0(ret) _bncavx512_set0_td((ret))

// ret := val
static inline void _bncavx512_rtd_set(__m512d ret[TDSIZE], __m512d val[TDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = val[2];
}

// ret := (double)val
static inline void _bncavx512_rtd_set_d(__m512d ret[TDSIZE], __m512d val)
{
    ret[0] = val;
    ret[1] = _mm512_setzero_pd(); // val[1];
    ret[2] = _mm512_setzero_pd(); // val[2];
}

// ret := (DD)val
static inline void _bncavx512_rtd_set_dd(__m512d ret[TDSIZE], __m512d val[DDSIZE])
{
    ret[0] = val[0];
    ret[1] = val[1];
    ret[2] = _mm512_setzero_pd(); // val[2];
}

// ret := [val val val val]
static inline void _bncavx512_rtd_set1_td(__m512d ret[TDSIZE], double val[TDSIZE])
{
    ret[0] = _mm512_set1_pd(val[0]);
    ret[1] = _mm512_set1_pd(val[1]);
    ret[2] = _mm512_set1_pd(val[2]);
}

// ret := ret8[][avx_index]
static inline void _bncavx512_get_td_m512d_i(tdfloat *ret, __m512d ret8[TDSIZE], int avx_index)
{
    ret->val[0] = ret8[0][avx_index];
    ret->val[1] = ret8[1][avx_index];
    ret->val[2] = ret8[2][avx_index];

    return;
}

// ret := ret8[0] + ret8[1] + ... + ret8[7]
static void _bncavx512_rtd_sum512d(double ret[TDSIZE], __m512d ret8[TDSIZE])
{
    tdfloat ret8_i[8];

    // ret8_i := ret8
    _bncavx512_get_td_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_td_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_td_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_td_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_td_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_td_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_td_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_td_m512d_i(&ret8_i[7], ret8, 7);

    rtd_set(ret, ret8_i[0].val);
    rtd_add(ret, ret, ret8_i[1].val);
    rtd_add(ret, ret, ret8_i[2].val);
    rtd_add(ret, ret, ret8_i[3].val);
    rtd_add(ret, ret, ret8_i[4].val);
    rtd_add(ret, ret, ret8_i[5].val);
    rtd_add(ret, ret, ret8_i[6].val);
    rtd_add(ret, ret, ret8_i[7].val);
}

// abs
static inline void _bncavx512_rtd_abs(__m512d ret[TDSIZE], __m512d a[TDSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < 8; avx_index++)
    {
        if(a[0][avx_index] < 0.0)
        {
            ret[0][avx_index] = -a[0][avx_index];
            ret[1][avx_index] = -a[1][avx_index];
            ret[2][avx_index] = -a[2][avx_index];
        }
        else
        {
            ret[0][avx_index] = a[0][avx_index];
            ret[1][avx_index] = a[1][avx_index];
            ret[2][avx_index] = a[2][avx_index];
        }
    }
}
// ret := |ret8[0]| + |ret8[1]| + |ret8[2]| + |ret8[3]|
static void _bncavx512_rtd_abssum512d(double ret[TDSIZE], __m512d ret8[TDSIZE])
{
    tdfloat ret8_i[8];
    static double tmp[TDSIZE];

    // ret8_i := ret8
    _bncavx512_get_td_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_td_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_td_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_td_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_td_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_td_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_td_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_td_m512d_i(&ret8_i[7], ret8, 7);

    rtd_abs(tmp, ret8_i[0].val);
    rtd_set(ret, tmp);

    rtd_abs(tmp, ret8_i[1].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[2].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[3].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[4].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[5].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[6].val); rtd_add(ret, ret, tmp);
    rtd_abs(tmp, ret8_i[7].val); rtd_add(ret, ret, tmp);
}

// ret := max(|ret8[0]|, |ret8[1]|, |ret8[2]|, |ret8[3]|)
static void _bncavx512_rtd_absmax256d(double ret[TDSIZE], __m512d ret8[TDSIZE])
{
    tdfloat ret8_i[8];
    static double tmp[TDSIZE];

    // ret8_i := ret8
    _bncavx512_get_td_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_td_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_td_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_td_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_td_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_td_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_td_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_td_m512d_i(&ret8_i[7], ret8, 7);

    rtd_abs(tmp, ret8_i[0].val); 
    rtd_set(ret, tmp); // ret:= |ret8_i[0]|

    rtd_abs(tmp, ret8_i[1].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[1]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[1]|

    rtd_abs(tmp, ret8_i[2].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[2]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[2]|

    rtd_abs(tmp, ret8_i[3].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[3]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[3]|

    rtd_abs(tmp, ret8_i[4].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[4]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[4]|

    rtd_abs(tmp, ret8_i[5].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[5]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[5]|

    rtd_abs(tmp, ret8_i[6].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[6]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[6]|

    rtd_abs(tmp, ret8_i[7].val);
    if(rtd_cmp(ret, tmp) < 0) // if(ret < |ret8_i[7]|)
        rtd_set(ret, tmp);    //   ret := |ret8_i[7]|

}

// ret := || ret8[0]^2 + ret8[1]^2 + ret8[2]^2 + ret8[3]^2 ||_2
static void _bncavx512_rtd_norm512d(double ret[TDSIZE], __m512d ret8[TDSIZE])
{
    tdfloat ret8_i[8];
    static double tmp[TDSIZE];

    // ret8_i := ret8
    _bncavx512_get_td_m512d_i(&ret8_i[0], ret8, 0);
    _bncavx512_get_td_m512d_i(&ret8_i[1], ret8, 1);
    _bncavx512_get_td_m512d_i(&ret8_i[2], ret8, 2);
    _bncavx512_get_td_m512d_i(&ret8_i[3], ret8, 3);
    _bncavx512_get_td_m512d_i(&ret8_i[4], ret8, 4);
    _bncavx512_get_td_m512d_i(&ret8_i[5], ret8, 5);
    _bncavx512_get_td_m512d_i(&ret8_i[6], ret8, 6);
    _bncavx512_get_td_m512d_i(&ret8_i[7], ret8, 7);

    rtd_mul(tmp, ret8_i[0].val, ret8_i[0].val);
    rtd_set(ret, tmp);

    rtd_mul(tmp, ret8_i[1].val, ret8_i[1].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[2].val, ret8_i[2].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[3].val, ret8_i[3].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[4].val, ret8_i[4].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[5].val, ret8_i[5].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[6].val, ret8_i[6].val); rtd_add(ret, ret, tmp);
    rtd_mul(tmp, ret8_i[7].val, ret8_i[7].val); rtd_add(ret, ret, tmp);

    rtd_sqrt(tmp, ret);
    rtd_set(ret, tmp);
}

// e[n] := vec_sum(x[n])
//inline void vec_sum(double *e, const double *x, int n)
static inline void _bncavx512_vec_sum(__m512d e[], const __m512d x[], int n)
{
//    double s;
    __m512d s;

	s = x[--n];
	while(--n >= 0)
    {
//		s = two_sum(x[n], s, &e[n + 1]);
        s = _bncavx512_dtwo_sum(x[n], s, &e[n + 1]);
    }
	e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
//inline void vseb(double *y, int ny, const double *e, int ne)
static inline void _bncavx512_vseb(__m512d y[], int ny, const __m512d e[], int ne)
{
//	int i, j[4], avx_index;
	int i, j[8], avx_index;
//	double r, eps, temp, in_y[16];
    double dtmp;
    __m512d r, eps, temp, in_y[16];

//	printf("ny = %d\n", ny);
//	if(ny > ne)
//		ny = ne;

//	j = 0;
    j[0] = 0; j[1] = 0; j[2] = 0; j[3] = 0;
	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
//		r = two_sum(eps, e[i + 1], &temp);
        r = _bncavx512_dtwo_sum(eps, e[i + 1], &temp);

        // if(temp != 0.0)
        //for(avx_index = 0; avx_index < 4; avx_index++)
        for(avx_index = 0; avx_index < 8; avx_index++)
		{
            if(temp[avx_index] != 0.0)
            {
    			in_y[j[avx_index]][avx_index] = r[avx_index];
	    		eps[avx_index] = temp[avx_index];
			    j[avx_index]++;
            }
    		else
    			eps[avx_index] = r[avx_index];
        }
//		printf("i, j = %d, %d\n", i, j);
	}
//	printf("j, j+1 = %d, %d\n", j, j + 1);
//	in_y[j] = two_sum(eps, e[ne - 1], &in_y[j + 1]);
    //for(avx_index = 0; avx_index < 4; avx_index++)
    for(avx_index = 0; avx_index < 8; avx_index++)
    {
//        in_y[j[avx_index]][avx_index] = two_sum(eps[avx_index], e[ne - 1][avx_index], &(in_y[j[avx_index] + 1][avx_index]));
        //in_y[j[avx_index]][avx_index] = two_sum(eps[avx_index], e[ne - 1][avx_index], (double *)&(in_y[j[avx_index] + 1][avx_index]));
        in_y[j[avx_index]][avx_index] = two_sum(eps[avx_index], e[ne - 1][avx_index], &dtmp);
        in_y[j[avx_index] + 1][avx_index] = dtmp;

    	for(i = j[avx_index] + 2; i < ne; i++)
    		in_y[i][avx_index] = 0.0;
    }

	for(i = 0; i < ny; i++)
		y[i] = in_y[i];

}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
//inline void vseb(double *y, int ny, const double *e, int ne)
static inline void _bncavx512_vseb_new(__m512d y[], int ny, const __m512d e[], int ne)
{
	int i;
//	double r, eps, temp, in_y[16];
    __m512d r, eps, temp, zeros, mask;
    __mmask8 mask8;

    zeros = _mm512_setzero_pd();

    // y := 0
    for(i = 0; i < ny; i++)
        y[i] = _mm512_setzero_pd();

	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
//		r = two_sum(eps, e[i + 1], &temp);
        eps = _bncavx512_dtwo_sum(eps, e[i + 1], &temp);

        // temp != 0
        //mask = _mm512_cmp_pd(temp, zeros, _CMP_NEQ_OQ);
        mask8 = _mm512_cmp_pd_mask(temp, zeros, _CMP_NEQ_OQ);
        eps = _mm512_maskz_and_pd(mask8, temp, zeros);
//		printf("i, j = %d, %d\n", i, j);
	}
}

// [TODO] double := c_td2d(a[3])

// Merge a[na] & b[nb] into c[na + nb]
// H.Okumura, "Elementary algorithms in C", 1991.
//inline void merge(double *c, double *a, int na, double *b, int nb)
static inline void _bncavx512_merge(__m512d c[], __m512d a[], int na, __m512d b[], int nb)
{
	int i, j, k;
    int avx_index;

    for(avx_index = 0; avx_index < 8; avx_index++)
    {
        i = j = k = 0;
        while((i < na) && (j < nb))
        {
            if(fabs(a[i][avx_index]) >= fabs(b[j][avx_index]))
                c[k++][avx_index] = a[i++][avx_index];
            else
                c[k++][avx_index] = b[j++][avx_index];
        }
        while(i < na)
            c[k++][avx_index] = a[i++][avx_index];
        while(j < nb)
            c[k++][avx_index] = b[j++][avx_index];
    }
}

#include "sort.h" // _bncavx512_bubble_sort

#ifdef USE_TD_BF
    #define _bncavx512_rtd_add _bncavx512_rtd_add_bf
    #define _bncavx512_rtd_mul _bncavx512_rtd_mul_bf
#else // USE_TD_BF
    #define _bncavx512_rtd_add _bncavx512_rtd_addq
    #define _bncavx512_rtd_mul _bncavx512_rtd_mult
#endif // USE_TD_BF
//#define _bncavx512_rtd_add _bncavx512_rtd_addt

// defined in QD arith
static inline void _bncavx512_rtd_addq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE]);

//static inline void _bncavx512_rtd_add(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
static inline void _bncavx512_rtd_addt(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//#if 0
//	double z[6], e[6];
    __m512d z[6], e[6];

//	merge(z, a, 3, b, 3);
    _bncavx512_merge(z, a, 3, b, 3);
//    z[0] = a[0]; z[1] = a[1]; z[2] = a[2];
//    z[3] = b[0]; z[4] = b[1]; z[5] = b[2];
//    _bncavx512_bubble_sort(z, 6, _BNC_SORT_ORDER_ABSMAX);

//	vec_sum(e, z, 6);
    _bncavx512_vec_sum(e, z, 6);
//	vseb(c, 3, e, 6);
    _bncavx512_vseb(ret, 3, e, 6);

//#endif // 0
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
//void Add3(const double x[3], const double y[3], double z[3]) {
static inline void _bncavx512_rtd_add_bf(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
	__m512d a0, b0, c0, d0, e0, f0;
	__m512d a1, b1, c1, d1, e1, f1;
	__m512d a2, b2, c2, d2, e2;
	__m512d a3, b3, c3, d3;
	__m512d a4, b4, c4, d4;
	__m512d a5, b5, c5, d5;
	__m512d a6, b6, c6;
	__m512d a7, b7, c7;
	__m512d a8, b8, c8;

	a0 = a[0];
    b0 = b[0];
    c0 = a[1];
    d0 = b[1];
    e0 = a[2];
    f0 = b[2];
    a1 = _bncavx512_dtwo_sum(a0, b0, &b1);
    c1 = _bncavx512_dtwo_sum(c0, d0, &d1);
    e1 = _bncavx512_dtwo_sum(e0, f0, &f1);
    a2 = _bncavx512_dquick_two_sum(a1, c1, &c2);
    b2 = _mm512_add_pd(b1, f1);
    d2 = _bncavx512_dtwo_sum(d1, e1, &e2);
    a3 = _bncavx512_dquick_two_sum(a2, d2, &d3);
    b3 = _bncavx512_dtwo_sum(b2, c2, &c3);
    c4 = _mm512_add_pd(c3, e2);
    c5 = _bncavx512_dtwo_sum(c4, d3, &d5);
    b6 = _bncavx512_dtwo_sum(b3, c5, &c6);
    a7 = _bncavx512_dquick_two_sum(a3, b6, &b7);
    c7 = _mm512_add_pd(c6, d5);
    b8 = _bncavx512_dquick_two_sum(b7, c7, &c8);
    //return MultiFloat<T, 3>{a7, b8, c8};
	ret[0] = a7;
	ret[1] = b8;
	ret[2] = c8;
}


// mul
static inline void _bncavx512_rtd_mulq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE]);

// mul
//static inline void _bncavx512_rtd_mul(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
static inline void _bncavx512_rtd_mult(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//#if 0
//	double z00[2], z01[2], z10[2];
//	double in_b[3], in_c, z[3], e[4], temp[4];
	__m512d z00[2], z01[2], z10[2];
	__m512d in_b[3], in_c, z[3], e[4], temp[4];

	//printf("c_td_mul_sloppy "); fflush(stdout);

//	z00[0] = two_prod(a[0], b[0], &z00[1]);
//	z01[0] = two_prod(a[0], b[1], &z01[1]);
//	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z00[0] = _bncavx512_dtwo_prod(a[0], b[0], &z00[1]);
	z01[0] = _bncavx512_dtwo_prod(a[0], b[1], &z01[1]);
	z10[0] = _bncavx512_dtwo_prod(a[1], b[0], &z10[1]);

	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	//vec_sum(in_b, z, 3);
	//in_c = fma(a[1], b[1], in_b[2]);
	_bncavx512_vec_sum(in_b, z, 3);
	in_c = _mm512_fmadd_pd(a[1], b[1], in_b[2]);

//	z[0] = fma(a[0], b[2], z10[1]);
//	z[1] = fma(a[2], b[0], z01[1]);
	z[0] = _mm512_fmadd_pd(a[0], b[2], z10[1]);
	z[1] = _mm512_fmadd_pd(a[2], b[0], z01[1]);
//	z[2] = z[0] + z[1];
    z[2] = _mm512_add_pd(z[0], z[1]);
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
//    temp[3] = in_c + z[2];
    temp[3] = _mm512_add_pd(in_c, z[2]);
//	vec_sum(e, temp, 4);
	_bncavx512_vec_sum(e, temp, 4);
	ret[0] = e[0];
//	vseb(&c[1], 2, &e[1], 3);
	_bncavx512_vseb(&ret[1], 2, &e[1], 3);
//#endif // 0
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm by D.K.Zhang and A.Aiken at SC2025
// void Mul3(const double x[3], const double y[3], double z[3]) {
static inline void _bncavx512_rtd_mul_bf(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
	__m512d a0, b0, c0, d0, e0, f0, g0, h0, i0;
	__m512d a1, b1, c1, d1, e1, f1, g1, h1, i1;
	__m512d a2, b2, c2, d2, e2, f2, g2;
	__m512d a3, b3, c3, d3, e3;
	__m512d a4, b4, c4;
	__m512d a5, b5, c5;
	__m512d a6, b6, c6;
	__m512d a7, b7, c7;
	
    a0 = _bncavx512_dtwo_prod(a[0], b[0], &b0);
    c0 = _bncavx512_dtwo_prod(a[0], b[1], &e0);
    d0 = _bncavx512_dtwo_prod(a[1], b[0], &f0);
    g0 = _mm512_mul_pd(a[0], b[2]);
    h0 = _mm512_mul_pd(a[1], b[1]);
    i0 = _mm512_mul_pd(a[2], b[0]);
    c1 = _bncavx512_dtwo_sum(c0, d0, &d1);
    e1 = _mm512_add_pd(e0, f0);
    g1 = _mm512_add_pd(g0, i0);
    b2 = _bncavx512_dtwo_sum(b0, c1, &c2);
    g2 = _mm512_add_pd(g1, h0);
    a3 = _bncavx512_dquick_two_sum(a0, b2, &b3);
    c3 = _mm512_add_pd(c2, d1);
    e3 = _mm512_add_pd(e1, g2);
    c4 = _mm512_add_pd(c3, e3);
    b5 = _bncavx512_dquick_two_sum(b3, c4, &c5);
    a6 = _bncavx512_dquick_two_sum(a3, b5, &b6);
    b7 = _bncavx512_dquick_two_sum(b6, c5, &c7);
    //return MultiFloat<T, 3>{a6, b7, c7};
	ret[0] = a6;
	ret[1] = b7;
	ret[2] = c7;
}


// c[3] := -a[3]
//static inline void c_td_neg(const double *a, double *c)
static inline void _bncavx512_rtd_neg(__m512d c[TDSIZE], __m512d a[TDSIZE])
{
    __m512d zero8;

    zero8 = _mm512_setzero_pd();

	//c[0] = -a[0];
    c[0] = _mm512_sub_pd(zero8, a[0]);
	//c[1] = -a[1];
    c[1] = _mm512_sub_pd(zero8, a[1]);
    //c[2] = -a[2];
    c[2] = _mm512_sub_pd(zero8, a[2]);    
}

// c[3] := a[3] - b[3]
//static inline void c_td_sub(double *a, double *b, double *c)
static inline void _bncavx512_rtd_sub(__m512d c[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//	double mb[3];
    __m512d mb[TDSIZE];

//	c_td_neg(b, mb);
//	c_td_add(a, mb, c);
    _bncavx512_rtd_neg(mb, b);
    _bncavx512_rtd_add(c, a, mb);
}

// c[3] := a[3] - b[3]
static inline void _bncavx512_rtd_subq(__m512d c[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
//	double mb[3];
    __m512d mb[TDSIZE];

    _bncavx512_rtd_neg(mb, b);
    _bncavx512_rtd_addq(c, a, mb);
}

// divq
static inline void _bncavx512_rtd_divq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE]);

// r[3] := to_td(a, b, c)
//static inline void c_to_td(double *r, double a, double b, double c)
static inline void _bncavx512_to_td(__m512d r[TDSIZE], __m512d a, __m512d b, __m512d c)
{
//	double d[3], e[3];
	__m512d d[3], e[3];

	//d[0] = two_sum(a, b, &d[1]);
	//d[2] = c;
	//vec_sum(e, d, 3);
	//vseb(r, 3, e, 3);
	d[0] = _bncavx512_dtwo_sum(a, b, &d[1]);
	d[2] = c;
	_bncavx512_vec_sum(e, d, 3);
	_bncavx512_vseb(r, 3, e, 3);
}

// c[3] := a * b[3]
//static inline void c_td_mul_d_td(double a, double *b, double *c)
//static inline void _bncavx512_rtd_mul_d(__m512d c[TDSIZE], __m512d a, __m512d b[TDSIZE])
static inline void _bncavx512_rtd_mul_d(__m512d c[TDSIZE], __m512d b[TDSIZE], __m512d a)
{
	//double z00[2], z01[2], z10[2];
	//double in_b[3], in_c, z[3], e[4], temp[4];
	__m512d z00[2], z01[2], z10[2];
	__m512d in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = _bncavx512_dtwo_prod(a, b[0], &z00[1]);
	z01[0] = _bncavx512_dtwo_prod(a, b[1], &z01[1]);
	//z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; //z[2] = z10[0];
	_bncavx512_vec_sum(in_b, z, 2);
	//in_c = fma(a[1], b[1], in_b[2]);

	z[0] = _mm512_fmadd_pd(a, b[2], z10[1]);
	//z[1] = z[0] + z01[1];
    z[1] = _mm512_add_pd(z[0], z01[1]);
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
	_bncavx512_vec_sum(e, temp, 4);
	c[0] = e[0];
	_bncavx512_vseb(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]
//static inline void c_td_mul_dd_td_sloppy(double *a, double *b, double *c)
static inline void _bncavx512_rtd_mul_dd(__m512d c[TDSIZE], __m512d a[DDSIZE], __m512d b[TDSIZE])
{
	//double z00[2], z01[2], z10[2];
	//double in_b[3], in_c, z[3], e[4], temp[4];
	__m512d z00[2], z01[2], z10[2];
	__m512d in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = _bncavx512_dtwo_prod(a[0], b[0], &z00[1]);
	z01[0] = _bncavx512_dtwo_prod(a[0], b[1], &z01[1]);
	z10[0] = _bncavx512_dtwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	_bncavx512_vec_sum(in_b, z, 3);
	in_c = _mm512_fmadd_pd(a[1], b[1], in_b[2]);

	z[0] = _mm512_fmadd_pd(a[0], b[2], z10[1]);
	//z[1] = z[0] + z01[1];
    z[1] = _mm512_add_pd(z[0], z01[1]);
	temp[0] = z00[0];
    temp[1] = in_b[0];
    temp[2] = in_b[1];
    //temp[3] = in_c + z[1];
    temp[3] = _mm512_add_pd(in_c, z[1]);
	_bncavx512_vec_sum(e, temp, 4);
	c[0] = e[0];
	_bncavx512_vseb(&c[1], 2, &e[1], 3);
}

//#define _bncavx512_rtd_div _bncavx512_rtd_divq
#define _bncavx512_rtd_div _bncavx512_rtd_divtq
//#define _bncavx512_rtd_div _bncavx512_rtd_divt

// div
static inline void _bncavx512_rtd_divt(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
	//double alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
	__m512d alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m512d zero4, two4, one_p_2dbl_eps4, one_m_2dbl_eps4;

	//c_to_td(d2, 2.0, 0.0, 0.0);
    zero4 = _mm512_setzero_pd();
    two4 = _mm512_set1_pd(2.0);
    one_p_2dbl_eps4 = _mm512_set1_pd(ONE_P_2DBL_EPS);
    one_m_2dbl_eps4 = _mm512_set1_pd(ONE_M_2DBL_EPS);

    _bncavx512_to_td(d2, two4, zero4, zero4);

	//alpha = ONE_P_2DBL_EPS / b[0];
    alpha = _mm512_div_pd(one_p_2dbl_eps4, b[0]);

	//h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
    h1 = _mm512_fmsub_pd(alpha, b[0], one_p_2dbl_eps4);

	//h1 = -fma(alpha, b[1], h1);
    h1 = _mm512_fmadd_pd(alpha, b[1], h1);
    h1 = _mm512_sub_pd(zero4, h1);

	//in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b[0] = _bncavx512_dtwo_prod(alpha, one_m_2dbl_eps4, &in_b[1]);

	//in_b12 = fma(alpha, h1, in_b[1]);
    in_b12 = _mm512_fmadd_pd(alpha, h1, in_b[1]);

	//in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
	in_b[0] = _bncavx512_dquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_td_2mtw_dd_td(in_b, b, temp); // temp := 2 - c_td_mul_dd_td(in_b, b, temp)
	//c_td_mul_dd_td(in_b, b, temp);
    //c_td_sub(d2, temp, temp);
	//c_td_mul_dd_td(in_b, a, in_c);
	//c_td_mul(in_c, temp, c);
	_bncavx512_rtd_mul_dd(temp, in_b, b);
    _bncavx512_rtd_sub(temp, d2, temp);
	_bncavx512_rtd_mul_dd(in_c, in_b, a);
	_bncavx512_rtd_mul(ret, in_c, temp);
}
// div
static inline void _bncavx512_rtd_divtq(__m512d ret[TDSIZE], __m512d a[TDSIZE], __m512d b[TDSIZE])
{
	//double alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
	__m512d alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m512d zero4, two4, one_p_2dbl_eps4, one_m_2dbl_eps4;

	//c_to_td(d2, 2.0, 0.0, 0.0);
    zero4 = _mm512_setzero_pd();
    two4 = _mm512_set1_pd(2.0);
    one_p_2dbl_eps4 = _mm512_set1_pd(ONE_P_2DBL_EPS);
    one_m_2dbl_eps4 = _mm512_set1_pd(ONE_M_2DBL_EPS);

    _bncavx512_to_td(d2, two4, zero4, zero4);

	//alpha = ONE_P_2DBL_EPS / b[0];
    alpha = _mm512_div_pd(one_p_2dbl_eps4, b[0]);

	//h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
    h1 = _mm512_fmsub_pd(alpha, b[0], one_p_2dbl_eps4);

	//h1 = -fma(alpha, b[1], h1);
    h1 = _mm512_fmadd_pd(alpha, b[1], h1);
    h1 = _mm512_sub_pd(zero4, h1);

	//in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b[0] = _bncavx512_dtwo_prod(alpha, one_m_2dbl_eps4, &in_b[1]);

	//in_b12 = fma(alpha, h1, in_b[1]);
    in_b12 = _mm512_fmadd_pd(alpha, h1, in_b[1]);

	//in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
	in_b[0] = _bncavx512_dquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_td_2mtw_dd_td(in_b, b, temp); // temp := 2 - c_td_mul_dd_td(in_b, b, temp)
	//c_td_mul_dd_td(in_b, b, temp);
    //c_td_sub(d2, temp, temp);
	//c_td_mul_dd_td(in_b, a, in_c);
	//c_td_mul(in_c, temp, c);
	_bncavx512_rtd_mul_dd(temp, in_b, b);
    //_bncavx512_rtd_sub(temp, d2, temp);
    _bncavx512_rtd_subq(temp, d2, temp); //
	_bncavx512_rtd_mul_dd(in_c, in_b, a);
	_bncavx512_rtd_mul(ret, in_c, temp);
}

#endif // __AVX512F__
#endif //ndef __BNCAVX_TD_AVX512_H
