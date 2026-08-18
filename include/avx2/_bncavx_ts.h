// ------------------------
// ---------- TS ----------
// ------------------------
#ifndef TSSIZE
    #define TSSIZE 3
#endif // TSSIZE

// ret := 0
#if defined(__AVX2__)
static inline void _bncavx2_set0_ts(__m256 ret[TSSIZE])
{
    ret[0] = _mm256_setzero_ps();
    ret[1] = _mm256_setzero_ps();
    ret[2] = _mm256_setzero_ps();
}
#endif // __AVX2__

// ret := 0
#if defined(__AVX512F__)
static inline void _bncavx512_set0_ts(__m512 ret[TSSIZE])
{
    ret[0] = _mm512_setzero_ps();
    ret[1] = _mm512_setzero_ps();
    ret[2] = _mm512_setzero_ps();
}
#endif // __AVX512F__

#if defined(__AVX2__)
// ret := ret4[][avx_index]
static inline void _bncavx2_get_ts_m256_i(tsfloat *ret, __m256 ret4[TSSIZE], int avx_index)
{
    ret->val[0] = ret4[0][avx_index];
    ret->val[1] = ret4[1][avx_index];
    ret->val[2] = ret4[2][avx_index];

    return;
}

// ret := ret4[0] + ret4[1] + ret4[2]
static void _bncavx2_rts_sum256(float ret[TSSIZE], __m256 ret4[TSSIZE])
{
    tsfloat ret4_i[8]; /* __m256 holds 8 float lanes */
    int _l;

    for(_l = 0; _l < 8; _l++) _bncavx2_get_ts_m256_i(&ret4_i[_l], ret4, _l);

    rts_set(ret, ret4_i[0].val);
    for(_l = 1; _l < 8; _l++) rts_add(ret, ret, ret4_i[_l].val);
}
// abs
static inline void _bncavx2_rts_abs(__m256 ret[TSSIZE], __m256 a[TSSIZE])
{
    int avx_index;

    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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
// ret := |ret4[0]| + |ret4[1]| + |ret4[2]| + |ret4[3]|
static void _bncavx2_rts_abssum256(float ret[TSSIZE], __m256 ret4[TSSIZE])
{
    tsfloat ret4_i[4];
    float tmp[TSSIZE];

    // ret4_i := ret4
    _bncavx2_get_ts_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_ts_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_ts_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_ts_m256_i(&ret4_i[3], ret4, 3);

    rts_abs(tmp, ret4_i[0].val);
    rts_set(ret, tmp);

    rts_abs(tmp, ret4_i[1].val);
    rts_add(ret, ret, tmp);

    rts_abs(tmp, ret4_i[2].val);
    rts_add(ret, ret, tmp);

    rts_abs(tmp, ret4_i[3].val);
    rts_add(ret, ret, tmp);
}

// ret := max(|ret4[0]|, |ret4[1]|, |ret4[2]|, |ret4[3]|)
static void _bncavx2_rts_absmax256(float ret[TSSIZE], __m256 ret4[TSSIZE])
{
    tsfloat ret4_i[4];
    float tmp[TSSIZE];

    // ret4_i := ret4
    _bncavx2_get_ts_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_ts_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_ts_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_ts_m256_i(&ret4_i[3], ret4, 3);

    rts_abs(tmp, ret4_i[0].val); 
    rts_set(ret, tmp); // ret:= |ret4_i[0]|

    rts_abs(tmp, ret4_i[1].val);
    if(rts_cmp(ret, tmp) < 0) // if(ret < |ret4_i[1]|)
        rts_set(ret, tmp);    //   ret := |ret4_i[1]|

    rts_abs(tmp, ret4_i[2].val);
    if(rts_cmp(ret, tmp) < 0) // if(ret < |ret4_i[2]|)
        rts_set(ret, tmp);    //   ret := |ret4_i[2]|

    rts_abs(tmp, ret4_i[3].val);
    if(rts_cmp(ret, tmp) < 0) // if(ret < |ret4_i[2]|)
        rts_set(ret, tmp);    //   ret := |ret4_i[2]|
}

// ret := || ret4[0]^2 + ret4[1]^2 + ret4[2]^2 + ret4[3]^2 ||_2
static void _bncavx2_rts_norm256(float ret[TSSIZE], __m256 ret4[TSSIZE])
{
    tsfloat ret4_i[4];
    float tmp[TSSIZE];

    // ret4_i := ret4
    _bncavx2_get_ts_m256_i(&ret4_i[0], ret4, 0);
    _bncavx2_get_ts_m256_i(&ret4_i[1], ret4, 1);
    _bncavx2_get_ts_m256_i(&ret4_i[2], ret4, 2);
    _bncavx2_get_ts_m256_i(&ret4_i[3], ret4, 3);

    rts_mul(tmp, ret4_i[0].val, ret4_i[0].val);
    rts_set(ret, tmp);

    rts_mul(tmp, ret4_i[1].val, ret4_i[1].val);
    rts_add(ret, ret, tmp);

    rts_mul(tmp, ret4_i[2].val, ret4_i[2].val);
    rts_add(ret, ret, tmp);

    rts_mul(tmp, ret4_i[3].val, ret4_i[3].val);
    rts_add(ret, ret, tmp);

    rts_sqrt(tmp, ret);
    rts_set(ret, tmp);
}
#endif // __AVX2__

#if defined(__AVX2__)
// e[n] := vec_sum(x[n])
//inline void vec_sum(float *e, const float *x, int n)
static inline void _bncavx2_vec_sumf(__m256 e[], const __m256 x[], int n)
{
//    float s;
    __m256 s;

	s = x[--n];
	while(--n >= 0)
    {
//		s = two_sum(x[n], s, &e[n + 1]);
        s = _bncavx2_ftwo_sum(x[n], s, &e[n + 1]);
    }
	e[0] = s;
}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
//inline void vseb(float *y, int ny, const float *e, int ne)
static inline void _bncavx2_vsebf(__m256 y[], int ny, const __m256 e[], int ne)
{
	int i, j[8], avx_index;
//	float r, eps, temp, in_y[16];
    float ftmp;
    __m256 r, eps, temp, in_y[16];

//	printf("ny = %d\n", ny);
//	if(ny > ne)
//		ny = ne;

//	j = 0;
    j[0] = 0; j[1] = 0; j[2] = 0; j[3] = 0;
    j[4] = 0; j[5] = 0; j[6] = 0; j[7] = 0;

	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
//		r = two_sum(eps, e[i + 1], &temp);
        r = _bncavx2_ftwo_sum(eps, e[i + 1], &temp);

        // if(temp != 0.0)
        for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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
    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
    {
        ///in_y[j[avx_index]][avx_index] = ftwo_sum(eps[avx_index], e[ne - 1][avx_index], in_y[j[avx_index] + 1][avx_index]); // error!
        //in_y[j[avx_index]][avx_index] = ftwo_sum(eps[avx_index], e[ne - 1][avx_index], (float *)&(in_y[j[avx_index] + 1][avx_index]));
        in_y[j[avx_index]][avx_index] = ftwo_sum(eps[avx_index], e[ne - 1][avx_index], &ftmp);
        in_y[j[avx_index] + 1][avx_index] = ftmp;

    	for(i = j[avx_index] + 2; i < ne; i++)
    		in_y[i][avx_index] = 0.0f;
    }

	for(i = 0; i < ny; i++)
		y[i] = in_y[i];

}

// y[n] := vec_sum_err_branch(vseb)(k)(e[n])
//inline void vseb(float *y, int ny, const float *e, int ne)
static inline void _bncavx2_vsebf_new(__m256 y[], int ny, const __m256 e[], int ne)
{
	int i;
//	float r, eps, temp, in_y[16];
    __m256 r, eps, temp, zeros, mask;
    __m128i j;

    zeros = _mm256_setzero_ps();

    // y := 0
    for(i = 0; i < ny; i++)
        y[i] = _mm256_setzero_ps();

	eps = e[0];
	for(i = 0; i < (ne - 2); i++)
	{
//		r = two_sum(eps, e[i + 1], &temp);
        eps = _bncavx2_ftwo_sum(eps, e[i + 1], &temp);

        // temp != 0
        mask = _mm256_cmp_ps(temp, zeros, _CMP_NEQ_OQ);
        eps = _mm256_and_ps(mask, temp);
//		printf("i, j = %d, %d\n", i, j);
	}
}

// [TODO] float := c_ts2d(a[3])

// Merge a[na] & b[nb] into c[na + nb]
// H.Okumura, "Elementary algorithms in C", 1991.
//inline void merge(float *c, float *a, int na, float *b, int nb)
static inline void _bncavx2_mergef(__m256 c[], __m256 a[], int na, __m256 b[], int nb)
{
	int i, j, k;
    int avx_index;

    for(avx_index = 0; avx_index < _BNC_S_WIDTH; avx_index++)
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
#endif // __AVX2__

#include "sort.h" // _bncavx2_bubble_sort

#if defined(__AVX2__)

#define _bncavx2_rts_add _bncavx2_rts_addq
//#define _bncavx2_rts_add _bncavx2_rts_addt

// defined in QD arith
static inline void _bncavx2_rts_addq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE]);

//static inline void _bncavx2_rts_add(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
static inline void _bncavx2_rts_addt(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
#if 0
    float in_ret[4][TSSIZE], in_a[4][TSSIZE], in_b[4][TSSIZE];

    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0];
    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1];
    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2];
    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3];

    rts_add(in_ret[0], in_a[0], in_b[0]);
    rts_add(in_ret[1], in_a[1], in_b[1]);
    rts_add(in_ret[2], in_a[2], in_b[2]);
    rts_add(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2];
#endif // 0
//#if 0
//	float z[6], e[6];
    __m256 z[6], e[6];

//	merge(z, a, 3, b, 3);
    _bncavx2_mergef(z, a, 3, b, 3);
//    z[0] = a[0]; z[1] = a[1]; z[2] = a[2];
//    z[3] = b[0]; z[4] = b[1]; z[5] = b[2];
//    _bncavx2_bubble_sort(z, 6, _BNC_SORT_ORDER_ABSMAX);

//	vec_sum(e, z, 6);
    _bncavx2_vec_sumf(e, z, 6);
//	vseb(c, 3, e, 6);
    _bncavx2_vsebf(ret, 3, e, 6);

//#endif // 0
}

// mul
static inline void _bncavx2_rts_mulq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE]);

// mul
static inline void _bncavx2_rts_mul(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
#if 0
    float in_ret[4][TSSIZE], in_a[4][TSSIZE], in_b[4][TSSIZE];

    in_ret[0][0] = ret[0][0]; in_ret[0][1] = ret[1][0]; in_ret[0][2] = ret[2][0];
    in_ret[1][0] = ret[0][1]; in_ret[1][1] = ret[1][1]; in_ret[1][2] = ret[2][1];
    in_ret[2][0] = ret[0][2]; in_ret[2][1] = ret[1][2]; in_ret[2][2] = ret[2][2];
    in_ret[3][0] = ret[0][3]; in_ret[3][1] = ret[1][3]; in_ret[3][2] = ret[2][3];

    in_a[0][0] = a[0][0]; in_a[0][1] = a[1][0]; in_a[0][2] = a[2][0];
    in_a[1][0] = a[0][1]; in_a[1][1] = a[1][1]; in_a[1][2] = a[2][1];
    in_a[2][0] = a[0][2]; in_a[2][1] = a[1][2]; in_a[2][2] = a[2][2];
    in_a[3][0] = a[0][3]; in_a[3][1] = a[1][3]; in_a[3][2] = a[2][3];

    in_b[0][0] = b[0][0]; in_b[0][1] = b[1][0]; in_b[0][2] = b[2][0];
    in_b[1][0] = b[0][1]; in_b[1][1] = b[1][1]; in_b[1][2] = b[2][1];
    in_b[2][0] = b[0][2]; in_b[2][1] = b[1][2]; in_b[2][2] = b[2][2];
    in_b[3][0] = b[0][3]; in_b[3][1] = b[1][3]; in_b[3][2] = b[2][3];

    rts_mul(in_ret[0], in_a[0], in_b[0]);
    rts_mul(in_ret[1], in_a[1], in_b[1]);
    rts_mul(in_ret[2], in_a[2], in_b[2]);
    rts_mul(in_ret[3], in_a[3], in_b[3]);

    ret[0][0] = in_ret[0][0]; ret[1][0] = in_ret[0][1]; ret[2][0] = in_ret[0][2];
    ret[0][1] = in_ret[1][0]; ret[1][1] = in_ret[1][1]; ret[2][1] = in_ret[1][2];
    ret[0][2] = in_ret[2][0]; ret[1][2] = in_ret[2][1]; ret[2][2] = in_ret[2][2];
    ret[0][3] = in_ret[3][0]; ret[1][3] = in_ret[3][1]; ret[2][3] = in_ret[3][2];
#endif // 0
//#if 0
//	float z00[2], z01[2], z10[2];
//	float in_b[3], in_c, z[3], e[4], temp[4];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

	//printf("c_ts_mul_sloppy "); fflush(stdout);

//	z00[0] = two_prod(a[0], b[0], &z00[1]);
//	z01[0] = two_prod(a[0], b[1], &z01[1]);
//	z10[0] = two_prod(a[1], b[0], &z10[1]);
	z00[0] = _bncavx2_ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = _bncavx2_ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = _bncavx2_ftwo_prod(a[1], b[0], &z10[1]);

	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	//vec_sum(in_b, z, 3);
	//in_c = fma(a[1], b[1], in_b[2]);
	_bncavx2_vec_sumf(in_b, z, 3);
	in_c = _mm256_fmadd_ps(a[1], b[1], in_b[2]);

//	z[0] = fma(a[0], b[2], z10[1]);
//	z[1] = fma(a[2], b[0], z01[1]);
	z[0] = _mm256_fmadd_ps(a[0], b[2], z10[1]);
	z[1] = _mm256_fmadd_ps(a[2], b[0], z01[1]);
//	z[2] = z[0] + z[1];
    z[2] = _mm256_add_ps(z[0], z[1]);
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
//    temp[3] = in_c + z[2];
    temp[3] = _mm256_add_ps(in_c, z[2]);
//	vec_sum(e, temp, 4);
	_bncavx2_vec_sumf(e, temp, 4);
	ret[0] = e[0];
//	vseb(&c[1], 2, &e[1], 3);
	_bncavx2_vsebf(&ret[1], 2, &e[1], 3);
//#endif // 0
}


// c[3] := -a[3]
//static inline void c_ts_neg(const float *a, float *c)
static inline void _bncavx2_rts_neg(__m256 c[TSSIZE], __m256 a[TSSIZE])
{
    __m256 zero4;

    zero4 = _mm256_setzero_ps();

	//c[0] = -a[0];
    c[0] = _mm256_sub_ps(zero4, a[0]);
	//c[1] = -a[1];
    c[1] = _mm256_sub_ps(zero4, a[1]);
    //c[2] = -a[2];
    c[2] = _mm256_sub_ps(zero4, a[2]);    
}

// c[3] := a[3] - b[3]
//static inline void c_ts_sub(float *a, float *b, float *c)
static inline void _bncavx2_rts_sub(__m256 c[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
//	float mb[3];
    __m256 mb[TSSIZE];

//	c_ts_neg(b, mb);
//	c_ts_add(a, mb, c);
    _bncavx2_rts_neg(mb, b);
    _bncavx2_rts_add(c, a, mb);
}

// c[3] := a[3] - b[3]
static inline void _bncavx2_rts_subq(__m256 c[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
//	float mb[3];
    __m256 mb[TSSIZE];

    _bncavx2_rts_neg(mb, b);
    _bncavx2_rts_addq(c, a, mb);
}

// divq
static inline void _bncavx2_rts_divq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE]);

// r[3] := to_ts(a, b, c)
//static inline void c_to_ts(float *r, float a, float b, float c)
static inline void _bncavx2_to_ts(__m256 r[TSSIZE], __m256 a, __m256 b, __m256 c)
{
//	float d[3], e[3];
	__m256 d[3], e[3];

	//d[0] = two_sum(a, b, &d[1]);
	//d[2] = c;
	//vec_sum(e, d, 3);
	//vseb(r, 3, e, 3);
	d[0] = _bncavx2_ftwo_sum(a, b, &d[1]);
	d[2] = c;
	_bncavx2_vec_sumf(e, d, 3);
	_bncavx2_vsebf(r, 3, e, 3);
}

// c[3] := a * b[3]
//static inline void c_ts_mul_d_ts(float a, float *b, float *c)
static inline void _bncavx2_rts_mul_f(__m256 c[TSSIZE], __m256 a, __m256 b[TSSIZE])
{
	//float z00[2], z01[2], z10[2];
	//float in_b[3], in_c, z[3], e[4], temp[4];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = _bncavx2_ftwo_prod(a, b[0], &z00[1]);
	z01[0] = _bncavx2_ftwo_prod(a, b[1], &z01[1]);
	//z10[0] = two_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; //z[2] = z10[0];
	_bncavx2_vec_sumf(in_b, z, 2);
	//in_c = fma(a[1], b[1], in_b[2]);

	z[0] = _mm256_fmadd_ps(a, b[2], z10[1]);
	//z[1] = z[0] + z01[1];
    z[1] = _mm256_add_ps(z[0], z01[1]);
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; temp[3] = z[1];
	_bncavx2_vec_sumf(e, temp, 4);
	c[0] = e[0];
	_bncavx2_vsebf(&c[1], 2, &e[1], 3);
}

// c[3] := a[2] * b[3]
//static inline void c_ts_mul_dd_ts_sloppy(float *a, float *b, float *c)
static inline void _bncavx2_rts_mul_ds(__m256 c[TSSIZE], __m256 a[DSSIZE], __m256 b[TSSIZE])
{
	//float z00[2], z01[2], z10[2];
	//float in_b[3], in_c, z[3], e[4], temp[4];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = _bncavx2_ftwo_prod(a[0], b[0], &z00[1]);
	z01[0] = _bncavx2_ftwo_prod(a[0], b[1], &z01[1]);
	z10[0] = _bncavx2_ftwo_prod(a[1], b[0], &z10[1]);
	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];
	_bncavx2_vec_sumf(in_b, z, 3);
	in_c = _mm256_fmadd_ps(a[1], b[1], in_b[2]);

	z[0] = _mm256_fmadd_ps(a[0], b[2], z10[1]);
	//z[1] = z[0] + z01[1];
    z[1] = _mm256_add_ps(z[0], z01[1]);
	temp[0] = z00[0];
    temp[1] = in_b[0];
    temp[2] = in_b[1];
    //temp[3] = in_c + z[1];
    temp[3] = _mm256_add_ps(in_c, z[1]);
	_bncavx2_vec_sumf(e, temp, 4);
	c[0] = e[0];
	_bncavx2_vsebf(&c[1], 2, &e[1], 3);
}

//#define _bncavx2_rts_div _bncavx2_rts_divq
#define _bncavx2_rts_div _bncavx2_rts_divtq
//#define _bncavx2_rts_div _bncavx2_rts_divt

// div
static inline void _bncavx2_rts_divt(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
	//float alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
	__m256 alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m256 zero4, two4, one_p_2dbl_eps4, one_m_2dbl_eps4;

	//c_to_ts(d2, 2.0, 0.0, 0.0);
    zero4 = _mm256_setzero_ps();
    two4 = _mm256_set1_ps(2.0);
    one_p_2dbl_eps4 = _mm256_set1_ps(ONE_P_2DBL_EPS);
    one_m_2dbl_eps4 = _mm256_set1_ps(ONE_M_2DBL_EPS);

    _bncavx2_to_ts(d2, two4, zero4, zero4);

	//alpha = ONE_P_2DBL_EPS / b[0];
    alpha = _mm256_div_ps(one_p_2dbl_eps4, b[0]);

	//h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
    h1 = _mm256_fmsub_ps(alpha, b[0], one_p_2dbl_eps4);

	//h1 = -fma(alpha, b[1], h1);
    h1 = _mm256_fmadd_ps(alpha, b[1], h1);
    h1 = _mm256_sub_ps(zero4, h1);

	//in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b[0] = _bncavx2_ftwo_prod(alpha, one_m_2dbl_eps4, &in_b[1]);

	//in_b12 = fma(alpha, h1, in_b[1]);
    in_b12 = _mm256_fmadd_ps(alpha, h1, in_b[1]);

	//in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
	in_b[0] = _bncavx2_fquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_ts_2mtw_dd_ts(in_b, b, temp); // temp := 2 - c_ts_mul_dd_ts(in_b, b, temp)
	//c_ts_mul_dd_ts(in_b, b, temp);
    //c_ts_sub(d2, temp, temp);
	//c_ts_mul_dd_ts(in_b, a, in_c);
	//c_ts_mul(in_c, temp, c);
	_bncavx2_rts_mul_ds(temp, in_b, b);
    _bncavx2_rts_sub(temp, d2, temp);
	_bncavx2_rts_mul_ds(in_c, in_b, a);
	_bncavx2_rts_mul(ret, in_c, temp);
}
// div
static inline void _bncavx2_rts_divtq(__m256 ret[TSSIZE], __m256 a[TSSIZE], __m256 b[TSSIZE])
{
	//float alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
	__m256 alpha, h1, in_b[2], in_b12, temp[3], in_c[3], d2[3];
    __m256 zero4, two4, one_p_2dbl_eps4, one_m_2dbl_eps4;

	//c_to_ts(d2, 2.0, 0.0, 0.0);
    zero4 = _mm256_setzero_ps();
    two4 = _mm256_set1_ps(2.0);
    one_p_2dbl_eps4 = _mm256_set1_ps(ONE_P_2DBL_EPS);
    one_m_2dbl_eps4 = _mm256_set1_ps(ONE_M_2DBL_EPS);

    _bncavx2_to_ts(d2, two4, zero4, zero4);

	//alpha = ONE_P_2DBL_EPS / b[0];
    alpha = _mm256_div_ps(one_p_2dbl_eps4, b[0]);

	//h1 =  fma(alpha, b[0], -ONE_P_2DBL_EPS);
    h1 = _mm256_fmsub_ps(alpha, b[0], one_p_2dbl_eps4);

	//h1 = -fma(alpha, b[1], h1);
    h1 = _mm256_fmadd_ps(alpha, b[1], h1);
    h1 = _mm256_sub_ps(zero4, h1);

	//in_b[0] = two_prod(alpha, ONE_M_2DBL_EPS, &in_b[1]);
	in_b[0] = _bncavx2_ftwo_prod(alpha, one_m_2dbl_eps4, &in_b[1]);

	//in_b12 = fma(alpha, h1, in_b[1]);
    in_b12 = _mm256_fmadd_ps(alpha, h1, in_b[1]);

	//in_b[0] = quick_two_sum(in_b[0], in_b12, &in_b[1]);
	in_b[0] = _bncavx2_fquick_two_sum(in_b[0], in_b12, &in_b[1]);
//	c_ts_2mtw_dd_ts(in_b, b, temp); // temp := 2 - c_ts_mul_dd_ts(in_b, b, temp)
	//c_ts_mul_dd_ts(in_b, b, temp);
    //c_ts_sub(d2, temp, temp);
	//c_ts_mul_dd_ts(in_b, a, in_c);
	//c_ts_mul(in_c, temp, c);
	_bncavx2_rts_mul_ds(temp, in_b, b);
    //_bncavx2_rts_sub(temp, d2, temp);
    _bncavx2_rts_subq(temp, d2, temp); //
	_bncavx2_rts_mul_ds(in_c, in_b, a);
	_bncavx2_rts_mul(ret, in_c, temp);
}
#endif // __AVX2__

#if 0
// tsrel_diff
inline static tsfloat tsrel_diff(tsfloat a, tsfloat b)
{
    tsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rts_sub(rel_diff.val, a.val, b.val);
    rts_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rts_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rts_abs(abs_a.val, a.val);
        rts_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
#endif // 0