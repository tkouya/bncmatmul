/********************************************************************************/
/* ddlinear.c: Double-double and Quadruple precision Linear Computation Library */
/* Copyright (C) 2015-2020 Tomonori Kouya                                       */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
//#include "rdd.h"
#include "ddlinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dd_td_qd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ddrel_diff
ddfloat ddrel_diff(ddfloat a, ddfloat b)
{
    ddfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rdd_sub(rel_diff.val, a.val, b.val);
    rdd_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rdd_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rdd_abs(abs_a.val, a.val);
        rdd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}

ddfloat ddrel_diff_array(ddfloat approx_a[], ddfloat approx_b[], int dim, int print_flag)
{
    int i;
    ddfloat rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = ddrel_diff(approx_a[0], approx_b[0]);
    rdd_set(rel_min.val, rel_diff.val);
    rdd_set(rel_max.val, rel_diff.val);
    rdd_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = ddrel_diff(approx_a[i], approx_b[i]);
        if(rdd_cmp(rel_diff.val, rel_min.val) < 0) rdd_set(rel_min.val, rel_diff.val);
        if(rdd_cmp(rel_diff.val, rel_max.val) > 0) rdd_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rdd_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (ddfloat)dim;
    rdd_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rdd_out_str(rel_max.val); printf(" "); rdd_out_str(rel_min.val);  printf(" "); rdd_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}

#if defined(USE_GMP) && defined(USE_MPFR)
// Frobenius norm
ddfloat ddnormf(ddfloat array[], int dim)
{
    int i;
    ddfloat ret, tmp;
    mpfr_t mpfr_ret;

    rdd_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rdd_mul(tmp.val, array[i].val, array[i].val);
        rdd_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rdd_out_str(ret.val); printf("\n");
//  rdd_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_dd(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_dd(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/* void set_test_ddvector(ddfloat vec[], int sqrt_seed, int dim)
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
} */
#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize DDVector
DDVector init_ddvector(int dimension)
{
	DDVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_ddvector\n");
		return ret;
	}

	ret = (DDVector)BNC_MALLOC(sizeof(ddvector));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_dim = (long int)ceil((double)(dimension) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

	ret->element[0] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[0] == NULL)
	{ 	
		free(ret);
		return NULL;
	}
	ret->element[1] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][i]), zero4);
		_mm256_store_pd(&(ret->element[1][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
	}
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free DDVector
void free_ddvector(DDVector vec)
{
    int i;

    for(i = 0; i < DDSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// DDVector vec -> ddfloat array
void set_ddfloat_ddvec(ddfloat ret[], int ret_dim, DDVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DDSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// ddfloat array -> DDVector ret
void set_ddvector_ddfloat(DDVector ret, ddfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DDSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print ddvector
void print_ddvector(DDVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_dd_write((vec->element + index * DDSIZE));
		c_dd_write(GET_DDVECTOR_I(vec, index));
	}
}

// set a zero vector
void set0_ddvector(DDVector vec)
{
	long int i;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&vec->element[0][i], zero4);
		_mm256_store_pd(&vec->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&vec->element[0][i], zero8);
		_mm512_store_pd(&vec->element[1][i], zero8);
	}
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0;
		vec->element[1][i] = 0.0;
	}
#endif // __AVX2__
}

// set_ddvector_i_str
void set_ddvector_i_str(DDVector vec, long int index, const char *str)
{
	double tmp[DDSIZE];

	//rdd_get_str(tmp, str);
	rdd_set_str(tmp, str);

	set_ddvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for DDVector               */
/*
void add_ddvector(DDVector c, DDVector a, DDVector b)
void add2_ddvector(DDVector c, DDVector a)
void sub_ddvector(DDVector c, DDVector a, DDVector b)
void sub2_ddvector(DDVector c, DVector a)
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
void cmul2_ddvector(DDVector c, double val[DDSIZE])
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
double ip_ddvector(DDVector a, DDVector b)
double norm1_ddvector(DDVector a)
double norm2_ddvector(DDVector a)
double normi_ddvector(DDVector a)
void subst_ddvector(DDVector c, DDVector a)
*/
/*************************************************/
/* c = a + b */
void add_ddvector(DDVector c, DDVector a, DDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_add(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_add(tmp, get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_ddvector(DDVector c, DDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_ret, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_add(tmp, get_ddvector_i(c, i), get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_ddvector(DDVector c, DDVector a, DDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_sub(tmp, get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_ddvector(DDVector c, DDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_sub(in_ret, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_sub(in_ret, in_ret, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_sub(tmp, get_ddvector_i(c, i), get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(c4, val4, a4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_mul(c8, val8, a8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_ddvector(DDVector c, double val[DDSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		c4[0] = _mm256_load_pd(&(c->element[0][index]));
		c4[1] = _mm256_load_pd(&(c->element[1][index]));

		_bncavx2_rdd_mul(c4, val4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		c8[0] = _mm512_load_pd(&(c->element[0][index]));
		c8[1] = _mm512_load_pd(&(c->element[1][index]));

		_bncavx512_rdd_mul(c8, val8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(c, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__	double tmp[DDSIZE];

}

/* c = a + val * b */
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
{
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], b4[DDSIZE], c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx2_rdd_mul(c4, val4, b4);
		_bncavx2_rdd_add(c4, a4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], b8[DDSIZE], c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx512_rdd_mul(c8, val8, b8);
		_bncavx512_rdd_add(c8, a8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(b, i));
		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], b4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx2_rdd_mul(tmp4, a4, b4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rdd_sum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], b8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx512_rdd_mul(tmp8, a8, b8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	_bncavx2_rdd_sum512d(ret, ret8);
#else // others
	double tmp[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < a->dim; i++)
	{
		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
		rdd_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_ddvector(DDVector c, DDVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
	}
#else // others
	for(i = 0; i < a->dim; i++)
		set_ddvector_i(c, i, get_ddvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_ddvector(DDVector c, DDVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[DDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rdd_neg(tmp, get_ddvector_i(a, i));
		//set_ddvector_i(c, i, tmp);
		tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][i])));
		_mm256_store_pd(&(c->element[0][i]), tmp[0]);
		_mm256_store_pd(&(c->element[1][i]), tmp[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[DDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rdd_neg(tmp, get_ddvector_i(a, i));
		//set_ddvector_i(c, i, tmp);
		tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][i])));
		_mm512_store_pd(&(c->element[0][i]), tmp[0]);
		_mm512_store_pd(&(c->element[1][i]), tmp[1]);
	}
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rdd_neg(tmp, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_ddvector(double ret[DDSIZE], DDVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(a->element[0][index]));
		vec4[1] = _mm256_load_pd(&(a->element[1][index]));

		//rdd_abs(tmp, get_ddvector_i(a, i));
		//rdd_add(ret, ret, tmp);
		_bncavx2_rdd_abs(tmp4, vec4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rdd_abssum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(a->element[0][index]));
		vec8[1] = _mm512_load_pd(&(a->element[1][index]));

		//rdd_abs(tmp, get_ddvector_i(a, i));
		//rdd_add(ret, ret, tmp);
		_bncavx512_rdd_abs(tmp8, vec8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	_bncavx512_rdd_abssum512d(ret, ret8);
#else // others
	double tmp[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < a->dim; i++)
	{
		rdd_abs(tmp, get_ddvector_i(a, i));
		rdd_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_ddvector(double ret[DDSIZE], DDVector a)
{
	double tmp[DDSIZE];
	long int i;

	rdd_abs(ret, get_ddvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rdd_abs(tmp, get_ddvector_i(a, i));
		if(rdd_cmp(ret, tmp) < 0)
			rdd_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_ddvector(double ret[DDSIZE], DDVector vec)
{
	long int i, index, dim;

	dim = vec->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(vec->element[0][index]));
		vec4[1] = _mm256_load_pd(&(vec->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(vec, i), get_ddvector_i(vec, i));
//		rdd_add(ret, ret, tmp);
		_bncavx2_rdd_mul(tmp4, vec4, vec4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rdd_norm256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(vec->element[0][index]));
		vec8[1] = _mm512_load_pd(&(vec->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx512_rdd_mul(tmp8, vec8, vec8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	_bncavx512_rdd_norm512d(ret, ret8);
#else // others
	double tmp[DDSIZE];

	//c_dd_copy_d((double)0.0, tmp);
	//c_dd_copy_d((double)0.0, ret);
	rdd_set0(tmp);
	rdd_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_dd_sqr(GET_DDVECTOR_I(vec, i), tmp);
		//c_dd_add(tmp, ret, ret);
		rdd_mul(tmp, get_ddvector_i(vec, i), get_ddvector_i(vec, i));
		rdd_add(ret, ret, tmp);
	}

	//c_dd_sqrt(ret, tmp);
	//c_dd_copy(tmp, ret);
	rdd_sqrt(tmp, ret);
	rdd_set(ret, tmp);
#endif // __AVX2__
}

#if defined(__AVX2__)
/* double-double = double + double */
void _bncavx2_ddadd_d_d(ddfloat ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_dtwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        //_mm256_store_pd(&ret[index].val[0], s);
        //_mm256_store_pd(&ret[index].val[1], e);

        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
    }

   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index].val[0] = in_s[0]; ret[index].val[1] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        }
    }
//    printf("\n");
}

/* double-double = double + double */
//void _bncavx2_ddvadd_d_d(ddvector *ret, double a[], double b[], int dim)
void _bncavx2_ddvadd_d_d(DDVector ret, double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_dtwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_pd(&ret->element[0][index], s);
        _mm256_store_pd(&ret->element[1][index], e);
        /*
        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
    }

   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index] = in_s[0]; ret->element[1][index] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
            ret->element[0][index + 2] = in_s[2]; ret->element[1][index + 2] = in_e[2];
        }
    }
//    printf("\n");
}

/* add */
void _bncavx2_ddvadd(DDVector ret, DDVector a, DDVector b, int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2], s1, s2, t1, t2;
    double in_s1[4], in_s2[4], in_t1[4], in_t2[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
   }
}


void _bncavx2_ddadd(ddfloat ret[], ddfloat a[], ddfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_pd(
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_pd(
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_b_val[0] = _mm256_set_pd(
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_pd(
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
   }

}


/* double-double = double * double */
void _bncavx2_ddvmul_d_d(DDVector ret, double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_prod(a, b, &e);
        s = _bncavx2_dtwo_prod(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_pd(&ret->element[0][index], s);
        _mm256_store_pd(&ret->element[1][index], e);
        /*
        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
    }

   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index] = in_s[0]; ret->element[1][index] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
            ret->element[0][index + 2] = in_s[2]; ret->element[1][index + 2] = in_e[2];
        }
    }
//    printf("\n");
}

/* ddmul */
void _bncavx2_ddmul(ddfloat ret[], ddfloat a[], ddfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_pd(
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_pd(
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_b_val[0] = _mm256_set_pd(
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_pd(
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );

        _bncavx2_rdd_mul(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
   }

}

/* ddvmul */
void _bncavx2_ddvmul(DDVector ret, DDVector a, DDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_mul(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
   }
}
/* dddiv */
void _bncavx2_dddiv(ddfloat ret[], ddfloat a[], ddfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_pd(
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_pd(
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_b_val[0] = _mm256_set_pd(
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_pd(
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );

        _bncavx2_rdd_div(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
   }

}

/* div */
void _bncavx2_ddvdiv(DDVector ret, DDVector a, DDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_div(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
   }
}

#endif // __AVX2__

// ddmatmul_ddvec
void ddmatmul_ddvec(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    ddfloat tmp_mul, aik, bkj, cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rdd_set_ui(cij.val, 0UL);
            for(k = 0; k < mid_dim; k++)
            {
                aik.val[0] = mat_a->element[0][i * mid_dim + k];
                aik.val[1] = mat_a->element[1][i * mid_dim + k];
                bkj.val[0] = mat_b->element[0][k * col_dim + j];
                bkj.val[1] = mat_b->element[1][k * col_dim + j];

                rdd_mul(tmp_mul.val, aik.val, bkj.val);
                rdd_add(cij.val, cij.val, tmp_mul.val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
        }
    }
}


// ddmatmul_ddvec_ur4
void ddmatmul_ddvec_ur4(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
    ddfloat tmp_mul[4], aik[4], bkj[4], cij;

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            rdd_set_ui(cij.val, 0UL);
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

                bkj[0].val[0] = mat_b->element[0][k * col_dim + j];
                bkj[1].val[0] = mat_b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = mat_b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = mat_b->element[0][(k + 3) * col_dim + j];

                bkj[0].val[1] = mat_b->element[1][k * col_dim + j];
                bkj[1].val[1] = mat_b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = mat_b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = mat_b->element[1][(k + 3) * col_dim + j];

                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);

                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            }
            ret->element[0][i * col_dim + j] = cij.val[0];
            ret->element[1][i * col_dim + j] = cij.val[1];
        }
    }
}

// ddmatmul_ddvec_avx2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void ddmatmul_ddvec_avx2(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim)
{
    int i, j, k;
//    ddfloat tmp_mul[4], aik[4], bkj[4], cij;
    double cijval[4][DDSIZE];
    __m256d cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp_mul[DDSIZE];

    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
            for(k = 0; k < mid_dim; k += 4)
            {
            /*
                aik[0].val[0] = mat_a->element[0][i * mid_dim + k];
                aik[1].val[0] = mat_a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = mat_a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = mat_a->element[0][i * mid_dim + k + 3];
            */
            //    aik[0] = _mm256_load_pd(&(mat_a->element[0][i * mid_dim + k]));
                
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
            //    aik[1] = _mm256_load_pd(&(mat_a->element[1][i * mid_dim + k]));
                
                aik[1] = _mm256_set_pd(
                    mat_a->element[1][i * mid_dim + k],
                    mat_a->element[1][i * mid_dim + k + 1],
                    mat_a->element[1][i * mid_dim + k + 2],
                    mat_a->element[1][i * mid_dim + k + 3]
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
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rdd_mul(tmp_mul, aik, bkj);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rdd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            rdd_add(cijval[0], cijval[0], cijval[1]);
            rdd_add(cijval[0], cijval[0], cijval[2]);
            rdd_add(cijval[0], cijval[0], cijval[3]);

            ret->element[0][i * col_dim + j] = cijval[0][0];
            ret->element[1][i * col_dim + j] = cijval[0][1];
        }
    }
}
#endif // AVX2

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void set0_ddmatrix(DDMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&mat->element[0][i], zero4);
		_mm256_store_pd(&mat->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&mat->element[0][i], zero8);
		_mm512_store_pd(&mat->element[1][i], zero8);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize ddvector
DDMatrix init_ddmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	DDMatrix ret = NULL;

	ret = (DDMatrix)BNC_MALLOC(sizeof(ddmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_row_dim = (long int)ceil((double)(row_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_col_dim = (long int)ceil((double)(col_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_ddmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
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

	//printf("init_ddmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&ret->element[0][i], zero4);
		_mm256_store_pd(&ret->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&ret->element[0][i], zero8);
		_mm512_store_pd(&ret->element[1][i], zero8);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free ddvector
void free_ddmatrix(DDMatrix mat)
{
	long int i;

	for(i = 0; i < DDSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print ddvector
void print_ddmatrix(DDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * DDSIZE));
			c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// DDMatrix mat -> ddfloat array
void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, DDMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DDSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// ddfloat array -> DDmatrix ret
void set_ddmatrix_ddfloat(DDMatrix ret, ddfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DDSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}


// matrix multiplication
// ret := A * B
void mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[4][DDSIZE];
    __m256d cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp_mul[DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
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

            /*
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rdd_mul(tmp_mul, aik, bkj);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rdd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            rdd_add(cijval[0], cijval[0], cijval[1]);
            rdd_add(cijval[0], cijval[0], cijval[2]);
            rdd_add(cijval[0], cijval[0], cijval[3]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[8][DDSIZE];
    __m512d cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp_mul[DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_pd();
            cij[1] = _mm512_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            //    aik[0] = _mm256_load_pd(&(a->element[0][i * mid_dim + k]));
                
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
            
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
            //    aik[1] = _mm256_load_pd(&(a->element[1][i * mid_dim + k]));
                
                aik[1] = _mm512_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3],
                    a->element[1][i * real_mid_dim + k + 4],
                    a->element[1][i * real_mid_dim + k + 5],
                    a->element[1][i * real_mid_dim + k + 6],
                    a->element[1][i * real_mid_dim + k + 7]
                );
                
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm512_set_pd(
                    b->element[0][ k      * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 7) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm512_set_pd(
                    b->element[1][ k      * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j]
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 7) * real_col_dim + j]
                );

            /*
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rdd_mul(tmp_mul, aik, bkj);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rdd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7];
            rdd_add(cijval[0], cijval[0], cijval[1]);
            rdd_add(cijval[0], cijval[0], cijval[2]);
            rdd_add(cijval[0], cijval[0], cijval[3]);
            rdd_add(cijval[0], cijval[0], cijval[4]);
            rdd_add(cijval[0], cijval[0], cijval[5]);
            rdd_add(cijval[0], cijval[0], cijval[6]);
            rdd_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }
#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	double tmp[DDSIZE], ret_ij[DDSIZE];

	//printf("Non SIMD mul_ddmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rdd_set0(GET_DDMATRIX_IJ(ret, i, j));
			rdd_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rdd_mul(tmp, GET_DDMATRIX_IJ(a, i, k), GET_DDMATRIX_IJ(b, k, j));
				//rdd_add(GET_DDMATRIX_IJ(ret, i, j), tmp, GET_DDMATRIX_IJ(ret, i, j));
				rdd_add(ret_ij, tmp, ret_ij);
			}
			set_ddmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[DDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d ret4[DDSIZE], mat4[DDSIZE], tmp4[DDSIZE];

	ret4[0] = _mm256_setzero_pd();
	ret4[1] = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat4[0] = _mm256_load_pd(&(mat->element[0][i]));
		mat4[1] = _mm256_load_pd(&(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rdd_mul(tmp4, mat4, mat4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}

	_bncavx2_rdd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d ret8[DDSIZE], mat8[DDSIZE], tmp8[DDSIZE];

	ret8[0] = _mm512_setzero_pd();
	ret8[1] = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat8[0] = _mm512_load_pd(&(mat->element[0][i]));
		mat8[1] = _mm512_load_pd(&(mat->element[1][i]));

		// tmp8 := mat8[i]^2
		// ret8 += tmp8
		_bncavx512_rdd_mul(tmp8, mat8, mat8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}

	_bncavx2_rdd_sum512d(ret, ret8);

#else // others
	double mat1[DDSIZE];

	rdd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rdd_mul(tmp, mat1, mat1);
		rdd_add(ret, ret, tmp);
	}

#endif // __AVX2__

	rdd_sqrt(tmp, ret);
	rdd_set(ret, tmp);

}

// print normf
void print_normf_ddmatrix(const char *str, DDMatrix mat)
{
	static double tmp[DDSIZE];

	normf_ddmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rdd_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for DDMatrix            */
/*
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
void transpose_ddmatrix(DDMatrix c, DDMatrix a);
void inv_ddmatrix(DDMatrix a);
void subst_mpfmatrux(DDMatrix c, DDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i, j;
	double tmp[DDSIZE], sum[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_dd(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rdd_abs(tmp, get_ddmatrix_ij(mat, i, j));
			rdd_add(sum, sum, tmp);
		}
		if(rdd_cmp(ret, sum) < 0)
			rdd_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i, j;
	double tmp[DDSIZE], sum[DDSIZE];

	rdd_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rdd_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rdd_abs(tmp, get_ddmatrix_ij(mat, i, j));
			rdd_add(sum, sum, tmp);
		}
		if(rdd_cmp(ret, sum) < 0)
			rdd_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], aij4[DDSIZE], bij4[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_add(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], bij8[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_add(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}
#else // others
	double tmp[DDSIZE], aij[DDSIZE], bij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rdd_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	double tmp[DDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_add(tmp, get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	double tmp[DDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_sub(tmp, get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], aij4[DDSIZE], bij4[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_sub(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], bij8[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_sub(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}
#else // others
	double tmp[DDSIZE], aij[DDSIZE], bij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rdd_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], sc4[DDSIZE], aij4[DDSIZE];

	sc4[0] = _mm256_set1_pd(sc[0]);
	sc4[1] = _mm256_set1_pd(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(tmp4, sc4, aij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], sc8[DDSIZE];

	sc8[0] = _mm512_set1_pd(sc[0]);
	sc8[1] = _mm512_set1_pd(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_add(tmp8, sc8, aij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}
#else // others
	double tmp[DDSIZE], aij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];

		rdd_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	double tmp[DDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_mul(tmp, sc, get_ddmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_ddmatrix(DDMatrix c, DDMatrix a)
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
	__m256d aji4[DDSIZE];

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
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), aji4[0]);
			_mm256_store_pd(&(c->element[1][index]), aji4[1]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d aji8[DDSIZE];

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
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), aji8[0]);
			_mm512_store_pd(&(c->element[1][index]), aji8[1]);
		}
	}
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_ddmatrix(DDMatrix c, DDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix\n");
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
			_mm256_store_pd(&(c->element[0][i * real_col_dim + j]), _mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][i * real_col_dim + j]), _mm256_load_pd(&(a->element[1][index])));
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
			_mm512_store_pd(&(c->element[0][i * real_col_dim + j]), _mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][i * real_col_dim + j]), _mm512_load_pd(&(a->element[1][index])));
		}
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_ddmatrix(DDMatrix c)
{
	long int i, j;
	long int real_total_dim;
	double tmp1[DDSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&c->element[0][i], zero4);
		_mm256_store_pd(&c->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&c->element[0][i], zero8);
		_mm512_store_pd(&c->element[1][i], zero8);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
	}
#endif // __AVX2__

	rdd_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_ddmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	double tmp[DDSIZE], tmp1[DDSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrix_ddvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256d tmp4[DDSIZE], tmp1_4[DDSIZE];
	__m256d aij4[DDSIZE], vbj4[DDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_pd(&(a->element[1][ij_index]));
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rdd_add(tmp4, tmp4, tmp1_4);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx2_rdd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[DDSIZE], tmp1_8[DDSIZE];
	__m512d aij8[DDSIZE], vbj8[DDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[1] = _mm512_load_pd(&(a->element[1][ij_index]));
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rdd_add(tmp8, tmp8, tmp1_8);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rdd_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			rdd_add(tmp, tmp, tmp1);
		}
		set_ddvector_i(v, i, tmp);
	}
#endif // __AVX2__

}

/* v := a^T * vb */
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	double tmp[DDSIZE], tmp1[DDSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrixt_ddvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256d tmp4[DDSIZE], tmp1_4[DDSIZE];
	__m256d aij4[DDSIZE], vbj4[DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4);
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
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rdd_add(tmp4, tmp4, tmp1_4);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx2_rdd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[DDSIZE], tmp1_8[DDSIZE];
	__m512d aij8[DDSIZE], vbj8[DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp4);
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
				a->element[1][(j + 7) * real_col_dim + i)],
				a->element[1][(j + 6) * real_col_dim + i)],
				a->element[1][(j + 5) * real_col_dim + i)],
				a->element[1][(j + 4) * real_col_dim + i)],
				a->element[1][(j + 3) * real_col_dim + i)],
				a->element[1][(j + 2) * real_col_dim + i)],
				a->element[1][(j + 1) * real_col_dim + i)],
				a->element[1][(j    ) * real_col_dim + i)]
			);
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rdd_add(tmp8, tmp8, tmp1_8);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#else // others

	for(i = 0; i < a->col_dim; i++)
	{
		set0_dd(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rdd_mul(tmp1, get_ddmatrix_ij(a, j, i), get_ddvector_i(vb, j));
			rdd_add(tmp, tmp, tmp1);
		}
		set_ddvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_ddmatrix(DDMatrix a)
{
	long int i, j, k, dim;
	double tmp[DDSIZE], aii[DDSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_ddmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rdd_cmp_ui(get_ddmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_ddmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rdd_ui_div(aii, 1UL, get_ddmatrix_ij(a, i, i));
		set_ddmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rdd_mul(tmp, get_ddmatrix_ij(a, i, j), aii);
			set_ddmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rdd_mul(tmp, get_ddmatrix_ij(a, i, j), aii);
			set_ddmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rdd_neg(tmp, aii); /* tmp := -aii */
			rdd_mul(tmp, tmp, get_ddmatrix_ij(a, j, i));
			set_ddmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rdd_neg(tmp, aii); /* tmp := -aii */
			rdd_mul(tmp, tmp, get_ddmatrix_ij(a, j, i));
			set_ddmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_ddvec(MPFVector c, DDVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_dd(tmp, get_ddvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_ddvector_mpfvec(DDVector c, MPFVector a)
{
	long int i;
	double tmp[DDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_dd(tmp, get_mpfvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_ddmat(MPFMatrix c, DDMatrix a)
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
			mpf_set_dd(tmp, get_ddmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_ddmatrix_mpfmat(DDMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp[DDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_dd(tmp, get_mpfmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_ddvector_mpfvec(double relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_ddvec(mpf_approx_vec, approx_vec);

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
void relerr_element_ddvector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_ddvec(mpf_approx_vec, approx_vec);

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
	mpf_get_dd(max_relerr, mpf_max_relerr);
	mpf_get_dd(min_relerr, mpf_min_relerr);
	mpf_get_dd(norm_relerr, mpf_norm_relerr);

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

/* c := (dd)a */
void subst_ddvector_dvec(DDVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_ddvector_i_d(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_dvector_ddvec(DVector c, DDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rdd_get_d(get_ddvector_i(a, i));
	}
}


/* c := (dd)a */
void subst_ddmatrix_dmat(DDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij_d(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_dmatrix_ddmat(DMatrix c, DDMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (c->col_dim) + j;
			c->element[ij_index] = rdd_get_d(get_ddmatrix_ij(a, i, j));
		}
	}
}

/* Normwise relative error of vector */
void relerr_ddvector(double relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type)
{
	double norm_true_vec[DDSIZE], norm_diff_vec[DDSIZE];
	DDVector diff_vec;

	diff_vec = init_ddvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_ddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ddvector(norm_diff_vec, diff_vec);
			normi_ddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ddvector(norm_diff_vec, diff_vec);
			norm1_ddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ddvector(norm_diff_vec, diff_vec);
			norm2_ddvector(norm_true_vec, true_vec);
			break;
	}

	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(relerr, norm_diff_vec, norm_true_vec);

	free_ddvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_ddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type)
{
	double abs_true_vec[DDSIZE], abs_diff_vec[DDSIZE], norm_diff_vec[DDSIZE], norm_true_vec[DDSIZE];
	long int i;
	DDVector diff_vec;

	diff_vec = init_ddvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_ddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ddvector(norm_diff_vec, diff_vec);
			normi_ddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ddvector(norm_diff_vec, diff_vec);
			norm1_ddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ddvector(norm_diff_vec, diff_vec);
			norm2_ddvector(norm_true_vec, true_vec);
			break;
	}

	rdd_set(norm_relerr, norm_diff_vec);
	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rdd_set_ui(max_relerr, 0UL);
	normi_ddvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rdd_abs(abs_diff_vec, get_ddvector_i(diff_vec, i));
		rdd_abs(abs_true_vec, get_ddvector_i(true_vec, i));
		if(rdd_cmp_ui(abs_true_vec, 0UL) != 0)
			rdd_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rdd_cmp(max_relerr, abs_diff_vec) < 0)
			rdd_set(max_relerr, abs_diff_vec);
		if(rdd_cmp(min_relerr, abs_diff_vec) > 0)
			rdd_set(min_relerr, abs_diff_vec);
	}

	free_ddvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[DDSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
}

#ifdef __cplusplus
} // extern "C"
#endif
