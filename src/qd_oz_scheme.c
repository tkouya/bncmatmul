/********************************************************************************/
/* qd_oz_scheme: Multiple precision linear computation based on Ozaki scheme.   */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
#include "rdd.h" // [dtq]dfloat and its arithmetic r[dtq]d_*
#include "matmul_strassen.h"

// log2(x) := log10(x) / log10(2)
//#define DLOG2(x) (log10((x)) / 0.30102999566398119521373889472449)

// absmax_qdvector
void absmax_qdvector(double ret[QDSIZE], long int *max_index, QDVector vec)
{
    long int i, max_i, dim = vec->dim;
    double abs_val[QDSIZE];

    max_i = 0;
    set0_qd(ret);
    for(i = 0; i < dim; i++)
    {
        //abs_val = fabs(get_qdvector_i(vec, i));
        rqd_abs(abs_val, get_qdvector_i(vec, i));
        if(rqd_cmp(ret, abs_val) < 0)
        {
            rqd_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return;
}

/* c = a + (double)b */
void add_qdvector_dvec(QDVector c, QDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_qdvector_dvec\n");
		return;
	}

	double tmp[QDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqd_add_d(tmp, get_qdvector_i(a, i), get_dvector_i(b, i));
		set_qdvector_i(c, i, tmp);
	}
}

/* c = a - (double)b */
void sub_qdvector_dvec(QDVector c, QDVector a, DVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_qdvector_dvec\n");
		return;
	}

	double tmp[QDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqd_sub_d(tmp, get_qdvector_i(a, i),  get_dvector_i(b, i));
		set_qdvector_i(c, i, tmp);
	}
}

/* c := (d)a */
void subst_qdvector_dvec(QDVector c, DVector a)
{
	long int i, j;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: subst_qdvector_dvec\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_qdvector_i_d(c, i, get_dvector_i(a, i));
}

/* c := (qd)a */
void subst_dvector_qdvec(DVector c, QDVector a)
{
	long int i, j;
    qdfloat tmp;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: subst_dvector_qdvec\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
        tmp = get_qdvector_i_qdfloat(a, i);
		set_dvector_i(c, i, tmp.val[0]);
	}
}
// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_qdvector(DVector ret_vec[], int num_div, QDVector org_vec, int num_bits)
int split_qdvector_dvec(DVector ret_vec[], int num_div, QDVector org_vec) //, int num_bits)
{
    long int dim = org_vec->dim;
    int index, real_num_div, num_bits = 53; // IEEE754 binary64
    long int i;
    double org_vec_i, ret_high_vec_i, tmp[QDSIZE];
    double absmax_org_vec, threshold, t_exp; 
    QDVector tmp_org_vec;

    // tmp_org_vec := org_vec
    tmp_org_vec = init_qdvector(dim);
    subst_qdvector(tmp_org_vec, org_vec);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        subst_dvector_qdvec(ret_vec[index], tmp_org_vec);
        absmax_org_vec = absmax_dvector(NULL, ret_vec[index]);
    
        // ret_vec[index] == 0
        if(absmax_org_vec == 0.0) break;

        // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
        t_exp = ceil(DLOG2(absmax_org_vec)) + ceil(((double)num_bits + DLOG2((double)(dim))) / 2.0);
        
        //rqd_pow(threshold, two, t_exp);
        //rqd_pow_mpfr(threshold, two, t_exp);
        threshold = pow(2.0, t_exp);

        for(i = 0; i < dim; i++)
        {
            // set high vector
            //rqd_set(org_vec_i, get_qdvector_i(tmp_org_vec, i)); 
            org_vec_i = get_dvector_i(ret_vec[index], i);   
            ret_high_vec_i = org_vec_i + threshold;
            //rqd_add(ret_high_vec_i, org_vec_i, threshold);
            ret_high_vec_i -= threshold;
            //rqd_sub(ret_high_vec_i, ret_high_vec_i, threshold);
            set_dvector_i(ret_vec[index], i, ret_high_vec_i);

            // set low vector
            rqd_sub_d(tmp, get_qdvector_i(tmp_org_vec, i), ret_high_vec_i);           
            set_qdvector_i(tmp_org_vec, i, tmp);
        }

        real_num_div = index + 1;
    }

    free_qdvector(tmp_org_vec);

    return real_num_div;
}

// absmax_row_qdmatrix
void absmax_row_qdmatrix(double mu[QDSIZE], long int *max_j, long int row_index, QDMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[QDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
    rqd_abs(mu, get_qdmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		rqd_abs(abs_aij, get_qdmatrix_ij(mat, row_index, j));
		if(rqd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rqd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    //return mu;
    return;
}

/* c := a + (doble)b */
void add_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;
	//real_total_dim = row_dim * real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[QDSIZE], aij4[QDSIZE], bij4;//[QDSIZE];

    //bij4[0] = _mm256_setzero_pd();
    //bij4[1] = _mm256_setzero_pd();
    //bij4[2] = _mm256_setzero_pd();
    //bij4[3] = _mm256_setzero_pd();
    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        //index = i * real_col_dim + j;
        aij4[0] = _mm256_load_pd(&(a->element[0][index]));
        aij4[1] = _mm256_load_pd(&(a->element[1][index]));
        aij4[2] = _mm256_load_pd(&(a->element[2][index]));
        aij4[3] = _mm256_load_pd(&(a->element[3][index]));
        //bij4[0] = _mm256_load_pd(&(b->element[index]));
        //bij4[0] = _mm256_load_pd(&(get_dmatrix_ij(b, (index / real_col_dim), (index % real_col_dim))));
        //bij4 = _mm256_load_pd(&(b->element[index]));
        bij4 = _mm256_load_pd(&(get_dmatrix_ij(b, (index / real_col_dim), (index % real_col_dim))));

        _bncavx2_rqd_add_d(tmp4, aij4, bij4);
        //_bncavx2_rqd_add(tmp4, aij4, bij4);

        _mm256_store_pd(&(c->element[0][index]), tmp4[0]);
        _mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
        _mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
        _mm256_store_pd(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[QDSIZE], aij8[QDSIZE], bij8;

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[3] = _mm512_load_pd(&(a->element[3][index]));
		bij8 = _mm512_load_pd(&(b->element[index]));

		_bncavx512_rqd_add_d(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]);
		_mm512_store_pd(&(c->element[3][index]), tmp8[3]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t tmp_0, tmp_1, tmp_2, tmp_3, aij_0, aij_1, aij_2, aij_3, bij;
	for(index = 0; index < real_total_dim; index += (long int)svcntd())
	{
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)real_total_dim);
		aij_0 = svld1_f64(pg, &(a->element[0][index]));
		aij_1 = svld1_f64(pg, &(a->element[1][index]));
		aij_2 = svld1_f64(pg, &(a->element[2][index]));
		aij_3 = svld1_f64(pg, &(a->element[3][index]));
		bij   = svld1_f64(pg, &(get_dmatrix_ij(b, (index / real_col_dim), (index % real_col_dim))));
		_bncsve2_rqd_add_d(svptrue_b64(), &tmp_0, &tmp_1, &tmp_2, &tmp_3,
		                   aij_0, aij_1, aij_2, aij_3, bij);
		svst1_f64(pg, &(c->element[0][index]), tmp_0);
		svst1_f64(pg, &(c->element[1][index]), tmp_1);
		svst1_f64(pg, &(c->element[2][index]), tmp_2);
		svst1_f64(pg, &(c->element[3][index]), tmp_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float64x2_t tmp2[QDSIZE], aij2[QDSIZE], bij2;
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij2[0] = vld1q_f64(&(a->element[0][index]));
		aij2[1] = vld1q_f64(&(a->element[1][index]));
		aij2[2] = vld1q_f64(&(a->element[2][index]));
		aij2[3] = vld1q_f64(&(a->element[3][index]));
		bij2 = vld1q_f64(&(get_dmatrix_ij(b, (index / real_col_dim), (index % real_col_dim))));
		_bncneon_rqd_add_d(tmp2, aij2, bij2);
		vst1q_f64(&(c->element[0][index]), tmp2[0]);
		vst1q_f64(&(c->element[1][index]), tmp2[1]);
		vst1q_f64(&(c->element[2][index]), tmp2[2]);
		vst1q_f64(&(c->element[3][index]), tmp2[3]);
	}
#else // others
	double tmp[QDSIZE], bij[QDSIZE] = {0.0, 0.0, 0.0, 0.0};
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
            //bij[0] = get_dmatrix_ij(b, i, j);
			rqd_add_d(tmp, get_qdmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
			//rqd_add(tmp, get_qdmatrix_ij(a, i, j), bij);
            set_qdmatrix_ij(c, i, j, tmp);
		}
	}
#endif // defined(__AVX2__)
}

/* c := a - (doble)b */
void sub_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_dmat\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qdmatrix_dmat\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

	double tmp[QDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqd_sub_d(tmp, get_qdmatrix_ij(a, i, j), get_dmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c := (d)a */
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

/* c := (qd)a */
void subst_dmatrix_qdmat(DMatrix c, QDMatrix a)
{
	long int i, j, real_row_dim, real_col_dim, real_total_dim;
    qdfloat tmp;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_qdmat\n");
		return;
	}

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
            tmp = get_qdmatrix_ij_qdfloat(a, i, j);
			set_dmatrix_ij(c, i, j, tmp.val[0]);
		}
	}
/*	__m256d aij4;;

    real_row_dim = c->real_row_dim;
    real_col_dim = c->real_col_dim;
    real_total_dim = real_row_dim * real_col_dim;
    memcpy((void *)(c->element), (void *)(a->element[0]), (size_t)(real_total_dim * sizeof(double)));
*/
    /*
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        aij4 = _mm256_load_pd(&(a->element[0][i]));
        //_mm256_store_pd(&(get_dmatrix_ij(c, (i / real_col_dim), (i % real_col_dim))), aij4);
        _mm256_store_pd(&(c->element[i]), aij4);
    }
    */

#elif defined(__AVX512F__) // __AVX512F__
	//__m512d tmp8[QDSIZE], aij8[QDSIZE], bij8;

    real_total_dim = c->real_row_dim * c->real_col_dim;
    //memcpy(c->element, a->element[0], real_total_dim * sizeof(double));

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, hi-limb copy)
    {
        long int index;
        real_total_dim = c->real_row_dim * c->real_col_dim;
        for(index = 0; index < real_total_dim; index += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)real_total_dim);
            svst1_f64(pg, &(c->element[index]), svld1_f64(pg, &(a->element[0][index])));
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon (hi-limb copy)
    {
        long int index;
        real_total_dim = c->real_row_dim * c->real_col_dim;
        for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
            vst1q_f64(&(c->element[index]), vld1q_f64(&(a->element[0][index])));
    }
#else // __AVX2__
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
            tmp = get_qdmatrix_ij_qdfloat(a, i, j);
			set_dmatrix_ij(c, i, j, tmp.val[0]);
		}
	}
#endif // __AVX2__
}

//#define SPLIT_NUM_DIGITS 64

// SplitMat_A
int split_qdmatrix_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat)
{
    int flag_stop = 0;
	long int i, j, index, row_dim, col_dim, real_total_dim;
	long int num_digits = 53; // Too little? IEEE double prec.
    int real_num_div;
	//double *s;
    QDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2]; 
	double mu, abs_aij, t_exp, t_exp_init, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_qdmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    //tmp_mat[0] = init_qdmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_qdmatrix(row_dim, col_dim);
    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_qdmatrix(row_dim, col_dim);
    subst_qdmatrix(tmp_org_mat, org_mat);

    t_exp_init = ceil(((double)num_digits + DLOG2((double)(col_dim))) / 2.0);
    //printf("t_exp = %25.17e -> ", t_exp_init);

    real_num_div = 0;
    //for(index = 0; index < num_div; index++)
    for(index = 0; index < num_div - 1; index++)
    {
        //printf("In split_qdmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
        subst_dmatrix_qdmat(ret_mat[index], tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_qdmatrix(mu, NULL, i, tmp_org_mat);
            mu = absmax_row_dmatrix(NULL, i, ret_mat[index]);
            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + t_exp_init;
            //printf("%d %25.17e\n", i, t_exp);
            //if(isnan(t_exp))
            //    flag_stop = 1;

            // s[i, j] = 2^t_exp
            //rqd_pow(power2, two, t_exp);
            //rqd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            for(j = 0; j < col_dim; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_qdmatrix_ij(s, i, j, power2);
                set_dmatrix_ij(s, i, j, power2);
            }
        }

        // ret_mat[index] == 0
        if(mu_total == 0.0) break;

        // split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_IMKL
        real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);

        // high_mat := tmp_mat - s
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_qdmatrix(tmp_mat[0], tmp_org_mat, s);
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        //sub_qdmatrix(tmp_mat[1], tmp_mat[0], s);
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);
#endif // USE_IMKL

        // low_mat := mat - high_mat
        sub_qdmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_qdmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // subst remaining part
    //if(flag_stop != 1)
    //    subst_dmatrix_qdmat(ret_mat[num_div - 1], tmp_org_mat);

	// free s
	//free_qdmatrix(s);
	free_dmatrix(s);
    free_qdmatrix(tmp_org_mat);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);

    return real_num_div;
}

// absmax_col_qdmatrix
void absmax_col_qdmatrix(double mu[QDSIZE], long int *max_i, long int col_index, QDMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[QDSIZE];

	//mu = fabs(mat[0 * col_dim + j]);
    rqd_abs(mu, get_qdmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		rqd_abs(abs_aij, get_qdmatrix_ij(mat, i, col_index));
		if(rqd_cmp(abs_aij, mu) > 0)
        {
			rqd_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    return;
}

// SplitMat_B
int split_qdmatrix_t_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div;
    long int num_digits = 53; // Too litte? IEEE double prec.
    //long int num_digits = SPLIT_NUM_DIGITS;
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    QDMatrix tmp_org_mat;
    DMatrix s, tmp_mat[2];
	//double mu[QDSIZE], abs_aij[QDSIZE], t_exp[QDSIZE], power2[QDSIZE], two[QDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;

	// initialize s
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_qdmatrix(row_dim, col_dim);
    s = init_dmatrix(row_dim, col_dim);

    tmp_mat[0] = init_dmatrix(row_dim, col_dim);
    tmp_mat[1] = init_dmatrix(row_dim, col_dim);

    tmp_org_mat = init_qdmatrix(row_dim, col_dim);
    subst_qdmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;

    //for(index = 0; index < num_div; index++)
    for(index = 0; index < num_div - 1; index++)
    {
        subst_dmatrix_qdmat(ret_mat[index], tmp_org_mat);
        // mu[j] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
        for(j = 0; j < col_dim; j++)
        {
            /* mu = fabs(mat[0 * col_dim + j]);
            for(i = 1; i < row_dim; i++)
            {
                abs_aij = fabs(mat[i * col_dim + j]);
                if(abs_aij > mu)
                    mu = abs_aij;
            }
            */
            mu = absmax_col_dmatrix(NULL, j, ret_mat[index]);
            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim))) / 2.0);
            //if(isnan(t_exp))
            //    flag_stop = 1;

            // s[i, j] = 2^t_exp
            //rqd_pow(power2, two, t_exp);
            //rqd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_dmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        if(mu_total == 0.0) break;

#ifdef USE_IMKL
        real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);

        // high_mat := tmp_mat - s
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        add_dmatrix(tmp_mat[0], ret_mat[index], s);

        // high_mat := tmp_mat - s
        sub_dmatrix(tmp_mat[1], tmp_mat[0], s);
        subst_dmatrix(ret_mat[index], tmp_mat[1]);
#endif // USE_IMKL

        // low_mat := mat - high_mat
        sub_qdmatrix_dmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        //sub_qdmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // subst remaining part
    //if(flag_stop != 1)
    //    subst_dmatrix_qdmat(ret_mat[num_div - 1], tmp_org_mat);

    // free s
	free_dmatrix(s);
    free_dmatrix(tmp_mat[0]);
    free_dmatrix(tmp_mat[1]);
    free_qdmatrix(tmp_org_mat);

    return real_num_div;
}

// Matrix multiplication based on Ozaki scheme
void mul_qdmatrix_oz(QDMatrix ret, QDMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, div_ret;
    QDMatrix tmp_ret;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_qdmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    tmp_ret = init_qdmatrix(row_dim, col_dim);

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, mid_dim);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    div_ret = init_dmatrix(row_dim, col_dim);

    real_num_div_a = split_qdmatrix_dmat(div_a, max_num_div_a, a);
    //printf("split_qdmatrix_dmat  ->%d\n", real_num_div_a);
    real_num_div_b = split_qdmatrix_t_dmat(div_b, max_num_div_b, b);
    //printf("split_qdmatrix_t_dmat->%d\n", real_num_div_b);

    set0_qdmatrix(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
        //for(j = 0; j < real_num_div_b; j++)
        for(j = 0; j < real_num_div_b - i; j++)
        {
#ifdef USE_IMKL
            set0_dmatrix(div_ret);
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                div_a[i]->real_row_dim, // m
                div_b[j]->real_col_dim, // n
                div_a[i]->real_col_dim, // k
                1.0,
                div_a[i]->element,
                div_a[i]->real_col_dim, // k
                div_b[j]->element,
                div_b[j]->real_col_dim, // n
                1.0,
                div_ret->element,
                div_ret->real_col_dim   // n
            );
            //mul_dmatrix(div_ret, div_a[i], div_b[j]);
#else // USE_IMKL
            //printf("%d %d->%15.7e, %15.7e\n", i, j, normf_dmatrix(div_a[i]), normf_dmatrix(div_b[j]));
            mul_dmatrix(div_ret, div_a[i], div_b[j]);
#endif // USE_IMKL
            //subst_qdmatrix_dmat(tmp_ret, div_ret);
            add_qdmatrix_dmat(ret, ret, div_ret);
            //add_qdmatrix(ret, ret, tmp_ret);
        }
    }

    free_dmatrix(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);

    free_qdmatrix(tmp_ret);
}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_qdmatrix_qdvec_oz(QDVector ret, QDMatrix a, int max_num_div_a, QDVector vb, int max_num_div_vb)
{
    int i, j;
    int real_num_div_a, real_num_div_vb;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DMatrix *div_a;
    DVector *div_vb, div_ret;

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_dmatrix(row_dim, col_dim);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);
    div_ret = init_dvector(vec_dim);

    real_num_div_a = split_qdmatrix_dmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_qdvector_dvec(div_vb, max_num_div_vb, vb);

    set0_qdvector(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
        for(j = 0; j < real_num_div_vb; j++)
        {

#ifdef USE_IMKL
            set0_dvector(div_ret);
            cblas_dgemv(
                CblasRowMajor,
                CblasNoTrans,
                div_a[i]->real_row_dim,
                div_a[i]->real_col_dim,
                1.0,
                div_a[i]->element,
                div_a[i]->real_row_dim,
                div_vb[j]->element,
                1,
                1.0,
                div_ret->element,
                1
            );
#else // USE_IMKL
            mul_dmatrix_dvec(div_ret, div_a[i], div_vb[j]);
#endif // USE_IMKL

            add_qdvector_dvec(ret, ret, div_ret);
       }
    }

    free_dvector(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_dmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);

}

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, a->re->col_dim);
    t4 = init_qdmatrix(b->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_qdmatrix(ret->im, t1, t2);
    #else // USE_4M
    */
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_b_image);
    sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_b_image);
        add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */  

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}


// Testing
#ifdef DEBUG

//#define DIM 3
//#define DIM 10
//#define DIM 50
#define DIM 256

//#define MAX_NUM_DIV 2
#define MAX_NUM_DIV 10
//#define MAX_NUM_DIV 20

int main()
{
    long int i, j;
    int real_num_div;
    double tmp[QDSIZE], tmp2[QDSIZE];
    QDVector vec_org, vec, vec_c, vec_b;
    DVector dvec[MAX_NUM_DIV];
    QDMatrix mat_a, mat_b, mat_c, mat;
    DMatrix dmat[MAX_NUM_DIV];
    qdfloat ddtmp, ddtmp2;

    //goto qdmatrix;

// DVector 
    vec_org = init_qdvector(DIM);
    vec = init_qdvector(DIM);
    vec_b = init_qdvector(DIM);
    vec_c = init_qdvector(DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dvec[i] = init_dvector(DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        rqd_set_d(tmp, (double)rand());
        rqd_mul_d(tmp, tmp, (double)rand());
        rqd_div_d(tmp, tmp, (double)rand());
        set_qdvector_i(vec_org, i, tmp);
    }
    subst_qdvector(vec_b, vec_org);

    // split
    real_num_div = split_qdvector_dvec(dvec, MAX_NUM_DIV, vec_org); //, 53);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dvec[%d]:\n", i); print_dvector(dvec[i]);

        add_qdvector_dvec(vec, vec, dvec[i]);
    }
    //sub_qdvector(vec, vec, vec_org);
    //printf("vec - vec_org:\n"); print_qdvector(vec);
    //printf("\n");
    for(i = 0; i < DIM; i++)
    {
        ddtmp = get_qdvector_i_qdfloat(vec, i);
        ddtmp2 = get_qdvector_i_qdfloat(vec_org, i);
        printf("%5d %25.17e %25.17e %25.17e %25.17e\n%5d %25.17e %25.17e %25.17e %25.17e\n", i, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], ddtmp.val[3], i, ddtmp2.val[0], ddtmp2.val[1], ddtmp2.val[2], ddtmp2.val[3]);
    }
    
    //sgoto end;
// DMatrix
qdmatrix:

    mat_a = init_qdmatrix(DIM, DIM);
    mat_b = init_qdmatrix(DIM, DIM);
    mat_c = init_qdmatrix(DIM, DIM);
    mat = init_qdmatrix(DIM, DIM);
    for(i = 0; i < MAX_NUM_DIV; i++)
        dmat[i] = init_dmatrix(DIM, DIM);

    // set random
    srand(DIM);
    for(i = 0; i < DIM; i++)
    {
        for(j = 0; j < DIM; j++)
        {
            rqd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rqd_mul_d(tmp, tmp, (double)rand());
            rqd_div_d(tmp, tmp, (double)rand());
            set_qdmatrix_ij(mat_a, i, j, tmp);

            rqd_set_d(tmp, (double)rand() / (double)RAND_MAX);
            rqd_mul_d(tmp, tmp, (double)rand());
            rqd_div_d(tmp, tmp, (double)rand());
            set_qdmatrix_ij(mat_b, i, j, tmp);
        }
    }

    // split_A
    printf("Split_A:\n");
    real_num_div = split_qdmatrix_dmat(dmat, MAX_NUM_DIV, mat_a);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_qdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_qdmatrix(mat, mat, mat_a);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_a);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat);
    printf("\n"); 

    // split_B
    printf("Split_B:\n");
    real_num_div = split_qdmatrix_t_dmat(dmat, MAX_NUM_DIV, mat_b);
    printf("num_div, real_num_div = %d, %d\n", MAX_NUM_DIV, real_num_div);

    for(i = 0; i < real_num_div; i++)
    {
        //printf("dmat[%d]:\n", i); print_dmatrix(dmat[i]);

        add_qdmatrix_dmat(mat, mat, dmat[i]);
    }
    sub_qdmatrix(mat, mat, mat_b);
    printf("||mat - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_a);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat);
    printf("\n"); 

    // a * b
    mul_qdmatrix_oz(mat, mat_a, MAX_NUM_DIV, mat_b, MAX_NUM_DIV);
    //mul_qdmatrix(mat_c, mat_a, mat_b);
    mul_qdmatrix_strassen(mat_c, mat_a, mat_b, 32);
/*
    for(i = 0; i < mat->row_dim; i++)
    {
        for(j = 0; j < mat->col_dim; j++)
        {
            ddtmp = get_qdmatrix_ij_qdfloat(mat, i, j);
            ddtmp2 = get_qdmatrix_ij_qdfloat(mat_c, i, j);
            if(ddtmp.val[1] != ddtmp2.val[1])
                printf("%5d,%5d %25.17e %25.17e %25.17e %25.17e\n%5d,%5d %25.17e %25.17e %25.17e %25.17e\n", i, j, ddtmp.val[0], ddtmp.val[1], ddtmp.val[2], ddtmp.val[3], i, j, ddtmp2.val[0], ddtmp2.val[1], ddtmp2.val[2], ddtmp2.val[3]);
        }
    }
*/
    sub_qdmatrix(mat, mat, mat_c);
    //print_qdmatrix(mat);
    printf("div = %ld\n", MAX_NUM_DIV);
    printf("||mat_oz - mat_org|| / ||mat_org||:\n"); 
    normf_qdmatrix(tmp, mat);
    normf_qdmatrix(tmp2, mat_c);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat); 
    printf("\n"); 


    // a * vec
    mul_qdmatrix_qdvec_oz(vec, mat_a, MAX_NUM_DIV, vec_b, MAX_NUM_DIV);
    mul_qdmatrix_qdvec(vec_c, mat_a, vec_b);

    sub_qdvector(vec, vec, vec_c);
    //print_qdmatrix(mat);
    printf("div = %ld\n", MAX_NUM_DIV);
    printf("||vec_oz - vec_org|| / ||vec_org||:\n"); 
    norm2_qdvector(tmp, vec);
    norm2_qdvector(tmp2, vec_c);
    rqd_div(tmp, tmp, tmp2);
    rqd_out_str(tmp); //print_qdmatrix(mat); 
    printf("\n"); 

    free_qdmatrix(mat_a);
    free_qdmatrix(mat_b);
    free_qdmatrix(mat_c);
    free_qdmatrix(mat);
    for(i = 0; i < 10; i++)
        free_dmatrix(dmat[i]);

    free_qdvector(vec_org);
    free_qdvector(vec);
    for(i = 0; i < 10; i++)
        free_dvector(dvec[i]);

end:
    return 0;
}

// Testing
#endif // DEBUG
