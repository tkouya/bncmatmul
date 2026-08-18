/********************************************************************************/
/* bncomp_linear_mpf.c: Parallelized DD Precision Linear Computation Library    */
/*                                                                  with OpenMP */
/* Copyright (C) 2023 Tomonori Kouya                                            */
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
// BNCpack with OpenMP
#include "matmul_strassen.h"
#include "bncomp.h"

//---------------------------------------
// GMP & MPFR
//---------------------------------------
#ifdef USE_GMP

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_mpfvector(MPFVector c, MPFVector a, long int dim)
{
	long int i;//, dim;

//	dim = c->dim;

	#pragma omp parallel for
	for(i = 0; i < dim; i++)
		set_mpfvector_i(c, i, get_mpfvector_i(a, i));
}

/* c = a - b */
void _bncomp_sub_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim)
{
	long int i;//, dim;
	int thread_num, thread_index;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR:_bncomp_sub_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_sub(tmp[thread_index], get_mpfvector_i(a, i), get_mpfvector_i(b, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c = a + b */
void _bncomp_add_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim)
{
	long int i;//, dim;
	int thread_num, thread_index;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR:_bncomp_add_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_add(tmp[thread_index], get_mpfvector_i(a, i), get_mpfvector_i(b, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c = val * a */
void _bncomp_cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a, long int dim)
{
	long int i;//, dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_mul(tmp[thread_index], val, get_mpfvector_i(a, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* inner product of vector blocks */
void _bncomp_ip_mpfvector(mpf_t ret, MPFVector va, MPFVector vb)
{
	long int i, dim;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if(va->dim != vb->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_mpfvector: va->dim = %ld, vb->dim = %ld !\n", va->dim, vb->dim);
		return;
	}

	dim = va->dim;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL); // = 0.0;
#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_mul(tmp[thread_index], get_mpfvector_i(va, i), get_mpfvector_i(vb, i));

#pragma omp critical
		mpf_add(ret, ret, tmp[thread_index]);
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp[thread_index]);
}

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_endend_J)
void _bncomp_row_swap_mpfmatrix(MPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], mat->prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_set(tmp[thread_index], get_mpfmatrix_ij(mat, row_index0, i));
		set_mpfmatrix_ij(mat, row_index0, i, get_mpfmatrix_ij(mat, row_index1, i));
		set_mpfmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_set_ui(tmp[thread_index], 0UL);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_add(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c := a - b */
void _bncomp_sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_sub(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c := sc * a */
void _bncomp_cmul_mpfmatrix(MPFMatrix c, mpf_t sc, MPFMatrix a)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], in_sc[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(in_sc[thread_index], prec);
		mpf_set(in_sc[thread_index], sc);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_mul(tmp[thread_index], in_sc[thread_index], get_mpfmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(in_sc[thread_index]);
	}
}


/* c = a * b */
void _bncomp_mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, k, row_dim, col_dim, mid_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrix\n");
		return;
	}

	prec = prec_mpfmatrix(c);

	row_dim = c->row_dim;
	col_dim = c->col_dim;
	mid_dim = a->col_dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_set_ui(tmp[thread_index], 0UL);
			for(k = 0; k < mid_dim; k++)
			{
#ifndef USE_MPFR
				mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j));
				mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
#else
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j), tmp[thread_index], bnc_default_rounding_mode);
				mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j), tmp[thread_index], MPFR_RNDN);
#endif
			}
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}

/* c := a */
void _bncomp_subst_mpfmatrix(MPFMatrix c, MPFMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_subst_mpfmatrix\n");
		return;
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, get_mpfmatrix_ij(a, i, j));
	}
}

/* c := 0 */
void _bncomp_set0_mpfmatrix(MPFMatrix c)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij_ui(c, i, j, 0UL);
	}
}

/* c := I */
void _bncomp_setI_mpfmatrix(MPFMatrix c)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij_ui(c, i, j, 0UL);

		if(i < c->col_dim)
			set_mpfmatrix_ij_ui(c, i, i, 1UL);
	}
}

// Fully rewrite: 2022-03-29(Tue) T.Kouya
/* v := a * vb */
void _bncomp_mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrix_dvec\n");
		return;
	}

	prec = prec_mpfvector(v);
	row_dim = a->row_dim;
	col_dim = vb->dim;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);
	}
	
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < col_dim; j++)
		{
	#ifndef USE_MPFR
			mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j));
			mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
	#else
			//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
			mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j), tmp[thread_index], MPFR_RNDN);
	#endif
		}
		set_mpfvector_i(v, i, tmp[thread_index]);
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}

// Fully rewrite: 2022-03-29(Tue) T.Kouya
/* v := a^T * vb */
void _bncomp_mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrixt_mpfvec\n");
		return;
	}

	prec = prec_mpfvector(v);

	row_dim = vb->dim;
	col_dim = a->col_dim;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);
	}
	
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < col_dim; j++)
		{
	#ifndef USE_MPFR
			mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j));

			#pragma omp critical
			mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
	#else
			//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
			mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], MPFR_RNDN);
	#endif
		}
		set_mpfvector_i(v, i, tmp[thread_index]);
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}

// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b)
{
    int i, j;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    long int real_total_dim;
    int real_num_div_a, real_num_div_b;
    DMatrix *div_a, *div_b, *div_ret;
    MPFMatrix tmp_ret;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_mpfmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    //tmp_ret = init2_mpfmatrix(row_dim, col_dim, ret->prec);

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    div_ret = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));

    #pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
    {
        div_a[i] = init_dmatrix(row_dim, mid_dim);
        div_ret[i] = init_dmatrix(row_dim, col_dim);
    }

    #pragma omp parallel for
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    //div_ret = init_dmatrix(row_dim, col_dim);

    #pragma omp parallel sections
    {
        #pragma omp section
        real_num_div_a = split_mpfmatrix_dmat(div_a, max_num_div_a, a);

        #pragma omp section
        real_num_div_b = split_mpfmatrix_t_dmat(div_b, max_num_div_b, b);
    }

    set0_mpfmatrix(ret);
    #pragma omp parallel for private(i, j) 
    for(i = 0; i < real_num_div_a; i++)
    {
        //for(j = 0; j < real_num_div_b; j++)
        for(j = 0; j < real_num_div_b - i; j++)
        {
#ifdef USE_IMKL
            set0_dmatrix(div_ret[i]);
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
                div_ret[i]->element,
                div_ret[i]->real_col_dim   // n
              );
#else // USE_IMKL
            mul_dmatrix(div_ret[i], div_a[i], div_b[j]);
#endif // USE_IMKL

            #pragma omp critical
                add_mpfmatrix_dmat(ret, ret, div_ret[i]);
       }
    }

    //free_dmatrix(div_ret);
    #pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
    {
        free_dmatrix(div_a[i]);
        free_dmatrix(div_ret[i]);
    }

    #pragma omp parallel for
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);
    free(div_ret);

    //free_mpfmatrix(tmp_ret);

}

// Matrix-Vector multiplication based on Ozaki scheme
void _bncomp_mul_mpfmatrix_mpfvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_vb;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DMatrix *div_a;
    DVector *div_vb, *div_ret;

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));
    div_ret = (DVector *)calloc(max_num_div_a, sizeof(DMatrix));

    #pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
    {
        div_a[i] = init_dmatrix(row_dim, col_dim);
        div_ret[i] = init_dvector(vec_dim);
    }

    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);

    //div_ret = init_dvector(vec_dim);

    #pragma omp parallel sections
    {
        #pragma omp section
            real_num_div_a = split_mpfmatrix_dmat(div_a, max_num_div_a, a);
        
        #pragma omp section
            real_num_div_vb = split_mpfvector_dvec(div_vb, max_num_div_vb, vb);
    }

    set0_mpfvector(ret);
    #pragma omp parallel for private(j)
    for(i = 0; i < real_num_div_a; i++)
    {
        for(j = 0; j < real_num_div_vb; j++)
        {

#ifdef USE_IMKL
            set0_dvector(div_ret[i]);
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
                div_ret[i]->element,
                1
            );
#else // USE_IMKL
            //mul_dmatrix(div_ret[i], div_a[i], div_vb[j]);
            mul_dmatrix_dvec(div_ret[i], div_a[i], div_vb[j]); // 2025-07-09 fixed!
#endif // USE_IMKL
            #pragma omp critical
                add_mpfvector_dvec(ret, ret, div_ret[i]);
       }
    }

    //free_dvector(div_ret);

    #pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
    {
        free_dmatrix(div_a[i]);
        free_dvector(div_ret[i]);
    }

    #pragma omp parallel for
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);
    free(div_ret);

}

#if 0
// Matrix multiplication based on Ozaki scheme (4M)
void _bncomp_mul_cmpfmatrix_oz_4m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    MPFMatrix a_real, a_image, b_real, b_image, c_real[2], c_image[2];

    a_real  = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    a_image = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    b_real  = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    b_image = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    c_real[0]  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image[0] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_real[1]  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image[1] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);

    // A_r + A_i * i := A
    #pragma omp parallel sections
    {
        #pragma omp section
            separate_cmpfmatrix(a_real, a_image, a);

        #pragma omp section
            separate_cmpfmatrix(b_real, b_image, b);
    }

    // C := (A_r + A_i *i) * (B_r + B_i * i)
    //    = (A_r + B_r - A_i * B_i) + (A_r * B_i + A_i * B_r) * i
    _bncomp_mul_mpfmatrix_oz(c_real[0], a_real, max_num_div_a_real, b_real, max_num_div_b_real);
    _bncomp_mul_mpfmatrix_oz(c_real[1], a_image, max_num_div_a_image, b_image, max_num_div_b_image);
    _bncomp_sub_mpfmatrix(c_real[0], c_real[0], c_real[1]);

    _bncomp_mul_mpfmatrix_oz(c_image[0], a_real, max_num_div_a_real, b_image, max_num_div_b_image);
    _bncomp_mul_mpfmatrix_oz(c_image[1], a_image, max_num_div_a_image, b_real, max_num_div_b_real);
    _bncomp_add_mpfmatrix(c_image[0], c_image[0], c_image[1]);

    // C := C_r + C_i * i
    merge_cmpfmatrix(ret, c_real[0], c_image[0]);

    free_mpfmatrix(a_real);
    free_mpfmatrix(a_image);
    free_mpfmatrix(b_real);
    free_mpfmatrix(b_image);
    free_mpfmatrix(c_real[0]);
    free_mpfmatrix(c_image[0]);
    free_mpfmatrix(c_real[1]);
    free_mpfmatrix(c_image[1]);
}

// Matrix multiplication based on Ozaki scheme (3M)
void _bncomp_mul_cmpfmatrix_oz_3m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    int max_num_div_apa, max_num_div_bpb;
    MPFMatrix a_real, a_image, b_real, b_image, c_real, c_image, apa, bpb, t[2];

    // Allocate
    a_real  = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    a_image = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
    apa     = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);

    b_real  = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    b_image = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);
    bpb     = init2_mpfmatrix(b->row_dim, b->col_dim, b->prec);

    c_real  = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    c_image = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    t[0] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);
    t[1] = init2_mpfmatrix(ret->row_dim, ret->col_dim, ret->prec);

    // A_r + A_i * i := A
    // B_r + B_i * i := B
    #pragma omp parallel sections
    {
        #pragma omp section
            separate_cmpfmatrix(a_real, a_image, a);

        #pragma omp section
            separate_cmpfmatrix(b_real, b_image, b);
    }

    // T0 := A_r * B_r
    // T1 := A_i * B_i
    _bncomp_mul_mpfmatrix_oz(t[0], a_real, max_num_div_a_real, b_real, max_num_div_b_real);
    _bncomp_mul_mpfmatrix_oz(t[1], a_image, max_num_div_a_image, b_image, max_num_div_b_image);

    // C_r := T0 - T1
    _bncomp_sub_mpfmatrix(c_real, t[0], t[1]);

    // C_i := (A_r + A_i) * (B_r + B_i) - T0 - T1
    _bncomp_add_mpfmatrix(apa, a_real, a_image);
    _bncomp_add_mpfmatrix(bpb, b_real, b_image);
    max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
    max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
    _bncomp_mul_mpfmatrix_oz(c_image, apa, max_num_div_apa, bpb, max_num_div_bpb);
    _bncomp_sub_mpfmatrix(c_image, c_image, t[0]);
    _bncomp_sub_mpfmatrix(c_image, c_image, t[1]);

    // C := C_r + C_i * i
    merge_cmpfmatrix(ret, c_real, c_image);

    // Free
    free_mpfmatrix(a_real);
    free_mpfmatrix(a_image);
    free_mpfmatrix(apa);
    free_mpfmatrix(b_real);
    free_mpfmatrix(b_image);
    free_mpfmatrix(bpb);
    free_mpfmatrix(c_real);
    free_mpfmatrix(c_image);
    free_mpfmatrix(t[0]);
    free_mpfmatrix(t[1]);

}
#endif // 0
#endif // USE_GMP
