/********************************************************************************/
/* get_linear_system.c: Set a linear equation for testing                       */
/* Copyright (C) 2016 Tomonori Kouya                                            */
/*                                                                              */
/* Referred from "A Numerical Comparison of Several Codition Number Estimators  */
/* written by T.Matsuo, S.Masaaki, M.Mori (JSIAM vol.7, no.3 (1997) pp.307-319  */
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
#include <stdio.h>
#include <math.h>
#include "bnc.h"

#include "get_linear_system.h"

/* set a vector for numerical validation */
/* type =  0 ... test_v = [1 2 ... n]^T                            */
/* type =  1 ... test_v = [ ... (-1)^irand * irand ...]^T          */
/* type >= 2 ... test_v = [ ... (-1)^irand * irand/randmax ... ]^T */
void set_test_mpfvector(MPFVector test_v, int type)
{
	unsigned long prec;
	long int i;
	mpf_t val;

	prec = test_v->prec;

	mpf_init2(val, prec);

	switch(type)
	{
		/* test_v = [1 2 ... n]^T */
		case 0:
			for(i = 0; i < test_v->dim; i++)
			{
				mpf_set_ui(val, (unsigned long)(i + 1));
				set_mpfvector_i(test_v, i, val);
			}
			break;

		/* seed = type */
		/* test_v = [ ... (-1)^irand * irand ... ]^T */
		case 1:
			srand(type);
			for(i = 0; i < test_v->dim; i++)
			{
				mpf_set_ui(val, (unsigned long)rand());
				if((rand() % 2) != 0)
					mpf_neg(val, val);;
				set_mpfvector_i(test_v, i, val);
			}
			break;

		/* type >= 2 ... test_v = [ ... (-1)^irand * irand/randmax ... ]^T */
		case 2:
		default:
			srand(type);
			for(i = 0; i < test_v->dim; i++)
			{
				mpf_set_ui(val, (unsigned long)rand());
				mpf_div_ui(val, val, (unsigned long)RAND_MAX);
				if((rand() % 2) != 0)
					mpf_neg(val, val);
				set_mpfvector_i(test_v, i, val);
			}
			break;
	}

	mpf_clear(val);

}

/* generate random regular matrix */
void generate_regular_mpfmatrix(mpf_t cond, MPFMatrix mat, MPFMatrix inv_mat, int matrix_type, int seed)
{
	unsigned long prec;
	long i, j;
	mpf_t tmp[3];

	prec = mat->prec;

	mpf_init2(tmp[0], prec);
	mpf_init2(tmp[1], prec);
	mpf_init2(tmp[2], prec);

	/* set seed for rand() */
	srand(seed);

	set0_mpfmatrix(mat);
	set0_mpfmatrix(inv_mat);

	mpf_set_ui(tmp[0], RAND_MAX);

	/* generate dense matrix */
	for(i = 0; i < mat->row_dim; i++)
	{
		switch(matrix_type)
		{
			case _BNC_GENMAT_REAL_DENSE: /* General Dense matrix */
				for(j = 0; j < mat->col_dim; j++)
				{
					mpf_set_ui(tmp[1], rand());
					mpf_div(tmp[2], tmp[1], tmp[0]); // rand() / RAND_MAX
					set_mpfmatrix_ij(mat, i, j, tmp[2]);
				}
				break;
			case _BNC_GENMAT_REAL_TRIDIAG: /* Tridiagonal matrix */
				for(j = max(0, i - 1); j <= min(i + 1, mat->col_dim - 1); j++)
				{
					mpf_set_ui(tmp[1], rand());
					mpf_div(tmp[2], tmp[1], tmp[0]); // rand() / RAND_MAX
					set_mpfmatrix_ij(mat, i, j, tmp[2]);
				}
				break;
		}
		for(j = 0; j < mat->col_dim; j++)
			set_mpfmatrix_ij(inv_mat, i, j, get_mpfmatrix_ij(mat, i, j));
	}

	/* get inverse of matrix */
	inv_mpfmatrix(inv_mat);

	/* return condition number of mat */
	normi_mpfmatrix(tmp[0], mat);
	normi_mpfmatrix(tmp[1], inv_mat);
	mpf_mul(cond, tmp[0], tmp[1]);

	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	return;
}

/* Generate Real Dense Matrix with Eigenvalues as eig[] */
/* A := X^(-1) * Diag(eig[]) * X */
/* matrix_type = 0: Unsymmetrix Dense Matrix */
/* mat: A, trans_mat X */
void generate_mpfmatrix(MPFMatrix mat, MPFMatrix trans_mat, MPFVector eig, int matrix_type, int seed)
{
	unsigned long prec;
	long int row_dim, col_dim, i;
	mpf_t condi;
	MPFMatrix tmp_mat[3];

	prec = mat->prec;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	mpf_init2(condi, prec);

	tmp_mat[0] = init2_mpfmatrix(row_dim, col_dim, prec);
	tmp_mat[1] = init2_mpfmatrix(row_dim, col_dim, prec);
	tmp_mat[2] = init2_mpfmatrix(row_dim, col_dim, prec);

	/* mat := diag(eig(i)) */
	set0_mpfmatrix(mat);
	for(i = 0; i < row_dim; i++)
		set_mpfmatrix_ij(mat, i, i, get_mpfvector_i(eig, i));

	/* set X and X^(-1) */
	generate_regular_mpfmatrix(condi, tmp_mat[1], tmp_mat[0], matrix_type, seed);
	fprintf(stderr, "cond_inf(X): ");
	mpf_out_str(stderr, 10, 17, condi);
	fprintf(stderr, "\n");

	/* X^(-1) * diag(eig(i)) */
	mul_mpfmatrix(tmp_mat[2], tmp_mat[0], mat);

	/* (X^(-1) * diag(eig(i))) * X */
	mul_mpfmatrix(mat, tmp_mat[2], tmp_mat[1]);

	/* trans_mat := tmp_mat[1] */
	if(trans_mat != NULL)
		subst_mpfmatrix(trans_mat, tmp_mat[1]);

	free_mpfmatrix(tmp_mat[0]);
	free_mpfmatrix(tmp_mat[1]);
	free_mpfmatrix(tmp_mat[2]);
}

/* check equarity of two vectors */
/* return 0 if vec_a == mat_b */
/* return -1 if vec_a_i < vec_b_i for all i, j */
/* return +1 if vec_a_i > vec_b_i for all i, j */
/* return +2 otherwise */
int compare_mpfvector(MPFVector vec_a, MPFVector vec_b)
{
	int ret = +2, ret_mpf_cmp, num_big, num_small, num_equal;
	long int i, dim;

	dim = vec_a->dim;

	if(dim != vec_b->dim)
	{
		fprintf(stderr, "Warning: different sizes! (vec_a[%ld] != vec_b[%ld])\n", dim, vec_b->dim);
		return ret;
	}

	num_big   = 0;
	num_small = 0;
	num_equal = 0;
	for(i = 0; i < dim; i++)
	{
		ret_mpf_cmp = mpf_cmp(get_mpfvector_i(vec_a, i), get_mpfvector_i(vec_b, i));
		if(ret_mpf_cmp == 0)
			num_equal++;
		else if(ret_mpf_cmp < 0)
			num_small++;
		else if(ret_mpf_cmp > 0)
			num_big++;
	}

	if(num_equal >= dim)
		ret = 0;
	else if(num_big >= dim)
		ret = +1;
	else if(num_small >= dim)
		ret = -1;

	return ret;
}

/* check equarity of two matrices */
/* return 0 if mat_a == mat_b */
/* return -1 if mat_a_ij < mat_b_ij for all i, j */
/* return +1 if mat_a_ij > mat_b_ij for all i, j */
/* return +2 otherwise */
int compare_mpfmatrix(MPFMatrix mat_a, MPFMatrix mat_b)
{
	int ret = +2, ret_mpf_cmp, num_big, num_small, num_equal;
	long int i, j, row_dim, col_dim;

	row_dim = mat_a->row_dim;
	col_dim = mat_a->col_dim;

	if((row_dim != mat_b->row_dim) || (col_dim != mat_b->col_dim))
	{
		fprintf(stderr, "Warning: different sizes! (mat_a[%ld][%ld] != mat_b[%ld][%ld])\n", row_dim, col_dim, mat_b->row_dim, mat_b->col_dim);
		return ret;
	}


	num_big   = 0;
	num_small = 0;
	num_equal = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			ret_mpf_cmp = mpf_cmp(get_mpfmatrix_ij(mat_a, i, j), get_mpfmatrix_ij(mat_b, i, j));
			if(ret_mpf_cmp == 0)
				num_equal++;
			else if(ret_mpf_cmp < 0)
				num_small++;
			else if(ret_mpf_cmp > 0)
				num_big++;
		}
	}

	if(num_equal >= row_dim * col_dim)
		ret = 0;
	else if(num_big >= row_dim * col_dim)
		ret = +1;
	else if(num_small >= row_dim * col_dim)
		ret = -1;

	return ret;
}
/* get nearly correctly rounded coefficient matrix A */
int get_correct_rounded_mpfmatrix(MPFMatrix mat, void(* get_coefmat_mpfmatrix)(MPFMatrix mat))
{
	unsigned long prec_l, prec_s, prec, prec_diff, initial_prec_diff; // prec = max(mat->prec, vec->prec)
	long int row_dim, col_dim, flag_stop;
	MPFMatrix tmp_mat, mat_l, mat_s;

	/* set dims and precs */
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;
	prec = mat->prec;
	initial_prec_diff = (prec / 10 > 32) ? prec / 10 : 32; // 32 bits

	/* Initialize */
	tmp_mat = init2_mpfmatrix(row_dim, col_dim, prec);

	/* main loop */
	flag_stop = 0;
	for(prec_diff = initial_prec_diff; prec_diff < prec * 128; prec_diff *= 2)
	{
		prec_s = prec + prec_diff;
		prec_l = prec + prec_diff;

		/* initialize variables */
		mat_l = init2_mpfmatrix(row_dim, col_dim, prec_l);
		mat_s = init2_mpfmatrix(row_dim, col_dim, prec_s);

		/* get mat_l, mat_s */
		get_coefmat_mpfmatrix(mat_l);
		get_coefmat_mpfmatrix(mat_s);

		/* mat     := (prec)mat_l */
		/* tmp_mat := (prec)mat_s */
		/* continue whether mat != tmp_mat */
		subst_mpfmatrix(    mat, mat_l);
		subst_mpfmatrix(tmp_mat, mat_s);
		if(compare_mpfmatrix(mat, tmp_mat) == 0)
		{
				flag_stop = 1;
		}

		/* free variables */
		free_mpfmatrix(mat_l);
		free_mpfmatrix(mat_s);

		if(flag_stop == 1)
		{
			fprintf(stderr, "SUCCESS! (%ld, %ld bits)\n", prec_s, prec_l);
			break;
		}
		fprintf(stderr, "Not SUCCESS... (%ld, %ld bits)\n", prec_s, prec_l);

		/* end of loop */
	}
	free_mpfmatrix(tmp_mat);

	if(flag_stop == 0)
		fprintf(stderr, "Warning: matrix would not be correctly rounded...\n");

	return flag_stop;
}

/* get nearly correctly rounded coefficient matrix A and constant vector b */
/* A * x = b ... mat as A, vec as b, ans_vec as x */
int get_correct_rounded_mpfmatrix_mpfvec(MPFMatrix mat, MPFVector vec, MPFVector ans_vec, void(* get_coefmat_mpfmatrix)(MPFMatrix mat))
{
	unsigned long prec_l, prec_s, prec, prec_diff, initial_prec_diff; // prec = max(mat->prec, vec->prec)
	long int row_dim, col_dim, flag_stop;
	MPFMatrix tmp_mat, mat_l, mat_s;
	MPFVector tmp_vec, vec_l, vec_s;

	/* set dims and precs */
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;
	prec = (mat->prec > vec->prec) ? mat->prec : vec->prec;
	initial_prec_diff = (prec / 10 > 32) ? prec / 10 : 32; // 32 bits

	/* Initialize */
	tmp_mat = init2_mpfmatrix(row_dim, col_dim, prec);
	tmp_vec = init2_mpfvector(col_dim, prec);

	/* main loop */
	flag_stop = 0;
	for(prec_diff = initial_prec_diff; prec_diff < prec * 128; prec_diff *= 2)
	{
		prec_s = prec + prec_diff;
		prec_l = prec + prec_diff;

		/* initialize variables */
		mat_l = init2_mpfmatrix(row_dim, col_dim, prec_l);
		mat_s = init2_mpfmatrix(row_dim, col_dim, prec_s);
		vec_l = init2_mpfvector(col_dim, prec_l);
		vec_s = init2_mpfvector(col_dim, prec_s);

		/* get mat_l, mat_s */
		get_coefmat_mpfmatrix(mat_l);
		get_coefmat_mpfmatrix(mat_s);

		/* mat     := (prec)mat_l */
		/* tmp_mat := (prec)mat_s */
		/* continue whether mat != tmp_mat */
		subst_mpfmatrix(    mat, mat_l);
		subst_mpfmatrix(tmp_mat, mat_s);
		if(compare_mpfmatrix(mat, tmp_mat) == 0)
		{
			/* b := A * x */
			mul_mpfmatrix_mpfvec(vec_l, mat_l, ans_vec);
			mul_mpfmatrix_mpfvec(vec_s, mat_s, ans_vec);

			/*     vec := (prec)vec_l */
			/* tmp_vec := (prec)vec_s */
			/* exit whether vec == tmp_vec */
			subst_mpfvector(    vec, vec_l);
			subst_mpfvector(tmp_vec, vec_s);
			if(compare_mpfvector(vec, tmp_vec) == 0)
				flag_stop = 1;
		}

		/* free variables */
		free_mpfmatrix(mat_l);
		free_mpfmatrix(mat_s);
		free_mpfvector(vec_l);
		free_mpfvector(vec_s);

		if(flag_stop == 1)
		{
			fprintf(stderr, "SUCCESS! (%ld, %ld bits)\n", prec_s, prec_l);
			break;
		}
		fprintf(stderr, "Not SUCCESS... (%ld, %ld bits)\n", prec_s, prec_l);

		/* end of loop */
	}
	free_mpfmatrix(tmp_mat);
	free_mpfvector(tmp_vec);

	if(flag_stop == 0)
		fprintf(stderr, "Warning: matrix and vector would not be correctly rounded...\n");

	return flag_stop;
}

#ifdef DEBUG
/* get testmatrix */
void testmat(MPFMatrix mat)
{
	long int dim;

	dim = (mat->row_dim < mat->col_dim) ? mat->row_dim : mat->col_dim;

	hilbert_mpfmatrix(mat, dim);
}

#define DIM 100

int main(int argc, char *argv[])
{
	unsigned long dprec;
	long int i;
	MPFMatrix mpfmat;
	MPFVector mpfvec, mpfans;
	
	if(argc <= 1)
	{
		printf("Usage: %s [prec_in_decimal]\n", argv[0]);
		return 0;
	}

	dprec = atol(argv[1]);
	if(dprec <= 0)
	{
		printf("Warning: %d decimal digits is too small!\n", dprec);
		return 0;
	}

	set_bnc_default_prec_decimal(dprec);

	mpfmat = init_mpfmatrix(DIM, DIM);
	mpfvec = init_mpfvector(DIM);
	mpfans = init_mpfvector(DIM);

	/* set answer vector */
	for(i = 0; i < DIM; i++)
		set_mpfvector_i_ui(mpfans, i, (unsigned long)i);

	get_correct_rounded_mpfmatrix_mpfvec(mpfmat, mpfvec, mpfans, testmat);

	//print_mpfmatrix(mpfmat);
	//print_mpfvector(mpfvec);
	//print_mpfvector(mpfans);

	free_mpfmatrix(mpfmat);
	free_mpfvector(mpfans);
	free_mpfvector(mpfvec);

	return 0;
}
#endif // DEBUG
