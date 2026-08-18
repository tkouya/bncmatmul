/********************************************************************************/
/* test_lanczos.c:                                                              */
/* Copyright (C) 2004-2011 Tomonori Kouya                                       */
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
#include <stdlib.h>
#include <math.h>

#include "bnc.h"

/* c = a - sc * b */
void subcmul_dvector(DVector c, DVector a, double sc, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: subcmul_dvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) - sc * get_dvector_i(b, i));

}

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_dmatrix_dvec(DMatrix mat, DVector vec[])
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			set_dmatrix_ij(mat, i, j, get_dvector_i(vec[j], i));
	}
}

int dslanczos(DMatrix trimat, DVector qvec[], DMatrix mat)
{
	long int times, i, j, dim;
	DVector rvec, wvec;
	double beta, alpha;

	dim = mat->row_dim;
	set0_dmatrix(trimat);
	rvec = init_dvector(dim);
	wvec = init_dvector(dim);

	/* q_1 = [1 ... 1]^T */
	beta = sqrt((double)dim);
	for(i = 0; i < dim; i++)
	{
		set_dvector_i(qvec[0], i, 1.0/beta);
	}

	/* main loop */
	for(i = 0; i < dim; i++)
	{
		mul_dmatrix_dvec(wvec, mat, qvec[i]);
		alpha = ip_dvector(qvec[i], wvec);
		set_dmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;
		subcmul_dvector(rvec, wvec, alpha, qvec[i]);
		if(i > 0)
			subcmul_dvector(rvec, rvec, beta, qvec[i - 1]);
		beta = norm2_dvector(rvec);
		cmul_dvector(qvec[i + 1], 1.0/beta, rvec);
		set_dmatrix_ij(trimat, i, i + 1, beta);
		set_dmatrix_ij(trimat, i + 1, i, beta);
	}

	free_dvector(rvec);
	free_dvector(wvec);
}

#ifdef USE_GMP
/* c = a - sc * b */
void subcmul_mpfvector(MPFVector c, MPFVector a, mpf_t sc, MPFVector b)
{
	long int i;
	mpf_t tmp, tmp1;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: subcmul_mpfvector\n");
		return;
	}

	//cmul_mpfvector(c, sc, b);
	//sub_mpfvector(c, a, c);
	mpf_init2(tmp, c->prec);
	mpf_init2(tmp1, c->prec);
	for(i = 0; i < c->dim; i++)
	{
		mpf_set(tmp1, get_mpfvector_i(a, i));
		mpf_mul(tmp, sc, get_mpfvector_i(b, i));
		mpf_sub(tmp, tmp1, tmp);
		set_mpfvector_i(c, i, tmp);
	}
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_mpfmatrix_mpfvec(MPFMatrix mat, MPFVector vec[])
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			set_mpfmatrix_ij(mat, i, j, get_mpfvector_i(vec[j], i));
	}
}

int mpf_slanczos(MPFMatrix trimat, MPFVector qvec[], MPFMatrix mat)
{
	long int times, i, j, dim;
	MPFVector rvec, wvec;
	mpf_t beta, alpha, tmp;

	mpf_init2(beta, trimat->prec);
	mpf_init2(alpha, trimat->prec);
	mpf_init2(tmp, trimat->prec);

	dim = mat->row_dim;
	set0_mpfmatrix(trimat);
	rvec = init2_mpfvector(dim, trimat->prec);
	wvec = init2_mpfvector(dim, trimat->prec);

	/* q_1 = [1 ... 1]^T */
	mpf_set_ui(beta, (unsigned long)dim);
	mpf_sqrt(beta, beta);
	mpf_ui_div(tmp, 1UL, beta);
	for(i = 0; i < dim; i++)
	{
		set_mpfvector_i(qvec[0], i, tmp);
	}

	/* main loop */
	for(i = 0; i < dim; i++)
	{
		mul_mpfmatrix_mpfvec(wvec, mat, qvec[i]);
		ip_mpfvector(alpha, qvec[i], wvec);
		set_mpfmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;
		subcmul_mpfvector(rvec, wvec, alpha, qvec[i]);
		if(i > 0)
			subcmul_mpfvector(rvec, rvec, beta, qvec[i - 1]);
		norm2_mpfvector(beta, rvec);
		mpf_ui_div(tmp, 1UL, beta);
		cmul_mpfvector(qvec[i + 1], tmp, rvec);
		set_mpfmatrix_ij(trimat, i, i + 1, beta);
		set_mpfmatrix_ij(trimat, i + 1, i, beta);
	}

	free_mpfvector(rvec);
	free_mpfvector(wvec);
	mpf_clear(alpha);
	mpf_clear(beta);
	mpf_clear(tmp);
}
#endif

#define DIM 5

main()
{
	DMatrix da, dta, dq, dqt;
	DVector dqv[DIM];
#ifdef USE_GMP
	MPFMatrix mpfa, mpfta, mpfq, mpfqt;
	MPFVector mpfqv[DIM];
#endif
	long int i, j;

/* Double */
dstart:
	/* initialize */
	da = init_dmatrix(DIM, DIM);
	dta = init_dmatrix(DIM, DIM);
	dq = init_dmatrix(DIM, DIM);
	dqt = init_dmatrix(DIM, DIM);
	for(i = 0; i < DIM; i++)
		dqv[i] = init_dvector(DIM);

	/* get problem */
	hilbert_dmatrix(da, DIM);
	print_dmatrix(da);

	/* lanczos */
	dslanczos(dta, dqv, da);

	/* print */
	printf("Tridiagonal:\n");print_dmatrix(dta);

	subst_dmatrix_dvec(dq, dqv);
	transpose_dmatrix(dqt, dq);
	mul_dmatrix(da, dq, dqt);
	printf("Q:\n"); print_dmatrix(dq);
	printf("Q*Q^T:\n"); print_dmatrix(da);

	mul_dmatrix(da, dq, dta);
	mul_dmatrix(dq, da, dqt);
	printf("Original?:\n");print_dmatrix(dq);

	/* end */
	free_dmatrix(da);
	free_dmatrix(dta);
	free_dmatrix(dq);
	free_dmatrix(dqt);
	for(i = 0; i < DIM; i++)
		free(dqv[i]);

//	goto end;

#ifdef USE_GMP
/* MPF */
mpfstart:
	set_bnc_default_prec_decimal(50);
	/* initialize */
	mpfa = init_mpfmatrix(DIM, DIM);
	mpfta = init_mpfmatrix(DIM, DIM);
	mpfq = init_mpfmatrix(DIM, DIM);
	mpfqt = init_mpfmatrix(DIM, DIM);
	for(i = 0; i < DIM; i++)
		mpfqv[i] = init_mpfvector(DIM);

	/* get problem */
	hilbert_mpfmatrix(mpfa, DIM);
	print_mpfmatrix(mpfa);

	/* lanczos */
	mpf_slanczos(mpfta, mpfqv, mpfa);

	/* print */
	printf("Tridiagonal:\n");print_mpfmatrix(mpfta);

	subst_mpfmatrix_mpfvec(mpfq, mpfqv);
	transpose_mpfmatrix(mpfqt, mpfq);
	mul_mpfmatrix(mpfa, mpfq, mpfqt);
	printf("Q:\n"); print_mpfmatrix(mpfq);
	printf("Q*Q^T:\n"); print_mpfmatrix(mpfa);

	mul_mpfmatrix(mpfa, mpfq, mpfta);
	mul_mpfmatrix(mpfq, mpfa, mpfqt);
	printf("Original?:\n");print_mpfmatrix(mpfq);

	/* end */
	free_mpfmatrix(mpfa);
	free_mpfmatrix(mpfta);
	free_mpfmatrix(mpfq);
	free_mpfmatrix(mpfqt);
	for(i = 0; i < DIM; i++)
		free_mpfvector(mpfqv[i]);
#endif

	/* print itimes */
end:
	return 0;
}

