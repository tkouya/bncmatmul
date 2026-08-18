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

//#include "bnc.h"
#include "bncmatmul.h"

/* subcmul_dvector and subst_dmatrix_dvec are provided by libbncmatmul
   (see src/dlinear.c).  Local definitions kept here for reference only. */
#if 0
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
#endif

#if 0
// Lanczos method to transform real symmetric matrix to symmetric tridiagonal matrix
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
        // w := A * q
		mul_dmatrix_dvec(wvec, mat, qvec[i]);

        // alpha := (q, w) = (q, A * q)
		alpha = ip_dvector(qvec[i], wvec);
		set_dmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

        // r := w - alpha * q
		subcmul_dvector(rvec, wvec, alpha, qvec[i]);

        // r := w - alpha * q - beta * q
		if(i > 0)
			subcmul_dvector(rvec, rvec, beta, qvec[i - 1]);

        // beta := ||r||_2
		beta = norm2_dvector(rvec);

        // q := 1/beta * r
		cmul_dvector(qvec[i + 1], 1.0/beta, rvec);

		set_dmatrix_ij(trimat, i, i + 1, beta);
		set_dmatrix_ij(trimat, i + 1, i, beta);
	}

	free_dvector(rvec);
	free_dvector(wvec);
}

// Lanczos method to transform real matrix to tridiagonal matrix
int dlanczos(DMatrix trimat, DVector qvec[], DVector pvec[], DMatrix mat)
{
	long int times, i, j, dim;
	DVector rvec, svec, wvec, uvec;
	double alpha, beta, gamma;

	dim = mat->row_dim;
	set0_dmatrix(trimat);
	rvec = init_dvector(dim);
    svec = init_dvector(dim);
	wvec = init_dvector(dim);
	uvec = init_dvector(dim);

	#ifndef USE_RAND_START_LANCZOS
	/* q_1 := sqrt(n)^(-1) * [1 1 ... 1]^T */
    /* p_1 := q_1 */
	beta = sqrt((double)dim);
    gamma = beta;
	for(i = 0; i < dim; i++)
	{
		set_dvector_i(qvec[0], i, 1.0 / beta);
	}
    subst_dvector(pvec[0], qvec[0]);
	#else // USE_RAND_START_LANCZOS
	// q, p := rand
	srand(dim);
	for(i = 0; i < dim; i++)
	{
		set_dvector_i(qvec[0], i, (double)rand() / (double)RAND_MAX);
		set_dvector_i(pvec[0], i, (double)rand() / (double)RAND_MAX);
	}
	beta = norm2_dvector(qvec[0]);
	cmul2_dvector(qvec[0], 1.0 / beta);
	cmul2_dvector(pvec[0], 1.0 / norm2_dvector(pvec[0]));
	gamma = ip_dvector(qvec[0], pvec[0]); // / beta;
	//subst_dvector(pvec[0], qvec[0]);
	#endif // USE_RAND_START_LANCZOS

	/* main loop */
	for(i = 0; i < dim; i++)
	{
        // w := A * q
		mul_dmatrix_dvec(wvec, mat, qvec[i]);
        // u := A^T * p
        mul_dmatrixt_dvec(uvec, mat, pvec[i]);

        // alpha := (p, w) = (p, A * q)
		alpha = ip_dvector(pvec[i], wvec);
		set_dmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

        // r := w - alpha * q = A * q - alpha * q
        // s := u - alpha * p = A^T * p - alpha * p
		subcmul_dvector(rvec, wvec, alpha, qvec[i]);
        subcmul_dvector(svec, uvec, alpha, pvec[i]);

        // r := A * q - alpha * q - gamma * q
        // s := A^T * p - alpha * p - beta * p
		if(i > 0)
        {
			subcmul_dvector(rvec, rvec, gamma, qvec[i - 1]);
            subcmul_dvector(svec, svec, beta, pvec[i - 1]);
        }

        // beta := ||r||_2
		beta = norm2_dvector(rvec);

        // gamma := (r, s) / beta
        gamma = ip_dvector(rvec, svec) / beta;

        // q := 1/beta * r
		cmul_dvector(qvec[i + 1], 1.0 / beta, rvec);
        // p := 1/gamma * s
        cmul_dvector(pvec[i + 1], 1.0 / gamma, svec);    

		set_dmatrix_ij(trimat, i, i + 1, gamma);
		set_dmatrix_ij(trimat, i + 1, i, beta);
	}

	free_dvector(rvec);
    free_dvector(svec);
	free_dvector(wvec);
    free_dvector(uvec);
}
#endif // 0

#ifdef USE_GMP
/* subcmul_mpfvector and subst_mpfmatrix_mpfvec are provided by libbncmatmul
   (see src/mpflinear.c).  Local definitions kept here for reference only. */
#if 0
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
#endif

#if 0
// Lanczos method to transform real symmetric matrix to symmetric tridiagonal matrix
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

// Lanczos method to transform real matrix to tridiagonal matrix
int mpf_lanczos(MPFMatrix trimat, MPFVector qvec[], MPFVector pvec[], MPFMatrix mat)
{
	long int times, i, j, dim;
	MPFVector rvec, svec, wvec, uvec;
	mpf_t alpha, beta, gamma, tmp;

	mpf_init2(alpha, trimat->prec);
	mpf_init2(beta, trimat->prec);
	mpf_init2(gamma, trimat->prec);
	mpf_init2(tmp, trimat->prec);

	dim = mat->row_dim;
	set0_mpfmatrix(trimat);
	rvec = init2_mpfvector(dim, trimat->prec);
	svec = init2_mpfvector(dim, trimat->prec);
	wvec = init2_mpfvector(dim, trimat->prec);
	uvec = init2_mpfvector(dim, trimat->prec);

	/* q_1 = [1 ... 1]^T */
    /* p_1 := p_1 */
	mpf_set_ui(beta, (unsigned long)dim);
	mpf_sqrt(beta, beta);
	mpf_ui_div(tmp, 1UL, beta);
    mpf_set(gamma, beta);
	for(i = 0; i < dim; i++)
	{
		set_mpfvector_i(pvec[0], i, tmp);
		set_mpfvector_i(qvec[0], i, tmp);
	}

	/* main loop */
	for(i = 0; i < dim; i++)
	{
		mul_mpfmatrix_mpfvec(wvec, mat, qvec[i]);
		mul_mpfmatrixt_mpfvec(uvec, mat, pvec[i]);
		ip_mpfvector(alpha, pvec[i], wvec);
		set_mpfmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

		subcmul_mpfvector(rvec, wvec, alpha, qvec[i]);
		subcmul_mpfvector(svec, uvec, alpha, pvec[i]);

		if(i > 0)
        {
			subcmul_mpfvector(rvec, rvec, gamma, qvec[i - 1]);
			subcmul_mpfvector(svec, svec, beta, pvec[i - 1]);
        }
		norm2_mpfvector(beta, rvec);
		mpf_ui_div(tmp, 1UL, beta);
		cmul_mpfvector(qvec[i + 1], tmp, rvec);

		ip_mpfvector(gamma, rvec, svec);
        mpf_div(gamma, gamma, beta);
		mpf_ui_div(tmp, 1UL, gamma);
		cmul_mpfvector(pvec[i + 1], tmp, svec);

		set_mpfmatrix_ij(trimat, i, i + 1, gamma);
		set_mpfmatrix_ij(trimat, i + 1, i, beta);
	}

	free_mpfvector(rvec);
    free_mpfvector(svec);
	free_mpfvector(wvec);
    free_mpfvector(uvec);
	mpf_clear(alpha);
	mpf_clear(beta);
    mpf_clear(gamma);
	mpf_clear(tmp);
}
#endif // USE_GMP
#endif // 0

//#define DIM 3
//#define DIM 5
#define DIM 10
//#define DIM 50

int main(int argc, char *argv[])
{
	DMatrix da, dta, dq, dqt, dtmpmat;
	//DVector dpv[dim], dqv[dim];
	DVector *dpv, *dqv;
#ifdef USE_GMP
	unsigned long prec;
	MPFMatrix mpfa, mpfta, mpfq, mpfqt, mpftmpmat;
	//MPFVector mpfpv[dim], mpfqv[dim];
	MPFVector *mpfpv, *mpfqv;
#endif // USE_GMP
	long int i, j, dim;

#ifdef USE_GMP
	if(argc < 3)
	{
		printf("Usage: %s [dimension] [prec_in_bits]\n", argv[0]);
		return 0;
	}
	dim = (long int)atoi(argv[1]);
	prec = (unsigned long)atoi(argv[2]);
#else // USE_GMP
	if(argc <= 0)
	{
		print("Usage: %s [dimension]\n", argv[0]);
		return 0;
	}
	dim = atoi(argv[1]);

#endif // USE_GMP


/* Double */
dstart:
	/* initialize */
	da = init_dmatrix(dim, dim);
	dta = init_dmatrix(dim, dim);
	dq = init_dmatrix(dim, dim);
	dqt = init_dmatrix(dim, dim);
	dtmpmat = init_dmatrix(dim, dim);

	dpv = (DVector *)calloc(dim, sizeof(DVector));
	dqv = (DVector *)calloc(dim, sizeof(DVector));
	for(i = 0; i < dim; i++)
	{
		dqv[i] = init_dvector(dim);
       		dpv[i] = init_dvector(dim);
	}

	/* get problem */
	//hilbert_dmatrix(da, dim);
	lotkin_dmatrix(da, dim);
	print_dmatrix(da);

	/* lanczos */
	//dslanczos(dta, dqv, da);
	dlanczos(dta, dqv, dpv, da);

	/* print */
	//printf("Tridiagonal:\n");print_dmatrix(dta);
	printf("Tridiagonal:\n");printf("%25.17e\n", normf_dmatrix(dta));

	//subst_dmatrix_dvec(dq, dqv);
	// TMPMAT := Q * P^T 
	subst_dmatrix_dvec(dq, dpv);
	transpose_dmatrix(dqt, dq);
	subst_dmatrix_dvec(dq, dqv);
	mul_dmatrix(dtmpmat, dq, dqt);
	//printf("Q:\n"); print_dmatrix(dq);
	printf("P^T, Q:"); printf("%25.17e, %25.17e\n", normf_dmatrix(dqt), normf_dmatrix(dq));
	//printf("Q*Q^T:\n"); print_dmatrix(da);
	printf("Q * P^T:"); printf("%25.17e\n", normf_dmatrix(dtmpmat));

	mul_dmatrix(dtmpmat, dq, dta);
	mul_dmatrix(dq, dtmpmat, dqt);
	sub_dmatrix(dtmpmat, da, dq);
	printf("Original?:\n");print_dmatrix(dq);
	printf("||A - Q * T * P^T||_F / ||A||_F = %25.17e\n", normf_dmatrix(dtmpmat) / normf_dmatrix(da));

	/* end */
	free_dmatrix(da);
	free_dmatrix(dta);
	free_dmatrix(dq);
	free_dmatrix(dqt);
	free_dmatrix(dtmpmat);
	for(i = 0; i < dim; i++)
	{
		free_dvector(dqv[i]);
	        free_dvector(dpv[i]);
	}
	free(dpv);
	free(dqv);

//	goto end;

#ifdef USE_GMP
/* MPF */
mpfstart:
//	set_bnc_default_prec_decimal(50);
//	set_bnc_default_prec_decimal(100);
//	set_bnc_default_prec_decimal(5000);
	set_bnc_default_prec(prec);
	/* initialize */
	mpfa = init_mpfmatrix(dim, dim);
	mpfta = init_mpfmatrix(dim, dim);
	mpfq = init_mpfmatrix(dim, dim);
	mpfqt = init_mpfmatrix(dim, dim);

	mpfpv = (MPFVector *)calloc(dim, sizeof(MPFVector));
	mpfqv = (MPFVector *)calloc(dim, sizeof(MPFVector));
	for(i = 0; i < dim; i++)
    {
		mpfqv[i] = init_mpfvector(dim);
        mpfpv[i] = init_mpfvector(dim);
    }

	/* get problem */
	//hilbert_mpfmatrix(mpfa, dim);
    lotkin_mpfmatrix(mpfa, dim);
    print_mpfmatrix(mpfa);

	/* lanczos */
	//mpf_slanczos(mpfta, mpfqv, mpfa);
    mpf_lanczos(mpfta, mpfqv, mpfpv, mpfa);

	/* print */
	printf("Tridiagonal:\n");print_mpfmatrix(mpfta);

	//subst_mpfmatrix_mpfvec(mpfq, mpfqv);
	subst_mpfmatrix_mpfvec(mpfq, mpfpv);
	transpose_mpfmatrix(mpfqt, mpfq);
    subst_mpfmatrix_mpfvec(mpfq, mpfqv);
	mul_mpfmatrix(mpfa, mpfq, mpfqt);
	//mul_mpfmatrix_strassen(mpfa, mpfq, mpfqt, 4);
	printf("Q:\n"); print_mpfmatrix(mpfq);
	printf("Q*Q^T:\n"); print_mpfmatrix(mpfa);

	mul_mpfmatrix(mpfa, mpfq, mpfta);
	mul_mpfmatrix(mpfq, mpfa, mpfqt);
	//mul_mpfmatrix_strassen(mpfa, mpfq, mpfta, 4);
	//mul_mpfmatrix_strassen(mpfq, mpfa, mpfqt, 4);

	printf("Original?:\n");print_mpfmatrix(mpfq);

	/* end */
	free_mpfmatrix(mpfa);
	free_mpfmatrix(mpfta);
	free_mpfmatrix(mpfq);
	free_mpfmatrix(mpfqt);
	for(i = 0; i < dim; i++)
    {
		free_mpfvector(mpfqv[i]);
        free_mpfvector(mpfpv[i]);
    }
	free(mpfpv);
	free(mpfqv);
#endif // USE_GMP

	/* print itimes */
end:
	return 0;
}

