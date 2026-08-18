/********************************************************************************/
/* qr.c: Gram-Schmidt and Modified Gram-Schmidt                                 */
/* Copyright (c) 2003-2011 Tomonori Kouya                                       */
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
#include <stdlib.h>
// #include <alloc.h>
#include <string.h>

#include "bnc.h"


/* Double */

/************************************************/
/* Gram-Schmidt                                 */
/************************************************/
void dgram_schmidt(DMatrix q, DMatrix r, DMatrix a)
{
	double dtmp, qnorm;
	long int dim, i, j, k;

	dim = a->row_dim;

	/* Main Loop */
	for(i = 0; i < dim; i++)
	{
		/* r_ji := (q_j, a_i) */
		for(j = 0; j < i; j++)
		{
			dtmp = 0.0;
			for(k = 0; k < dim; k++)
				dtmp += gdmij(q, k, j) * gdmij(a, k, i);
			sdmij(r, j, i, dtmp);
		}

		/* q'_i := a_i - \sum^(i-1)_{j=1} r_ji * q_j */
		qnorm = 0.0;
		for(k = 0; k < dim; k++)
		{
			dtmp = gdmij(a, k, i);
			for(j = 0; j < i; j++)
				dtmp -= gdmij(r, j, i) * gdmij(q, k, j);
			sdmij(q, k, i, dtmp);
			qnorm += dtmp * dtmp;
		}
		qnorm = sqrt(qnorm);
		sdmij(r, i, i, qnorm);
		for(k = 0; k < dim; k++)
		{
			dtmp = gdmij(q, k, i);
			sdmij(q, k, i, dtmp / qnorm);
		}
	}
}

/************************************************/
/* Modified Gram-Schmidt                        */
/************************************************/
void dmgram_schmidt(DMatrix q, DMatrix r, DMatrix a)
{
	double dtmp, anorm;
	long int dim, i, j, k;

	dim = a->row_dim;

	/* Main Loop */
	for(i = 0; i < dim; i++)
	{
		/* r_ii := ||a^(i)_i||_2 */
		anorm = 0.0;
		for(k = 0; k < dim; k++)
			anorm += gdmij(a, k, i) * gdmij(a, k, i);
		anorm = sqrt(anorm);
		sdmij(r, i, i, anorm);

		/* q_i := a^(i)_i / ||a^(i)_i||_2 */
		for(k = 0; k < dim; k++)
			sdmij(q, k, i, gdmij(a, k, i) / anorm);

		for(j = i + 1; j < dim; j++)
		{
			dtmp = 0.0;
			for(k = 0; k < dim; k++)
				dtmp += gdmij(q, k, i) * gdmij(a, k, j);
			sdmij(r, i, j, dtmp);

			for(k = 0; k < dim; k++)
			{
				dtmp = gdmij(a, k, j);
				dtmp -= gdmij(r, i, j) * gdmij(q, k, i);
				sdmij(a, k, j, dtmp);
			}
		}
	}
}

/************************************************/
/* QR Method with Modified Gram-Schmidt         */
/************************************************/
void dqr(DMatrix a, long int maxtimes)
{
	long int dim, times;
	DMatrix q, r;

	dim = a->row_dim;
	q = init_dmatrix(dim, dim);
	r = init_dmatrix(dim, dim);

	for(times = 0; times < maxtimes; times++)
	{
		dmgram_schmidt(q, r, a);
		mul_dmatrix(a, r, q);
	}

	free_dmatrix(q);
	free_dmatrix(r);
}

/* MPF */
#ifdef USE_GMP
/************************************************/
/* Gram-Schmidt                                 */
/************************************************/
void mpf_gram_schmidt(MPFMatrix q, MPFMatrix r, MPFMatrix a)
{
	mpf_t dtmp, tmp, qnorm;
	long int dim, i, j, k;
	unsigned long prec;

	dim = a->row_dim;
	prec = a->prec;

	mpf_init2(dtmp, prec);
	mpf_init2(tmp, prec);
	mpf_init2(qnorm, prec);

	/* Main Loop */
	for(i = 0; i < dim; i++)
	{
		/* r_ji := (q_j, a_i) */
		for(j = 0; j < i; j++)
		{
			mpf_set_ui(dtmp, 0UL);
			for(k = 0; k < dim; k++)
			{
				mpf_mul(tmp, gmpfmij(q, k, j), gmpfmij(a, k, i));
				mpf_add(dtmp, dtmp, tmp);
			}
			smpfmij(r, j, i, dtmp);
		}

		/* q'_i := a_i - \sum^(i-1)_{j=1} r_ji * q_j */
		mpf_set_ui(qnorm, 0UL);
		for(k = 0; k < dim; k++)
		{
			mpf_set(dtmp, gmpfmij(a, k, i));
			for(j = 0; j < i; j++)
			{
				mpf_mul(tmp, gmpfmij(r, j, i), gmpfmij(q, k, j));
				mpf_sub(dtmp, dtmp, tmp);
			}
			smpfmij(q, k, i, dtmp);
			mpf_mul(tmp, dtmp, dtmp);
			mpf_add(qnorm, qnorm, tmp);
		}
		mpf_sqrt(qnorm, qnorm);
		smpfmij(r, i, i, qnorm);
		for(k = 0; k < dim; k++)
		{
			mpf_set(dtmp, gmpfmij(q, k, i));
			mpf_div(tmp, dtmp, qnorm);
			smpfmij(q, k, i, tmp);
		}
	}

	mpf_clear(dtmp);
	mpf_clear(tmp);
	mpf_clear(qnorm);
}

/************************************************/
/* Modified Gram-Schmidt                        */
/************************************************/
void mpf_mgram_schmidt(MPFMatrix q, MPFMatrix r, MPFMatrix a)
{
	mpf_t dtmp, tmp, anorm;
	long int dim, i, j, k;
	unsigned long prec;

	dim = a->row_dim;
	prec = a->prec;

	mpf_init2(dtmp, prec);
	mpf_init2(tmp, prec);
	mpf_init2(anorm, prec);

	/* Main Loop */
	for(i = 0; i < dim; i++)
	{
		/* r_ii := ||a^(i)_i||_2 */
		mpf_set_ui(anorm, 0UL);
		for(k = 0; k < dim; k++)
		{
			mpf_mul(tmp, gmpfmij(a, k, i), gmpfmij(a, k, i));
			mpf_add(anorm, anorm, tmp);
		}
		mpf_sqrt(anorm, anorm);
		smpfmij(r, i, i, anorm);

		/* q_i := a^(i)_i / ||a^(i)_i||_2 */
		for(k = 0; k < dim; k++)
		{
			mpf_div(tmp, gmpfmij(a, k, i), anorm);
			smpfmij(q, k, i, tmp);
		}

		for(j = i + 1; j < dim; j++)
		{
			mpf_set_ui(dtmp, 0UL);
			for(k = 0; k < dim; k++)
			{
				mpf_mul(tmp, gmpfmij(q, k, i), gmpfmij(a, k, j));
				mpf_add(dtmp, dtmp, tmp);
			}
			smpfmij(r, i, j, dtmp);

			for(k = 0; k < dim; k++)
			{
				mpf_set(dtmp, gmpfmij(a, k, j));
				mpf_mul(tmp, gmpfmij(r, i, j), gmpfmij(q, k, i));
				mpf_sub(dtmp, dtmp, tmp);
				smpfmij(a, k, j, dtmp);
			}
		}
	}

	mpf_clear(dtmp);
	mpf_clear(tmp);
	mpf_clear(anorm);
}

/************************************************/
/* QR Method with Modified Gram-Schmidt         */
/************************************************/
void mpf_qr(MPFMatrix a, long int maxtimes)
{
	long int dim, times;
	MPFMatrix q, r;

	dim = a->row_dim;
	q = init_mpfmatrix(dim, dim);
	r = init_mpfmatrix(dim, dim);

	for(times = 0; times < maxtimes; times++)
	{
		mpf_mgram_schmidt(q, r, a);
		mul_mpfmatrix(a, r, q);
	}

	free_mpfmatrix(q);
	free_mpfmatrix(r);
}
#endif
