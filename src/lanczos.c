/********************************************************************************/
/*                                                                              */
/* lanczos.c : Lanczos methods for sparse and dense matrices                    */
/* Copyright (c) 2024 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* Version 0.1 2024-05-09 : Create lanczos.c from old test programs             */
/* Version 0.2 2024-10-21 : Append double-mp version                            */
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
#include "bncmatmul.h"

// ---------------- //
// Double precision //
// ---------------- //

// Lanczos method to transform real symmetric matrix to symmetric tridiagonal matrix
#ifdef USE_SPARSE_VERSION
//long int dslancsoz_sp(DMatrix trimat, DVector qvec[], DRSMatrix mat_sp)
long int dslancsoz_sp(DMatrix trimat, DVector *qvec, DRSMatrix mat_sp)
#else // USE_SPARSE_VERSION
//long int dslanczos(DMatrix trimat, DVector qvec[], DMatrix mat)
long int dslanczos(DMatrix trimat, DVector *qvec, DMatrix mat)
#endif // USE_SPARSE_VERSION
{
	long int times, i, j, dim;
	DVector rvec, wvec, qi, qim1;
	double beta, alpha;

#ifdef USE_SPARSE_VERSION
	dim = mat_sp->row_dim;
#else // USE_SPARSE_VERSION
	dim = mat->row_dim;
#endif // USE_SPARSE_VERSION

	set0_dmatrix(trimat);
	rvec = init_dvector(dim);
	wvec = init_dvector(dim);
    qi = init_dvector(dim); // q[i]
    qim1 = init_dvector(dim); // q[i - 1]

	/* q_1 = [1 ... 1]^T */
	beta = sqrt((double)dim);
	for(i = 0; i < dim; i++)
	{
		//set_dvector_i(qvec[0], i, 1.0/beta);
		set_dvector_i(qi, i, 1.0 / beta);
	}
    if(qvec != NULL) subst_dvector(qvec[0], qi);

	/* main loop */
	for(i = 0; i < dim; i++)
	{
        // w := A * q
#ifdef USE_SPARSE_VERSION
		//mul_drsmatrix_dvec(wvec, mat_sp, qvec[i]);
		mul_drsmatrix_dvec(wvec, mat_sp, qi);
#else // USE_SPARSE_VERSION
		//mul_dmatrix_dvec(wvec, mat, qvec[i]);
		mul_dmatrix_dvec(wvec, mat, qi);
#endif // USE_SPARSE_VERSION

        // alpha := (q, w) = (q, A * q)
		//alpha = ip_dvector(qvec[i], wvec);
		alpha = ip_dvector(qi, wvec);
		set_dmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

        // r := w - alpha * q
		//subcmul_dvector(rvec, wvec, alpha, qvec[i]);
		subcmul_dvector(rvec, wvec, alpha, qi);

        // r := w - alpha * q - beta * q
		if(i > 0)
        {
			//subcmul_dvector(rvec, rvec, beta, qvec[i - 1]);
			subcmul_dvector(rvec, rvec, beta, qim1);
        }
        // beta := ||r||_2
		beta = norm2_dvector(rvec);

        // q := 1/beta * r
		//cmul_dvector(qvec[i + 1], 1.0/beta, rvec);
        subst_dvector(qim1, qi); // q[i - 1] := q[i]
        cmul_dvector(qi, 1.0/beta, rvec); // q[i + 1] := r // ||r||_2
        if(qvec != NULL) subst_dvector(qvec[i + 1], qi);

		set_dmatrix_ij(trimat, i, i + 1, beta);
		set_dmatrix_ij(trimat, i + 1, i, beta);
	}

	free_dvector(rvec);
	free_dvector(wvec);
    free_dvector(qi);
    free_dvector(qim1);

    return i;
}

// Lanczos method to transform real matrix to tridiagonal matrix
#ifdef USE_SPARSE_VERSION
//long int dlanczos_sp(DMatrix trimat, DVector qvec[], DVector pvec[], DRSMatrix mat_sp)
long int dlanczos_sp(DMatrix trimat, DVector *qvec, DVector *pvec, DRSMatrix mat_sp)
#else // USE_SPARSE_VERSION
long int dlanczos(DMatrix trimat, DVector *qvec, DVector *pvec, DMatrix mat)
#endif // USE_SPARSE_VERSION
{
	long int times, i, j, dim;
	DVector rvec, svec, wvec, uvec;
    DVector pi, pim1, qi, qim1;
	double alpha, beta, gamma;

#ifdef USE_SPARSE_VERSION
	dim = mat_sp->row_dim;
#else // USE_SPARSE_VERSION
	dim = mat->row_dim;
#endif // USE_SPARSE_VERSION

	set0_dmatrix(trimat);
	rvec = init_dvector(dim);
    svec = init_dvector(dim);
	wvec = init_dvector(dim);
	uvec = init_dvector(dim);
    pi   = init_dvector(dim);
    pim1 = init_dvector(dim);
    qi   = init_dvector(dim);
    qim1 = init_dvector(dim);

	#ifndef USE_RAND_START_LANCZOS
	/* q_1 := sqrt(n)^(-1) * [1 1 ... 1]^T */
    /* p_1 := q_1 */
	beta = sqrt((double)dim);
    gamma = beta;
	for(i = 0; i < dim; i++)
	{
		//set_dvector_i(qvec[0], i, 1.0 / beta);
		set_dvector_i(qi, i, 1.0 / beta);
	}
    //subst_dvector(pvec[0], qvec[0]);
    subst_dvector(pi, qi);
	#else // USE_RAND_START_LANCZOS
	// q, p := rand
	srand(dim);
	for(i = 0; i < dim; i++)
	{
		//set_dvector_i(qvec[0], i, (double)rand() / (double)RAND_MAX);
		//set_dvector_i(pvec[0], i, (double)rand() / (double)RAND_MAX);
		set_dvector_i(qi, i, (double)rand() / (double)RAND_MAX);
		set_dvector_i(pi, i, (double)rand() / (double)RAND_MAX);
    }
	//beta = norm2_dvector(qvec[0]);
	//cmul2_dvector(qvec[0], 1.0 / beta);
	//cmul2_dvector(pvec[0], 1.0 / norm2_dvector(pvec[0]));
	//gamma = ip_dvector(qvec[0], pvec[0]); // / beta;
	beta = norm2_dvector(qi);
	cmul2_dvector(qi, 1.0 / beta);
	cmul2_dvector(qi, 1.0 / norm2_dvector(pi));
	gamma = ip_dvector(qvec[0], pvec[0]); // / beta;
	//subst_dvector(pvec[0], qvec[0]);
	#endif // USE_RAND_START_LANCZOS

    if(pvec != NULL) subst_dvector(pvec[0], pi);
    if(qvec != NULL) subst_dvector(qvec[0], qi);

	/* main loop */
	for(i = 0; i < dim; i++)
	{

#ifdef USE_SPARSE_VERSION
        // w := A * q
		mul_drsmatrix_dvec(wvec, mat_sp, qi);
        // u := A^T * p
		mul_drsmatrixt_dvec(uvec, mat_sp, pi);
#else // USE_SPARSE_VERSION
        // w := A * q
		mul_dmatrixt_dvec(wvec, mat, qi);
        // u := A^T * p
		mul_dmatrixt_dvec(uvec, mat, pi);
#endif // USE_SPARSE_VERSION

        // alpha := (p, w) = (p, A * q)
		alpha = ip_dvector(pi, wvec);
		set_dmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

        // r := w - alpha * q = A * q - alpha * q
        // s := u - alpha * p = A^T * p - alpha * p
		subcmul_dvector(rvec, wvec, alpha, qi);
        subcmul_dvector(svec, uvec, alpha, pi);

        // r := A * q - alpha * q - gamma * q
        // s := A^T * p - alpha * p - beta * p
		if(i > 0)
        {
			subcmul_dvector(rvec, rvec, gamma, qim1);
            subcmul_dvector(svec, svec, beta, pim1);
        }

        // beta := ||r||_2
		beta = norm2_dvector(rvec);

        // gamma := (r, s) / beta
        gamma = ip_dvector(rvec, svec) / beta;

        // p[i - 1] := pi
        // q[i - 1] := qi
        subst_dvector(pim1, pi);
        subst_dvector(qim1, qi);

        // q := 1/beta * r
		cmul_dvector(qi, 1.0 / beta, rvec);
        // p := 1/gamma * s
        cmul_dvector(pi, 1.0 / gamma, svec);    

        if(pvec != NULL) subst_dvector(pvec[i + 1], pi);
        if(qvec != NULL) subst_dvector(qvec[i + 1], qi);

		set_dmatrix_ij(trimat, i, i + 1, gamma);
		set_dmatrix_ij(trimat, i + 1, i, beta);
	}

	free_dvector(rvec);
    free_dvector(svec);
	free_dvector(wvec);
    free_dvector(uvec);
    free_dvector(pi);
    free_dvector(pim1);
    free_dvector(qi);
    free_dvector(qim1);

    return i;
}

#ifdef USE_GMP

// ------------------------ //
// MPFR arbitrary precision //
// ------------------------ //

// Lanczos method to transform real symmetric matrix to symmetric tridiagonal matrix
#ifdef USE_SPARSE_VERSION
long int mpf_slanczos_sp(MPFMatrix trimat, MPFVector *qvec, MPFRSMatrix mat_sp)
#elif USE_SPARSE_D_VERSION
long int mpf_slanczos_sp_d(MPFMatrix trimat, MPFVector *qvec, DSMatrix mat_sp)
#else // USE_SPARSE_VERSION
long int mpf_slanczos(MPFMatrix trimat, MPFVector *qvec, MPFMatrix mat)
#endif // USE_SPARSE_VERSION
{
	long int times, i, j, dim;
	MPFVector rvec, wvec, qi, qim1;
	mpf_t beta, alpha, tmp;

	mpf_init2(beta, trimat->prec);
	mpf_init2(alpha, trimat->prec);
	mpf_init2(tmp, trimat->prec);

#ifdef USE_SPARSE_VERSION
	dim = mat_sp->row_dim;
#elif USE_SPARSE_D_VERSION
	dim = mat_sp->row_dim;
#else // USE_SPARSE_VERSION
	dim = mat->row_dim;
#endif // USE_SPARSE_VERSION

	set0_mpfmatrix(trimat);
	rvec = init2_mpfvector(dim, trimat->prec);
	wvec = init2_mpfvector(dim, trimat->prec);
    qi   = init2_mpfvector(dim, trimat->prec);
    qim1 = init2_mpfvector(dim, trimat->prec);

	/* q_1 = [1 ... 1]^T */
	mpf_set_ui(beta, (unsigned long)dim);
	mpf_sqrt(beta, beta);
	mpf_ui_div(tmp, 1UL, beta);
	for(i = 0; i < dim; i++)
	{
		//set_mpfvector_i(qvec[0], i, tmp);
        set_mpfvector_i(qi, i, tmp);
	}
    if(qvec != NULL) subst_mpfvector(qvec[0], qi);

	/* main loop */
	for(i = 0; i < dim; i++)
	{

#ifdef USE_SPARSE_VERSION
		//mul_mpfrsmatrix_mpfvec(wvec, mat_sp, qvec[i]);
		mul_mpfrsmatrix_mpfvec(wvec, mat_sp, qi);
#elif USE_SPARSE_D_VERSION
		mul_drsmatrix_mpfvec(wvec, mat_sp, qi);
#else // USE_SPARSE_VERSION
		//mul_mpfmatrix_mpfvec(wvec, mat, qvec[i]);
		mul_mpfmatrix_mpfvec(wvec, mat, qi);
#endif // USE_SPARSE_VERSION

		//ip_mpfvector(alpha, qvec[i], wvec);
		ip_mpfvector(alpha, qi, wvec);
        set_mpfmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

		//subcmul_mpfvector(rvec, wvec, alpha, qvec[i]);
		subcmul_mpfvector(rvec, wvec, alpha, qi);
		if(i > 0)
        {
			//subcmul_mpfvector(rvec, rvec, beta, qvec[i - 1]);
            subcmul_mpfvector(rvec, rvec, beta, qim1);
        }

        cmul_mpfvector(qi, tmp, rvec);

		norm2_mpfvector(beta, rvec);
		mpf_ui_div(tmp, 1UL, beta);
		//cmul_mpfvector(qvec[i + 1], tmp, rvec);
        subst_mpfvector(qim1, qi);
        if(qvec != NULL) subst_mpfvector(qvec[i + 1], qi);

		set_mpfmatrix_ij(trimat, i, i + 1, beta);
		set_mpfmatrix_ij(trimat, i + 1, i, beta);
	}

	free_mpfvector(rvec);
	free_mpfvector(wvec);
    free_mpfvector(qi);
    free_mpfvector(qim1);
	mpf_clear(alpha);
	mpf_clear(beta);
	mpf_clear(tmp);

    return i;
}

// Lanczos method to transform real matrix to tridiagonal matrix
#ifdef USE_SPARSE_VERSION
//long int mpf_lanczos_sp(MPFMatrix trimat, MPFVector qvec[], MPFVector pvec[], MPFRSMatrix mat_sp)
long int mpf_lanczos_sp(MPFMatrix trimat, MPFVector *qvec, MPFVector *pvec, MPFRSMatrix mat_sp)
#elif USE_SPARSE_D_VERSION
long int mpf_lanczos_sp_d(MPFMatrix trimat, MPFVector *qvec, MPFVector *pvec, DRSMatrix mat_sp)
#else // USE_SPARSE_VERSION
//long int mpf_lanczos(MPFMatrix trimat, MPFVector qvec[], MPFVector pvec[], MPFMatrix mat)
long int mpf_lanczos(MPFMatrix trimat, MPFVector *qvec, MPFVector *pvec, MPFMatrix mat)
#endif // USE_SPARSE_VERSION
{
	long int times, i, j, dim;
	MPFVector rvec, svec, wvec, uvec;
    MPFVector pi, pim1, qi, qim1;
	mpf_t alpha, beta, gamma, tmp;

	mpf_init2(alpha, trimat->prec);
	mpf_init2(beta, trimat->prec);
	mpf_init2(gamma, trimat->prec);
	mpf_init2(tmp, trimat->prec);

#ifdef USE_SPARSE_VERSION
	dim = mat_sp->row_dim;
#else // USE_SPARSE_VERSION
	dim = mat->row_dim;
#endif // USE_SPARSE_VERSION

	set0_mpfmatrix(trimat);
	rvec = init2_mpfvector(dim, trimat->prec);
	svec = init2_mpfvector(dim, trimat->prec);
	wvec = init2_mpfvector(dim, trimat->prec);
	uvec = init2_mpfvector(dim, trimat->prec);
	pi   = init2_mpfvector(dim, trimat->prec);
	pim1 = init2_mpfvector(dim, trimat->prec);
	qi   = init2_mpfvector(dim, trimat->prec);
	qim1 = init2_mpfvector(dim, trimat->prec);

	/* q_1 = [1 ... 1]^T */
    /* p_1 := p_1 */
	mpf_set_ui(beta, (unsigned long)dim);
	mpf_sqrt(beta, beta);
	mpf_ui_div(tmp, 1UL, beta);
    mpf_set(gamma, beta);
	for(i = 0; i < dim; i++)
	{
		//set_mpfvector_i(pvec[0], i, tmp);
		//set_mpfvector_i(qvec[0], i, tmp);
		set_mpfvector_i(pi, i, tmp);
		set_mpfvector_i(qi, i, tmp);
	}

    if(pvec != NULL) subst_mpfvector(pvec[0], pi);
    if(qvec != NULL) subst_mpfvector(qvec[0], qi);

	/* main loop */
	for(i = 0; i < dim; i++)
	{
#ifdef USE_SPARSE_VERSION
		//mul_mpfrsmatrix_mpfvec(wvec, mat_sp, qvec[i]);
		//mul_mpfrsmatrixt_mpfvec(uvec, mat_sp, pvec[i]);
		mul_mpfrsmatrix_mpfvec(wvec, mat_sp, qi);
		mul_mpfrsmatrixt_mpfvec(uvec, mat_sp, pi);
#elif USE_SPARSE_D_VERSION
		//mul_mpfrsmatrix_mpfvec(wvec, mat_sp, qvec[i]);
		//mul_mpfrsmatrixt_mpfvec(uvec, mat_sp, pvec[i]);
		mul_drsmatrix_mpfvec(wvec, mat_sp, qi);
		mul_drsmatrixt_mpfvec(uvec, mat_sp, pi);
#else // USE_SPARSE_VERSION
		//mul_mpfmatrix_mpfvec(wvec, mat, qvec[i]);
		//mul_mpfmatrixt_mpfvec(uvec, mat, pvec[i]);
		mul_mpfmatrix_mpfvec(wvec, mat, qi);
		mul_mpfmatrixt_mpfvec(uvec, mat, pi);
#endif // USE_SPARSE_VERSION
		//ip_mpfvector(alpha, pvec[i], wvec);
		ip_mpfvector(alpha, pi, wvec);
		set_mpfmatrix_ij(trimat, i, i, alpha);
		if(i >= dim - 1)
			break;

		//subcmul_mpfvector(rvec, wvec, alpha, qvec[i]);
		//subcmul_mpfvector(svec, uvec, alpha, pvec[i]);
		subcmul_mpfvector(rvec, wvec, alpha, qi);
		subcmul_mpfvector(svec, uvec, alpha, pi);

		if(i > 0)
        {
			//subcmul_mpfvector(rvec, rvec, gamma, qvec[i - 1]);
			//subcmul_mpfvector(svec, svec, beta, pvec[i - 1]);
			subcmul_mpfvector(rvec, rvec, gamma, qim1);
			subcmul_mpfvector(svec, svec, beta, pim1);
        }
		norm2_mpfvector(beta, rvec);
		mpf_ui_div(tmp, 1UL, beta);
		//cmul_mpfvector(qvec[i + 1], tmp, rvec);
        subst_mpfvector(qim1, qi);
		cmul_mpfvector(qi, tmp, rvec);

		ip_mpfvector(gamma, rvec, svec);
        mpf_div(gamma, gamma, beta);
		mpf_ui_div(tmp, 1UL, gamma);
		//cmul_mpfvector(pvec[i + 1], tmp, svec);
        subst_mpfvector(pim1, pi);
		cmul_mpfvector(pi, tmp, svec);

        if(pvec != NULL) subst_mpfvector(pvec[i + 1], pi);
        if(qvec != NULL) subst_mpfvector(qvec[i + 1], qi);

		set_mpfmatrix_ij(trimat, i, i + 1, gamma);
		set_mpfmatrix_ij(trimat, i + 1, i, beta);
	}

	free_mpfvector(rvec);
    free_mpfvector(svec);
	free_mpfvector(wvec);
    free_mpfvector(uvec);
    free_mpfvector(pi);
    free_mpfvector(pim1);
    free_mpfvector(qi);
    free_mpfvector(qim1);
	mpf_clear(alpha);
	mpf_clear(beta);
    mpf_clear(gamma);
	mpf_clear(tmp);

    return i;
}
#endif // USE_GMP
