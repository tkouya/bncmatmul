/********************************************************************************/
/* test_iterative_ref.c: Simple Estimation of Condition Number                  */
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
//#include "bnc.h"
#include "matmul_strassen.h"

#ifdef _OPENMP
#include "bncomp.h"
#endif // _OPENMP

#include "get_secv.h"
#include "get_linear_system.h"

/* coef_f_itr = rho_f(n) * cond_1(A) * eps_d / (1 - psi_f(n) * cond_1(A) * eps_s) */
double coef_f_itr(double cond1, double eps_d, double eps_s, long int dim)
{
	double rho_f, psi_f;

	rho_f = sqrt((double)dim);
	psi_f = sqrt((double)dim);

	return rho_f * cond1 * eps_d / (1.0 - psi_f * cond1 * eps_s);
}

/* Condition Number Estimator : cond_1(A) = ||A||_1 * ||A^(-1)||_1 */
/* Based on LINPACK argorithm */
/* norm1_orgmat: ||A||_1 */
/* lumat: LU(=A) decomposed matrix */
/* row_ch, col_ch: indeces for row and column */
double condest_dmatrix_C(double norm1_orgmat, DMatrix lumat, long int row_ch[], long int col_ch[])
{
	double condition_num, t, evec_i;
	DVector zvec, yvec, xvec;
	long int dim, i, j;
	long int row_i, col_i;

	/* Initialize */
	dim = lumat->col_dim;
	zvec = init_dvector(dim);
	yvec = init_dvector(dim);
	xvec = init_dvector(dim);

	/* solve U^T * z = e */
	evec_i = 1.0;
	set_dvector_i(zvec, 0, evec_i / get_dmatrix_ij(lumat, row_ch[0], col_ch[0]));
	for(i = 1; i < dim; i++)
	{
		row_i = row_ch[i];
		col_i = col_ch[i];

		/* t = sum^(i-1)_{j=0} u_ji * z_j */
		t = 0.0;
		for(j = 0; j <= (i - 1); j++)
		{
			t += get_dmatrix_ij(lumat, row_ch[j], col_i) * get_dvector_i(zvec, j);
		}
		
		/* e[i] = sign(t) */
		if(t >= 0)
			evec_i = 1.0;
		else
			evec_i = -1.0;
		
		/* z[i] = (e[i] - t) / u[i][i] */
		set_dvector_i(zvec, i, (evec_i - t) / get_dmatrix_ij(lumat, row_i, col_i));
	}

	printf("||z_vec||_1 = %25.17e\n", norm1_dvector(zvec));

	/* solve L^T (P * y) = z */
	/* L[i][i] = 1 */
	for(i = (dim - 1); i >= 0; i--)
	{
		row_i = row_ch[i];
		col_i = col_ch[i];

		t = get_dvector_i(zvec, i);
		for(j = i + 1; j < dim; j++)
			t -= get_dmatrix_ij(lumat, row_ch[j], col_i) * get_dvector_i(zvec, j);
		set_dvector_i(yvec, col_i, t);
	}

	printf("||y_vec||_1 = %25.17e\n", norm1_dvector(yvec));

	/* solve A * x = y */
	SolveDLSC(xvec, lumat, yvec, row_ch, col_ch);

	printf("||x_vec||_1 = %25.17e\n", norm1_dvector(xvec));

	/* cond_1(A) = ||A||_1 * ||A^(-1)||_1 \approx ||A||_1 * ||x||_1 / ||y||_1 */
	condition_num = norm1_orgmat * norm1_dvector(xvec) / norm1_dvector(yvec);

	/* free */
	free_dvector(xvec);
	free_dvector(yvec);
	free_dvector(zvec);

	return condition_num;
}

#ifdef USE_GMP
void mpf_get_meps(mpf_t eps, mpf_t base_num)
{
	unsigned long prec, base;

	// bits
	prec = mpf_get_prec(base_num);
	base = 2UL; // binary floating-point number

	mpf_set_ui(eps, 1UL);
	while(prec-- > 0)
		mpf_div_ui(eps, eps, base);

	return;
}
void mpf_get_meps_base_prec(mpf_t eps, unsigned long base, unsigned long prec)
{
	mpf_set_ui(eps, 1UL);
	while(prec-- > 0)
		mpf_div_ui(eps, eps, base);

	return;
}


/* coef_f_itr = rho_f(n) * cond_1(A) * eps_l/ (1 - psi_f(n) * cond_1(A) * eps_s) */
void mpfcoef_f_itr(mpf_t coef_f_itr, mpf_t cond1, unsigned long long_prec, unsigned long short_prec, long int dim)
{
	unsigned long prec;
	mpf_t alpha_f, beta_f, eps_l, eps_s, tmp, tmp1;
	mpf_t phi, varphi_1, varphi_2;

	prec = mpf_get_prec(coef_f_itr);

	/* Initialize */
	mpf_init2(phi, prec);
	mpf_init2(varphi_1, prec);
	mpf_init2(varphi_2, prec);
	mpf_init2(alpha_f, prec);
	mpf_init2(beta_f, prec);
	mpf_init2(eps_l, prec);
	mpf_init2(eps_s, prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);

	mpf_sqrt_ui(phi, (unsigned long)dim);
	mpf_sqrt_ui(varphi_1, (unsigned long)dim);
	mpf_sqrt_ui(varphi_2, (unsigned long)dim);

	mpf_get_meps_base_prec(eps_l, 2UL, long_prec);
	mpf_get_meps_base_prec(eps_s, 2UL, short_prec);

	/* alpha_f := (phi * cond1 * eps_s) / (1 - phi * cond1 * eps_s) + 2 * varphi_1 * cond1 * eps_l + varphi_2 * eps_l + 2 * (1 + varphi_1 * eps_l) * varphi_2 * cond1 * eps_l */
	mpf_mul(tmp, phi, cond1);
	mpf_mul(tmp, tmp, eps_s);
	mpf_ui_sub(tmp1, 1UL, tmp);
	mpf_div(alpha_f, tmp, tmp1);

	mpf_mul_ui(tmp, varphi_1, 2UL);
	mpf_mul(tmp, tmp, cond1);
	mpf_mul(tmp, tmp, eps_l);
	mpf_add(alpha_f, alpha_f, tmp);

	mpf_mul(tmp, varphi_2, eps_l);
	mpf_add(alpha_f, alpha_f, tmp);

	mpf_mul(tmp, varphi_1, eps_l);
	mpf_add_ui(tmp, tmp, 1UL);
	mpf_mul_ui(tmp, tmp, 2UL);
	mpf_mul(tmp, tmp, varphi_2);
	mpf_mul(tmp, tmp, cond1);
	mpf_mul(tmp, tmp, eps_l);
	mpf_add(alpha_f, alpha_f, tmp);

	printf("S(bits): %ld\n", short_prec); printf("eps_s: "); mpf_out_str(stdout, 10, 5, eps_s); printf("\n");
	printf("L(bits): %ld\n", long_prec); printf("eps_l: "); mpf_out_str(stdout, 10, 5, eps_l); printf("\n");
	printf("alpha_f: "); mpf_out_str(stdout, 10, 5, alpha_f); printf("\n");

	/* beta_f := 4 * varphi_1 * cond1 * eps_l + varphi_2 * eps_l + 4 * (1 + varphi_1  * eps_l) * varphi_2 * cond1 * eps_l */
	mpf_mul_ui(beta_f, varphi_1, 4UL);
	mpf_mul(beta_f, beta_f, eps_l);

	mpf_mul(tmp, varphi_2, eps_l);
	mpf_add(beta_f, beta_f, tmp);

	mpf_mul(tmp, varphi_1, eps_l);
	mpf_add_ui(tmp, tmp, 1UL);
	mpf_mul_ui(tmp, tmp, 4UL);
	mpf_mul(tmp, tmp, varphi_2);
	mpf_mul(tmp, tmp, cond1);
	mpf_mul(tmp, tmp, eps_l);
	mpf_add(beta_f, beta_f, tmp);
	
	printf("beta_f: "); mpf_out_str(stdout, 10, 5, beta_f); printf("\n");

	/* coef_f_itr := beta_f / (1 - alpha_f) */
	mpf_ui_sub(tmp, 1UL, alpha_f);
	mpf_div(coef_f_itr, beta_f, tmp);

	printf("beta_f / (1 - alpha_f): "); mpf_out_str(stdout, 10, 5, coef_f_itr); printf("\n");

	/* free */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(alpha_f);
	mpf_clear(phi);
	mpf_clear(varphi_1);
	mpf_clear(varphi_2);
	mpf_clear(beta_f);
	mpf_clear(eps_l);
	mpf_clear(eps_s);
}
/* Condition Number Estimator : cond_1(A) = ||A||_1 * ||A||_1 */
/* Based on LINPACK argorithm */
/* norm1_orgmat: ||A||_1 */
/* lumat: LU(=A) decomposed matrix */
/* row_ch, col_ch: indeces for row and column */
void condest_mpfmatrix_C(mpf_t condition_num, mpf_t norm1_orgmat, MPFMatrix lumat, long int row_ch[], long int col_ch[])
{
	unsigned long prec;

	mpf_t t, evec_i, tmp;
	MPFVector zvec, yvec, xvec;
	long int dim, i, j;
	long int row_i, col_i;

	/* Initialize */
	prec = mpf_get_prec(condition_num);
	dim = lumat->col_dim;

	mpf_init2(t, prec);
	mpf_init2(evec_i, prec);
	mpf_init2(tmp, prec);

	zvec = init2_mpfvector(dim, prec);
	yvec = init2_mpfvector(dim, prec);
	xvec = init2_mpfvector(dim, prec);

	/* solve U^T * z = e */
	mpf_set_ui(evec_i, 1UL);
	mpf_div(evec_i, evec_i, get_mpfmatrix_ij(lumat, row_ch[0], col_ch[0]));
	set_mpfvector_i(zvec, 0, evec_i);
	for(i = 1; i < dim; i++)
	{
		row_i = row_ch[i];
		col_i = col_ch[i];

		/* t = sum^(i-1)_{j=0} u_ji * z_j */
		mpf_set_ui(t, 0UL);
		for(j = 0; j <= (i - 1); j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lumat, row_ch[j], col_i), get_mpfvector_i(zvec, j));
			mpf_add(t, t, tmp);
		}
		
		/* e[i] = sign(t) */
		mpf_set_ui(evec_i, 1UL);
		if(mpf_cmp_ui(t, 0UL) < 0)
			mpf_neg(evec_i, evec_i);

		/* z[i] = (e[i] - t) / u[i][i] */
		mpf_sub(tmp, evec_i, t);
		mpf_div(tmp, tmp, get_mpfmatrix_ij(lumat, row_i, col_i));
		set_mpfvector_i(zvec, i, tmp);
	}

	//printf("||z_vec||_1 = %25.17e\n", norm1_dvector(zvec));

	/* solve L^T (P * y) = z */
	/* L[i][i] = 1 */
	for(i = (dim - 1); i >= 0; i--)
	{
		row_i = row_ch[i];
		col_i = col_ch[i];

		mpf_set(t, get_mpfvector_i(zvec, i));
		for(j = i + 1; j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lumat, row_ch[j], col_i), get_mpfvector_i(zvec, j));
			mpf_sub(t, t, tmp);
		}
		set_mpfvector_i(yvec, col_i, t);
	}

	//printf("||y_vec||_1 = %25.17e\n", norm1_dvector(yvec));

	/* solve A * x = y */
	SolveMPFLSC(xvec, lumat, yvec, row_ch, col_ch);

	//printf("||x_vec||_1 = %25.17e\n", norm1_dvector(xvec));

	/* cond_1(A) = ||A||_1 * ||A^(-1)||_1 \approx ||A||_1 * ||x||_1 / ||y||_1 */
	norm1_mpfvector(condition_num, xvec);
	mpf_mul(condition_num, norm1_orgmat, condition_num);
	norm1_mpfvector(tmp, yvec);
	mpf_div(condition_num, condition_num, tmp);

	/* free */
	mpf_clear(t);
	mpf_clear(evec_i);
	mpf_clear(tmp);

	free_mpfvector(xvec);
	free_mpfvector(yvec);
	free_mpfvector(zvec);

	return;
}

#if 0
/* get residual in mpf_t precision */
/* r := b - A * x */
void residual_mpfmat_mpfvec(MPFVector r, MPFVector b, MPFMatrix mat, MPFVector x)
{
	mul_mpfmatrix_mpfvec(r, mat, x);
	sub_mpfvector(r, b, r);
}
#endif // 0

/* get residual in DD precision */
/* r := b - A * x */
void residual_ddmat_ddvec(DDVector r, DDVector b, DDMatrix mat, DDVector x)
{
#ifdef _OPENMP
	_bncomp_mul_ddmatrix_ddvec(r, mat, x);
	_bncomp_sub_ddvector(r, b, r);
#else // _OPENMP
	mul_ddmatrix_ddvec(r, mat, x);
	sub_ddvector(r, b, r);
#endif // _OPENMP
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int ddgesirsv(DDVector x, DDMatrix a, DDVector b, double rtol, double atol, long int maxtimes)
{
	long int dim, i;
	long int *af_ch;
	DMatrix af;
	DVector bf, xf, resf, zf;
	DDVector res, z;
	static double tmp[DDSIZE], norm_a[DDSIZE], norm_x[DDSIZE], norm_res[DDSIZE];

	// Initialize
	dim = x->dim;

	af = init_dmatrix(dim, dim);
	bf = init_dvector(dim);
	xf = init_dvector(dim);

	res = init_ddvector(dim);
	z = init_ddvector(dim);

	resf = init_dvector(dim);
	zf = init_dvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_ddmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_dmatrix_ddmat(af, a);
	subst_dvector_ddvec(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	DLUdecompP(af, af_ch);

	// Apply back-solve in short precision with short precision factors
	SolveDLSP(xf, af, bf, af_ch);

	// Promote te solution from short precision to long precision
	subst_ddvector_dvec(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_ddmat_ddvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_ddvector(norm_x, x);
		norm2_ddvector(norm_res, res);

		// normalization
		rdd_ui_div(tmp, 1UL, norm_res);
		cmul2_ddvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		rdd_set_ui(tmp, (unsigned long)dim);
		rdd_sqrt(tmp, tmp);
		rdd_mul_d(tmp, tmp, rtol);
		rdd_mul(tmp, tmp, norm_a);
		rdd_mul(tmp, tmp, norm_x);
		rdd_add_d(tmp, tmp, atol);
		if(rdd_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_dvector_ddvec(resf, res);

		// Back-solve on short precision residual and short precision factors
		SolveDLSP(zf, af, resf, af_ch);

		// Promote the correction from short precision to long precision
		subst_ddvector_dvec(z, zf);

		// Update solution in long precision
		//add_ddvector(x, x, z);
		add_cmul_ddvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		rdd_out_str(norm_res);
		//print_ddvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		DDLUdecomp(a);
		SolveDDLS(x, a, b);
	}

	// Clear
	free_dmatrix(af);
	free_dvector(bf);
	free_dvector(xf);
	free_dvector(resf);
	free_dvector(zf);
	free_ddvector(res);
	free_ddvector(z);
	free(af_ch);

	return i;
}

long int dditerative_refinement(DDVector x, DDMatrix a, DDVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	double rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = 32;
	short_prec = 16;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	rtol = 1.0e-30;
	atol = 0.0;

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : %10.2e\n", rtol);
	printf("atol            : %10.2e\n", atol);

	times = ddgesirsv(x, a, b, rtol, atol, maxtimes);

	return times;
}


// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int mpfgesirsv_d(MPFVector x, MPFMatrix a, MPFVector b, mpf_t rtol, mpf_t atol, long int maxtimes, unsigned long long_prec)
{
	unsigned long short_prec = (long_prec > 32) ? (long_prec / 2) : long_prec;
	long int dim, i;
	long int *af_ch;
	DMatrix af;
	DVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t tmp, norm_a, norm_x, norm_res;

	// Initialize
	dim = x->dim;
	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);

	af = init_dmatrix(dim, dim);
	bf = init_dvector(dim);
	xf = init_dvector(dim);

	res = init2_mpfvector(dim, long_prec);
	z = init2_mpfvector(dim, long_prec);

	resf = init_dvector(dim);
	zf = init_dvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_dmatrix_mpfmat(af, a);
	subst_dvector_mpfvec(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	DLUdecompP(af, af_ch);

	// Apply back-solve in short precision with short precision factors
	SolveDLSP(xf, af, bf, af_ch);

	// Promote te solution from short precision to long precision
	subst_mpfvector_dvec(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_mpfmat_mpfvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

		// normalization of residual
		mpf_ui_div(tmp, 1UL, norm_res);
		cmul2_mpfvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_dvector_mpfvec(resf, res);

		// Back-solve on short precision residual and short precision factors
		SolveDLSP(zf, af, resf, af_ch);

		// Promote the correction from short precision to long precision
		subst_mpfvector_dvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_mpfvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		mpf_out_str(stdout, 10, 15, norm_res);printf("\n");
		//print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
	}

	// Clear
	free_dmatrix(af);
	free_dvector(bf);
	free_dvector(xf);
	free_dvector(resf);
	free_dvector(zf);
	free_mpfvector(res);
	free_mpfvector(z);
	free(af_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);

	return i;
}

long int mpf_iterative_refinement_d(MPFVector x, MPFMatrix a, MPFVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	mpf_t rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
	short_prec = 16;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	times = mpfgesirsv_d(x, a, b, rtol, atol, maxtimes, long_prec);

	mpf_clear(rtol);
	mpf_clear(atol);

	return times;
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int mpfgesirsv_dd(MPFVector x, MPFMatrix a, MPFVector b, mpf_t rtol, mpf_t atol, long int maxtimes, unsigned long long_prec)
{
	unsigned long short_prec = (long_prec > 65) ? (long_prec / 2) : long_prec;
	long int dim, i;
	long int *af_ch;
	DDMatrix af;
	DDVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t tmp, norm_a, norm_x, norm_res;
	double stime, etime;

	// Initialize
	dim = x->dim;
	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);

	af = init_ddmatrix(dim, dim);
	bf = init_ddvector(dim);
	xf = init_ddvector(dim);

	res = init2_mpfvector(dim, long_prec);
	z = init2_mpfvector(dim, long_prec);

	resf = init_ddvector(dim);
	zf = init_ddvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_ddmatrix_mpfmat(af, a);
	subst_ddvector_mpfvec(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	stime = get_real_secv();
#ifdef _OPENMP
	DDLUdecompPM_omp(af, af_ch);
	SolveDDLSPM(xf, af, bf, af_ch);
#else // _OPENMP
	//DDLUdecompP(af, af_ch);
	DDLUdecompPM(af, af_ch);
	// Apply back-solve in short precision with short precision factors
	//SolveDDLSP(xf, af, bf, af_ch);
	SolveDDLSPM(xf, af, bf, af_ch);
#endif // _OPENMP
	etime = get_real_secv() - stime;
	printf("DD Direct sec = %15.8g\n", etime);

	// Promote te solution from short precision to long precision
	subst_mpfvector_ddvec(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
	#ifdef _OPENMP
		_bncomp_mul_mpfmatrix_mpfvec(z, a, x); // z := Ax
		_bncomp_sub_mpfvector(res, b, z, dim); // res := b - z
	#else // _OPENMP
		residual_mpfmat_mpfvec(res, b, a, x);
	#endif // _OPENMP

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

		// normalization of residual
		mpf_ui_div(tmp, 1UL, norm_res);
		cmul2_mpfvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_ddvector_mpfvec(resf, res);

		// Back-solve on short precision residual and short precision factors
#ifdef _OPENMP
		SolveDDLSPM(zf, af, resf, af_ch);
#else // _OPENMP
		//SolveDDLSP(zf, af, resf, af_ch);
		SolveDDLSPM(zf, af, resf, af_ch);
#endif // _OPENMP

		// Promote the correction from short precision to long precision
		subst_mpfvector_ddvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_mpfvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		mpf_out_str(stdout, 10, 15, norm_res);printf("\n");
		//print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		#ifdef _OPENMP
		MPFLUdecomp_omp(a);
		SolveMPFLS(x, a, b);
		#else // _OPENMP
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
		#endif // _OPENMP
	}

	// Clear
	free_ddmatrix(af);
	free_ddvector(bf);
	free_ddvector(xf);
	free_ddvector(resf);
	free_ddvector(zf);
	free_mpfvector(res);
	free_mpfvector(z);
	free(af_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);

	return i;
}


long int mpf_iterative_refinement_dd(MPFVector x, MPFMatrix a, MPFVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	mpf_t rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
	short_prec = 106;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	times = mpfgesirsv_dd(x, a, b, rtol, atol, maxtimes, long_prec);

	mpf_clear(rtol);
	mpf_clear(atol);

	return times;
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int mpfgesirsv_td(MPFVector x, MPFMatrix a, MPFVector b, mpf_t rtol, mpf_t atol, long int maxtimes, unsigned long long_prec)
{
	unsigned long short_prec = (long_prec > 65) ? (long_prec / 2) : long_prec;
	long int dim, i;
	long int *af_ch;
	TDMatrix af;
	TDVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t tmp, norm_a, norm_x, norm_res;
	double stime, etime;

	// Initialize
	dim = x->dim;
	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);

	af = init_tdmatrix(dim, dim);
	bf = init_tdvector(dim);
	xf = init_tdvector(dim);

	res = init2_mpfvector(dim, long_prec);
	z = init2_mpfvector(dim, long_prec);

	resf = init_tdvector(dim);
	zf = init_tdvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_tdmatrix_mpfmat(af, a);
	subst_tdvector_mpfvec(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	stime = get_real_secv();
#ifdef _OPENMP
	TDLUdecompPM_omp(af, af_ch);
	SolveTDLSPM(xf, af, bf, af_ch);
#else // _OPENMP
	TDLUdecompP(af, af_ch);
	// Apply back-solve in short precision with short precision factors
	SolveTDLSP(xf, af, bf, af_ch);
#endif // _OPENMP
	etime = get_real_secv() - stime;
	printf("TD Direct sec = %15.8g\n", etime);

	// Promote te solution from short precision to long precision
	subst_mpfvector_tdvec(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
	#ifdef _OPENMP
		_bncomp_mul_mpfmatrix_mpfvec(z, a, x); // z := Ax
		_bncomp_sub_mpfvector(res, b, z, dim); // res := b - z
	#else // _OPENMP
		residual_mpfmat_mpfvec(res, b, a, x);
	#endif // _OPENMP

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

		// normalization of residual
		mpf_ui_div(tmp, 1UL, norm_res);
		cmul2_mpfvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_tdvector_mpfvec(resf, res);

		// Back-solve on short precision residual and short precision factors
#ifdef _OPENMP
		SolveTDLSPM(zf, af, resf, af_ch);
#else // _OPENMP
		SolveTDLSP(zf, af, resf, af_ch);
#endif // _OPENMP

		// Promote the correction from short precision to long precision
		subst_mpfvector_tdvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_mpfvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		mpf_out_str(stdout, 10, 15, norm_res);printf("\n");
		//print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		#ifdef _OPENMP
		MPFLUdecomp_omp(a);
		SolveMPFLS(x, a, b);
		#else // _OPENMP
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
		#endif // _OPENMP
	}

	// Clear
	free_tdmatrix(af);
	free_tdvector(bf);
	free_tdvector(xf);
	free_tdvector(resf);
	free_tdvector(zf);
	free_mpfvector(res);
	free_mpfvector(z);
	free(af_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);

	return i;
}

long int mpf_iterative_refinement_td(MPFVector x, MPFMatrix a, MPFVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	mpf_t rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
	short_prec = 159;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	times = mpfgesirsv_td(x, a, b, rtol, atol, maxtimes, long_prec);

	mpf_clear(rtol);
	mpf_clear(atol);

	return times;
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int mpfgesirsv_qd(MPFVector x, MPFMatrix a, MPFVector b, mpf_t rtol, mpf_t atol, long int maxtimes, unsigned long long_prec)
{
	unsigned long short_prec = (long_prec > 65) ? (long_prec / 2) : long_prec;
	long int dim, i;
	long int *af_ch;
	QDMatrix af;
	QDVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t tmp, norm_a, norm_x, norm_res;
	double stime, etime;

	// Initialize
	dim = x->dim;
	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);

	af = init_qdmatrix(dim, dim);
	bf = init_qdvector(dim);
	xf = init_qdvector(dim);

	res = init2_mpfvector(dim, long_prec);
	z = init2_mpfvector(dim, long_prec);

	resf = init_qdvector(dim);
	zf = init_qdvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_qdmatrix_mpfmat(af, a);
	subst_qdvector_mpfvec(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	stime = get_real_secv();
#ifdef _OPENMP
	QDLUdecompPM_omp(af, af_ch);
	SolveQDLSPM(xf, af, bf, af_ch);
#else // _OPENMP
	QDLUdecompP(af, af_ch);
	// Apply back-solve in short precision with short precision factors
	SolveQDLSP(xf, af, bf, af_ch);
#endif // _OPENMP
	etime = get_real_secv() - stime;
	printf("QD Direct sec = %15.8g\n", etime);

	// Promote te solution from short precision to long precision
	subst_mpfvector_qdvec(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
	#ifdef _OPENMP
		_bncomp_mul_mpfmatrix_mpfvec(z, a, x); // z := Ax
		_bncomp_sub_mpfvector(res, b, z, dim); // res := b - z
	#else // _OPENMP
		residual_mpfmat_mpfvec(res, b, a, x);
	#endif // _OPENMP

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

		// normalization of residual
		mpf_ui_div(tmp, 1UL, norm_res);
		cmul2_mpfvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_qdvector_mpfvec(resf, res);

		// Back-solve on short precision residual and short precision factors
#ifdef _OPENMP
		SolveQDLSPM(zf, af, resf, af_ch);
#else // _OPENMP
		SolveQDLSP(zf, af, resf, af_ch);
#endif // _OPENMP

		// Promote the correction from short precision to long precision
		subst_mpfvector_qdvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_mpfvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		mpf_out_str(stdout, 10, 15, norm_res);printf("\n");
		//print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		#ifdef _OPENMP
		MPFLUdecomp_omp(a);
		SolveMPFLS(x, a, b);
		#else // _OPENMP
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
		#endif // _OPENMP
	}

	// Clear
	free_qdmatrix(af);
	free_qdvector(bf);
	free_qdvector(xf);
	free_qdvector(resf);
	free_qdvector(zf);
	free_mpfvector(res);
	free_mpfvector(z);
	free(af_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);

	return i;
}

long int mpf_iterative_refinement_qd(MPFVector x, MPFMatrix a, MPFVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	mpf_t rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
	short_prec = 212; //64;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	times = mpfgesirsv_qd(x, a, b, rtol, atol, maxtimes, long_prec);

	mpf_clear(rtol);
	mpf_clear(atol);

	return times;
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
long int mpfgesirsv(MPFVector x, MPFMatrix a, MPFVector b, mpf_t rtol, mpf_t atol, long int maxtimes, unsigned long long_prec, unsigned long short_prec)
{
	long int dim, i;
	long int *af_ch;
	MPFMatrix af;
	MPFVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t tmp, norm_a, norm_x, norm_res;

	// Initialize
	dim = x->dim;
	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);

	af = init2_mpfmatrix(dim, dim, short_prec);
	bf = init2_mpfvector(dim, short_prec);
	xf = init2_mpfvector(dim, short_prec);

	res = init2_mpfvector(dim, long_prec);

	resf = init2_mpfvector(dim, short_prec);
	z = init2_mpfvector(dim, long_prec);
	zf = init2_mpfvector(dim, short_prec);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_mpfmatrix(af, a);
	subst_mpfvector(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
	MPFLUdecompP(af, af_ch);

	// Apply back-solve in short precision with short precision factors
	SolveMPFLSP(xf, af, bf, af_ch);

	// Promote te solution from short precision to long precision
	subst_mpfvector(x, xf);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_mpfmat_mpfvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_mpfvector(resf, res);

		// Back-solve on short precision residual and short precision factors
		SolveMPFLSP(zf, af, resf, af_ch);

		// Promote the correction from short precision to long precision
		subst_mpfvector(z, zf);

		// Update solution in long precision
		add_mpfvector(x, x, z);

		// for debug
		printf("times: %ld\n", i);
		mpf_out_str(stdout, 10, 15, norm_res);
		//print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
	}

	// Clear
	free_mpfmatrix(af);
	free_mpfvector(bf);
	free_mpfvector(xf);
	free_mpfvector(resf);
	free_mpfvector(res);
	free_mpfvector(zf);
	free_mpfvector(z);
	free(af_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);

	return i;
}

long int mpf_iterative_refinement(MPFVector x, MPFMatrix a, MPFVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	mpf_t rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
//	short_prec = long_prec * 2 / 3;
	short_prec = long_prec / 2;
//	short_prec = long_prec * 2 / 5;
//	short_prec = long_prec / 3;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	times = mpfgesirsv(x, a, b, rtol, atol, maxtimes, long_prec, short_prec);

	mpf_clear(rtol);
	mpf_clear(atol);

	return times;
}

#if 0
int mpf_iterative_refinement(MPFVector x, MPFMatrix a, MPFVector b)
{
	unsigned long short_prec, long_prec;
	long int dim, i, maxtimes;
	long int *af_row_ch, *af_col_ch;
	MPFMatrix af;
	MPFVector bf, xf, resf, zf;
	MPFVector res, z;
	mpf_t rtol, atol;
	mpf_t tmp, norm_a, norm_x, norm_res, norm_res0;

	// Initialize
	dim = x->dim;
	long_prec = x->prec;
//	short_prec = long_prec * 2 / 3;
	short_prec = long_prec / 2;
//	short_prec = long_prec * 2 / 5;
//	short_prec = long_prec / 3;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	mpf_init2(rtol, short_prec);
	mpf_init2(atol, short_prec);
	mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	mpf_set_ui(atol, 0UL);

	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : "); mpf_out_str(stdout, 10, 2, rtol); printf("\n");
	printf("atol            : "); mpf_out_str(stdout, 10, 2, atol); printf("\n");

	mpf_init2(tmp, short_prec);
	mpf_init2(norm_a, short_prec);
	mpf_init2(norm_x, short_prec);
	mpf_init2(norm_res, short_prec);
	mpf_init2(norm_res0, short_prec);

	af = init2_mpfmatrix(dim, dim, short_prec);
	bf = init2_mpfvector(dim, short_prec);
	xf = init2_mpfvector(dim, short_prec);

	res = init2_mpfvector(dim, long_prec);

	resf = init2_mpfvector(dim, short_prec);
	z = init2_mpfvector(dim, long_prec);
	zf = init2_mpfvector(dim, short_prec);
	af_row_ch = (long int *)calloc(sizeof(long int), dim);
	af_col_ch = (long int *)calloc(sizeof(long int), dim);

	normf_mpfmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_mpfmatrix(af, a);
	subst_mpfvector(bf, b);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
//	MPFLUdecompC(af, af_row_ch, af_col_ch);
	MPFLUdecompP(af, af_row_ch);

	// Apply back-solve in short precision with short precision factors
//	SolveMPFLSC(xf, af, bf, af_row_ch, af_col_ch);
	SolveMPFLSP(xf, af, bf, af_row_ch);

	// Promote te solution from short precision to long precision
	subst_mpfvector(x, xf);

	// Compute residual in long precision
	residual_mpfmat_mpfvec(res, b, a, x);
	norm2_mpfvector(norm_res0, res);

	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_mpfmat_mpfvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_mpfvector(norm_x, x);
		norm2_mpfvector(norm_res, res);

//		printf("%d norm_x = ", i);
//		mpf_out_str(stdout, 10, 2, norm_x); printf("\n");
//		printf("%d norm_res = ", i);
		printf("%5d ", i); mpf_div(tmp, norm_res, norm_res0);
		mpf_out_str(stdout, 10, 2, tmp); printf("\n");

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		mpf_set_ui(tmp, (unsigned long)dim);
		mpf_sqrt(tmp, tmp);
		mpf_mul(tmp, tmp, rtol);
		mpf_mul(tmp, tmp, norm_a);
		mpf_mul(tmp, tmp, norm_x);
		mpf_add(tmp, tmp, atol);
		if(mpf_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_mpfvector(resf, res);

		// Back-solve on short precision residual and short precision factors
//		SolveMPFLSC(zf, af, resf, af_row_ch, af_col_ch);
		SolveMPFLSP(zf, af, resf, af_row_ch);

		// Promote the correction from short precision to long precision
		subst_mpfvector(z, zf);

		// Update solution in long precision
		add_mpfvector(x, x, z);

		// for debug
//		printf("times: %d\n", i);
//		print_mpfvector(x);
	}

	// if fail, retry in mpf_t precision
/*	if(i >= maxtimes)
	{
		// mpf_t precision
		MPFLUdecompC(a, af_row_ch, af_col_ch);
		SolveMPFLSC(x, a, b, af_row_ch, af_col_ch);
	}
*/
	// Clear
	free_mpfmatrix(af);
	free_mpfvector(bf);
	free_mpfvector(xf);
	free_mpfvector(resf);
	free_mpfvector(res);
	free_mpfvector(zf);
	free_mpfvector(z);
	free(af_row_ch);
	free(af_col_ch);

	mpf_clear(tmp);
	mpf_clear(norm_a);
	mpf_clear(norm_x);
	mpf_clear(norm_res);
	mpf_clear(norm_res0);

	return i;
}
#endif // 0
#endif

double dget_meps_base_prec(unsigned long base, unsigned long prec)
{
	double eps;
	eps = 1.0; //mpf_set_ui(eps, 1UL);
	while(prec-- > 0)
		eps /= (double)base;

	return eps;
}

/* get residual in TD precision */
/* r := b - A * x */
void residual_tdmat_tdvec(TDVector r, TDVector b, TDMatrix mat, TDVector x)
{
#ifdef _OPENMP
	_bncomp_mul_tdmatrix_tdvec(r, mat, x);
	_bncomp_sub_tdvector(r, b, r);
#else // _OPENMP
	mul_tdmatrix_tdvec(r, mat, x);
	sub_tdvector(r, b, r);
#endif // _OPENMP
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int tdgesirsv_dd(TDVector x, TDMatrix a, TDVector b, double rtol, double atol, long int maxtimes)
{
	long int dim, i;
	long int *af_ch;
	DDMatrix af;
	DDVector bf, xf, resf, zf;
	TDVector res, z;
	double tmp[TDSIZE], norm_a[TDSIZE], norm_x[TDSIZE], norm_res[TDSIZE];

	// Initialize
	dim = x->dim;
	set0_td(tmp);
	set0_td(norm_a);
	set0_td(norm_x);
	set0_td(norm_res);

	af = init_ddmatrix(dim, dim);
	bf = init_ddvector(dim);
	xf = init_ddvector(dim);

	res = init_tdvector(dim);
	z = init_tdvector(dim);

	resf = init_ddvector(dim);
	zf = init_ddvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_tdmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_ddmatrix_tdmat(af, a);
	subst_ddvector_tdvec(bf, b);
	//printf("||a||_F= %10.3e\n", norm_a[0]);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
#ifdef _OPENMP
	DDLUdecompPM_omp(af, af_ch);
	//DDLUdecomp_strassenPM_omp(af, af_ch, 32);
	SolveDDLSPM(xf, af, bf, af_ch);
#else // _OPENMP
	DDLUdecompP(af, af_ch);
	// Apply back-solve in short precision with short precision factors
	SolveDDLSP(xf, af, bf, af_ch);
#endif // _OPENMP

	// Promote te solution from short precision to long precision
	subst_tdvector_ddvec(x, xf);
	norm2_tdvector(tmp, x);	printf("||x||_2= %10.3e\n", tmp[0]);
	norm2_tdvector(tmp, b);	printf("||b||_2= %10.3e\n", tmp[0]);
	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_tdmat_tdvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_tdvector(norm_x, x);
		norm2_tdvector(norm_res, res);

		// normalization of residual
		rtd_ui_div(tmp, 1UL, norm_res);
		cmul2_tdvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		rtd_set_ui(tmp, (unsigned long)dim);
		rtd_sqrt(tmp, tmp);
		rtd_mul_d(tmp, tmp, rtol);
		rtd_mul(tmp, tmp, norm_a);
		rtd_mul(tmp, tmp, norm_x);
		rtd_add_d(tmp, tmp, atol);
		if(rtd_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_ddvector_tdvec(resf, res);

		// Back-solve on short precision residual and short precision factors
#ifdef _OPENMP
		SolveDDLSPM(zf, af, resf, af_ch);
#else // _OPENMP
		SolveDDLSP(zf, af, resf, af_ch);
#endif // _OPENMP

		// Promote the correction from short precision to long precision
		subst_tdvector_ddvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_tdvector(x, x, norm_res, z);

		// for debug
		printf("%ld %15.8e\n", i, norm_res[0]);

		//print_tdvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		#ifdef _OPENMP
		TDLUdecomp_omp(a);
		SolveTDLS(x, a, b);
		#else // _OPENMP
		TDLUdecomp(a);
		SolveTDLS(x, a, b);
		#endif // _OPENMP
	}

	// Clear
	free_ddmatrix(af);
	free_ddvector(bf);
	free_ddvector(xf);
	free_ddvector(resf);
	free_ddvector(zf);
	free_tdvector(res);
	free_tdvector(z);
	free(af_ch);

	return i;
}


long int td_iterative_refinement_dd(TDVector x, TDMatrix a, TDVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	double rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = 159; //x->prec;
	short_prec = 106;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	//mpf_init2(rtol, short_prec);
	//mpf_init2(atol, short_prec);
	//mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	rtol = dget_meps_base_prec(2UL, long_prec);
	atol = 0.0;
	
	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : %5.1e\n", rtol);
	printf("atol            : %5.1e\n", atol);

	times = tdgesirsv_dd(x, a, b, rtol, atol, maxtimes);

	//mpf_clear(rtol);
	//mpf_clear(atol);

	return times;
}

/* get residual in QD precision */
/* r := b - A * x */
void residual_qdmat_qdvec(QDVector r, QDVector b, QDMatrix mat, QDVector x)
{
#ifdef _OPENMP
	_bncomp_mul_qdmatrix_qdvec(r, mat, x);
	_bncomp_sub_qdvector(r, b, r);
#else // _OPENMP
	mul_qdmatrix_qdvec(r, mat, x);
	sub_qdvector(r, b, r);
#endif // _OPENMP
}

// interative refinement with single & mpf_t mixed precision arithmetic
// solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
// unsigned long long_prec, short_prec; // long_prec > short_prec
int qdgesirsv_dd(QDVector x, QDMatrix a, QDVector b, double rtol, double atol, long int maxtimes)
{
	long int dim, i;
	long int *af_ch;
	DDMatrix af;
	DDVector bf, xf, resf, zf;
	QDVector res, z;
	double tmp[QDSIZE], norm_a[QDSIZE], norm_x[QDSIZE], norm_res[QDSIZE];

	// Initialize
	dim = x->dim;
	set0_qd(tmp);
	set0_qd(norm_a);
	set0_qd(norm_x);
	set0_qd(norm_res);

	af = init_ddmatrix(dim, dim);
	bf = init_ddvector(dim);
	xf = init_ddvector(dim);

	res = init_qdvector(dim);
	z = init_qdvector(dim);

	resf = init_ddvector(dim);
	zf = init_ddvector(dim);
	af_ch = (long int *)calloc(sizeof(long int), dim);

	normf_qdmatrix(norm_a, a);

	// Make short precision copy of A and b
	subst_ddmatrix_qdmat(af, a);
	subst_ddvector_qdvec(bf, b);
	//printf("||a||_F= %10.3e\n", norm_a[0]);

	// Compute LU factorization in short precision
	// LU decomposition with partial pivoting
#ifdef _OPENMP
	DDLUdecompPM_omp(af, af_ch);
	SolveDDLSPM(xf, af, bf, af_ch);
#else // _OPENMP
	DDLUdecompP(af, af_ch);
	// Apply back-solve in short precision with short precision factors
	SolveDDLSP(xf, af, bf, af_ch);
#endif // _OPENMP

	// Promote te solution from short precision to long precision
	subst_qdvector_ddvec(x, xf);
	//norm2_qdvector(tmp, x);	printf("||x||_2= %10.3e\n", tmp[0]);
	//norm2_qdvector(tmp, b);	printf("||b||_2= %10.3e\n", tmp[0]);
	// repeat iterative refinement process
	for(i = 0; i < maxtimes; i++)
	{
		// Compute residual in long precision
		residual_qdmat_qdvec(res, b, a, x);

		// until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		norm2_qdvector(norm_x, x);
		norm2_qdvector(norm_res, res);

		// normalization of residual
		rqd_ui_div(tmp, 1UL, norm_res);
		cmul2_qdvector(res, tmp);

		//if(norm_res < sqrt((mpf_t)dim) * rtol * norm_a * norm_x + atol)
		rqd_set_ui(tmp, (unsigned long)dim);
		rqd_sqrt(tmp, tmp);
		rqd_mul_d(tmp, tmp, rtol);
		rqd_mul(tmp, tmp, norm_a);
		rqd_mul(tmp, tmp, norm_x);
		rqd_add_d(tmp, tmp, atol);
		if(rqd_cmp(norm_res, tmp) < 0)
			break;
	
		// Demote the residual from long precision to short precison
		subst_ddvector_qdvec(resf, res);

		// Back-solve on short precision residual and short precision factors
#ifdef _OPENMP
		SolveDDLSPM(zf, af, resf, af_ch);
#else // _OPENMP
		SolveDDLSP(zf, af, resf, af_ch);
#endif // _OPENMP

		// Promote the correction from short precision to long precision
		subst_qdvector_ddvec(z, zf);

		// Update solution in long precision
		//add_mpfvector(x, x, z);
		add_cmul_qdvector(x, x, norm_res, z);

		// for debug
		printf("times: %ld\n", i);
		rqd_out_str(norm_res);printf("\n");
		//print_tdvector(x);
	}

	// if fail, retry in mpf_t precision
	if(i >= maxtimes)
	{
		// mpf_t precision
		#ifdef _OPENMP
		QDLUdecomp_omp(a);
		SolveQDLS(x, a, b);
		#else // _OPENMP
		QDLUdecomp(a);
		SolveQDLS(x, a, b);
		#endif // _OPENMP
	}

	// Clear
	free_ddmatrix(af);
	free_ddvector(bf);
	free_ddvector(xf);
	free_ddvector(resf);
	free_ddvector(zf);
	free_qdvector(res);
	free_qdvector(z);
	free(af_ch);

	return i;
}


long int qd_iterative_refinement_dd(QDVector x, QDMatrix a, QDVector b)
{
	long int times, dim, maxtimes;
	unsigned long short_prec, long_prec;
	double rtol, atol;

	// Initialize
	dim = x->dim;
	long_prec = 212; //x->prec;
	short_prec = 106;
//	maxtimes = dim * 10;
	maxtimes = ((dim / 2) > 10) ? dim / 2 : 10;

	//mpf_init2(rtol, short_prec);
	//mpf_init2(atol, short_prec);
	//mpf_get_meps_base_prec(rtol, 2UL, long_prec);
	rtol = dget_meps_base_prec(2UL, long_prec);
	atol = 0.0;
	
	printf("dim             : %ld\n", dim);
	printf("maxtimes        : %ld\n", maxtimes);
	printf("long_prec (bits): %ld\n", long_prec);
	printf("short_prec(bits): %ld\n", short_prec);
	printf("long_prec (dec) : %ld\n", (long)(long_prec * log10(2.0)));
	printf("short_prec(dec) : %ld\n", (long)(short_prec * log10(2.0)));
	printf("rtol            : %5.1e\n", rtol);
	printf("atol            : %5.1e\n", atol);

	times = qdgesirsv_dd(x, a, b, rtol, atol, maxtimes);

	//mpf_clear(rtol);
	//mpf_clear(atol);

	return times;
}

//#define DIM 16384
//#define DIM 8192
//#define DIM 4096
//#define DIM 2048
//#define DIM 1024
//#define DIM 512
//#define DIM 256
//#define DIM 128
//#define DIM 64
//#define DIM 32
//#define DIM 16
//#define DIM 8

//#define MPFDPREC 50
//#define MPFDPREC 100
//#define MPFDPREC 500
//#define MPFDPREC 1000
//#define MPFDPREC 2000

//#define MPFBITS 1024
//#define MPFBITS 512
//#define MPFBITS 256
//#define MPFBITS 128
#define MPFBITS 53
#define DBL_M_EPSILON (1.0e-16)
#define FLT_M_EPSILON (1.0e-07)

void testmatrix(MPFMatrix mat)
{
	unsigned long prec;
	long int dim, i;
	MPFMatrix trans_mat;
	MPFVector eig;
	char fname[128];

	prec = mat->prec;
	dim = (mat->row_dim < mat->col_dim) ? mat->row_dim : mat->col_dim;

//	hilbert_mpfmatrix(mat, dim); printf("Hilbert Matrix(DIM = %d, PREC(BITS) = %d)\n", dim, prec);
//	frank_mpfmatrix(mat, dim); printf("Frank Matrix(DIM = %d, PREC(BITS) = %d)\n", dim, prec);
//	lotkin_mpfmatrix(mat, dim); printf("Lotkin Matrix(DIM = %d, PREC(BITS) = %d)\n", dim, prec);

	eig = init2_mpfvector(dim, prec);
	trans_mat = init2_mpfmatrix(mat->row_dim, mat->col_dim, prec);

	for(i = 0; i < eig->dim; i++)
		mpf_set_ui(get_mpfvector_i(eig, i), (eig->dim - i));
		

//	set_test_mpfvector(vec, 1);
//	vandermonde_mpfmatrix(mat, vec);printf("Vandermond Matrix(DIM = %d, PREC(BITS) = %d)\n", dim, prec);
//	free_mpfvector(vec);

//	sprintf(fname, "testmat%d-2000.dat", dim); fread_mpfmatrix_fname(fname, mat);
//	printf("Reading %s ...\n", fname);
	generate_mpfmatrix(mat, trans_mat, eig, 1, 10);

	free_mpfvector(eig);
	free_mpfmatrix(trans_mat);

}

/* proposed by matsuo */
/* U, V : Orthogonal Matrices due to random matrices */
/* D : Diagonal Matrix given by users */
void testmatrix2(MPFMatrix mat)
{
	unsigned long prec;
	long int dim;
	char fname[128];

	prec = mat->prec;
	dim = (mat->row_dim < mat->col_dim) ? mat->row_dim : mat->col_dim;

	sprintf(fname, "testmat%ld-2000.dat", dim);
	printf("Reading %s ...\n", fname);

	//fread_mpfmatrix_fname(fname, mat);

}

// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq_td_test(TDMatrix A, TDVector true_x, TDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits
	double tmp[TDSIZE];

	// A
	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read matrix A!\n", fname_A);
	}
	else{
		printf("%s was successfully read...\n", fname_A);
		do {
			ret_scan = fscanf(fp, "%d, %d, %s", &i, &j, str_num);
			//A[i * dim + j] = (T)str_num;
			rtd_set_str(tmp, str_num);
			set_tdmatrix_ij(A, i, j, tmp);
			//printf("%s -- \n", str_num); rdd_out_str(tmp); printf("\n");
		} while(!feof(fp));

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_true_x);
	}
	else{
		printf("%s was successfully read...\n", fname_true_x);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//true_x[i] = (T)str_num;
			rtd_set_str(tmp, str_num);
			set_tdvector_i(true_x, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_b);
	}
	else{
		printf("%s was successfully read...\n", fname_b);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//b[i] = (T)str_num;
			rtd_set_str(tmp, str_num);
			printf("%d, %s -> %15.8e\n", i, str_num, tmp[0]);
			set_tdvector_i(b, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}
}
// Euclid norm
void norm2_tdvector2(double ret[TDSIZE], TDVector vec)
{
	long int i, index, dim;
	double tmp[TDSIZE];

	dim = vec->dim;
	set0_td(tmp);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[TDSIZE], ret4[TDSIZE], tmp4[TDSIZE];

	_bncavx2_set0_td(ret4);
	_bncavx2_set0_td(tmp4);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(vec->element[0][index]));
		vec4[1] = _mm256_load_pd(&(vec->element[1][index]));
		vec4[2] = _mm256_load_pd(&(vec->element[2][index]));

//		rtd_mul(tmp, get_tdvector_i(vec, i), get_tdvector_i(vec, i));
//		rtd_add(ret, ret, tmp);
		//_bncavx2_rtd_mul(tmp4, vec4, vec4);
		_bncavx2_rtd_mulq(tmp4, vec4, vec4);
		_bncavx2_rtd_addq(ret4, ret4, tmp4);
	}
	//_bncavx2_rtd_norm256d(ret, ret4);
	_bncavx2_rtd_sum256d(tmp, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[TDSIZE], ret8[TDSIZE], tmp8[TDSIZE];

	_bncavx512_set0_td(ret8);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(vec->element[0][index]));
		vec8[1] = _mm512_load_pd(&(vec->element[1][index]));
		vec8[2] = _mm512_load_pd(&(vec->element[2][index]));

//		rtd_mul(tmp, get_tdvector_i(a, i), get_tdvector_i(b, i));
//		rtd_add(ret, ret, tmp);
		_bncavx512_rtd_mul(tmp8, vec8, vec8);
		_bncavx512_rtd_add(ret8, ret8, tmp8);
	}
	//_bncavx512_rtd_norm512d(ret, ret8);
	_bncavx512_rtd_sum512d(tmp, ret8);
#else // others
	//c_dd_copy_d((double)0.0, tmp);
	//c_dd_copy_d((double)0.0, ret);
	rtd_set0(tmp);
	rtd_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_dd_sqr(GET_TDVECTOR_I(vec, i), tmp);
		//c_dd_add(tmp, ret, ret);
		rtd_mul(ret, get_tdvector_i(vec, i), get_tdvector_i(vec, i));
		rtd_add(tmp, ret, ret);
	}

	//c_td_sqrt(ret, tmp);
	//c_td_copy(tmp, ret);
	//rtd_set(tmp, ret);
#endif // __AVX2__

	rtd_sqrt(ret, tmp);
}

int main(int argc, char *argv[])
{
	unsigned long dprec, prec;
	long int dim;
	double stime, etime;
	DMatrix dmat;
	double norm1_dmat, dcondest, dcoef_f;
//	long int row_ch[DIM], col_ch[DIM], i;
	long int *row_ch, *col_ch, i;
	DDMatrix ddmat;
	DDVector ddx, ddb;
	TDMatrix tdmat;
	TDVector tdx, td_true_vec, tdb;
	double td_relerr[TDSIZE], td_max_relerr[TDSIZE], td_min_relerr[TDSIZE], td_norm_relerr[TDSIZE];
	QDMatrix qdmat;
	QDVector qdx, qd_true_vec, qdb;
	double qd_relerr[QDSIZE], qd_max_relerr[QDSIZE], qd_min_relerr[QDSIZE], qd_norm_relerr[QDSIZE];
	char fname_A[256], fname_true_x[256], fname_vec_b[256];
	int num_threads;
#ifdef USE_GMP
	MPFMatrix mpfmat, mpfmat_s;
	MPFVector mpfx, mpfb, mpf_true_vec, mpfx_s, mpfb_s;
	mpf_t norm1_mpfmat, mpfcondest, mpfcoef_f, mpfrelerr, mpf_min_relerr, mpf_max_relerr, mpf_norm_relerr;
#endif

	if(argc <= 1)
	{
		#ifdef _OPENMP
		printf("[usage] %s dim prec #threads\n", argv[0]);
		#else //_OPENMP
		printf("[usage] %s dim prec\n", argv[0]);
		#endif // _OPENMP
		return 0;
	}

	dim = atoi(argv[1]);

//	dprec = 50;
	prec = 256;

	// dprec
	//if(argc >= 3)
	//	dprec = atoi(argv[2]);

	// prec
	if(argc >= 3)
		prec = atoi(argv[2]);

	row_ch = (long int *)calloc(dim, sizeof(long int));
	col_ch = (long int *)calloc(dim, sizeof(long int));

	// prec
	#ifdef _OPENMP
	num_threads = 2;
	if(argc >= 4)
		num_threads = atoi(argv[3]);

	set_bncomp_num_threads(num_threads);
	#endif //_OPENMP

goto mpf;

	dmat = init_dmatrix(dim, dim);

	hilbert_dmatrix(dmat, dim);
	norm1_dmat = norm1_dmatrix(dmat);
	printf("||A||_1 = %25.17e\n", norm1_dmat);

	DLUdecompC(dmat, row_ch, col_ch);

	dcondest = condest_dmatrix_C(norm1_dmat, dmat, row_ch, col_ch);
	printf("Estimated Cond1 = %25.17e\n", dcondest);

	dcoef_f = coef_f_itr(dcondest, DBL_M_EPSILON, FLT_M_EPSILON, dim);
	printf("Coef_f_itr      = %25.17e\n", dcoef_f);

	free_dmatrix(dmat);

mpf:
#ifdef USE_GMP

//	set_bnc_default_prec_decimal(1500);
//	set_bnc_default_prec(MPFBITS);

	//printf("DPREC = %ld\n", MPFDPREC);
	//printf("DPREC = %ld\n", dprec);
	printf("PREC = %ld\n", prec);
//	set_bnc_default_prec_decimal(MPFDPREC);
//	set_bnc_default_prec((unsigned long)ceil((double)dprec / log10(2.0)));
	set_bnc_default_prec(prec);

	mpf_init(mpfcondest);
	mpf_init(norm1_mpfmat);
	mpf_init(mpfcoef_f);
	mpf_init(mpfrelerr);
	mpf_init(mpf_max_relerr);
	mpf_init(mpf_min_relerr);
	mpf_init(mpf_norm_relerr);

	mpfmat = init_mpfmatrix(dim, dim);
	mpfx = init_mpfvector(dim);
	mpf_true_vec = init_mpfvector(dim);
	mpfb = init_mpfvector(dim);

//iteratige_ref_dd:
	// Initialize QD library
	fpu_fix_start(NULL);

	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	sprintf(fname_A, "../python/mat_a_%d_%d_b2048.txt", dim, dim);
	sprintf(fname_true_x, "../python/vec_true_x_%d_b2048.txt", dim);
	sprintf(fname_vec_b, "../python/vec_b_%d_b2048.txt", dim);	

	read_test_linear_eq(mpfmat, mpf_true_vec, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);

	printf("DD + MPFR(dim = %ld, prec = %ld) ...\n", dim, prec);
	stime = get_real_secv(); //get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	mpf_iterative_refinement_dd(mpfx, mpfmat, mpfb);
	etime = get_real_secv(); //get_secv();
//	print_mpfvector(mpfx);

	printf("Iter.Ref(dim=%ld, L=%ld, S=%ld)\n", dim, get_bnc_default_prec(),  get_bnc_default_prec() / 2);
	relerr_mpfvector(mpfrelerr, mpfx, mpf_true_vec, 2);
	relerr_element_mpfvector(mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr, mpfx, mpf_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): "); mpf_log10(mpfrelerr, mpfrelerr); printf("%5.2f\n", mpf_get_d(mpfrelerr));
	printf("lg(Relerr(max))  : "); mpf_log10(mpf_max_relerr, mpf_max_relerr); printf("%5.2f\n", mpf_get_d(mpf_max_relerr));
	printf("lg(Relerr(min))  : "); mpf_log10(mpf_min_relerr, mpf_min_relerr); printf("%5.2f\n", mpf_get_d(mpf_min_relerr));

// TD
	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	sprintf(fname_A, "../python/mat_a_%d_%d_b2048.txt", dim, dim);
	sprintf(fname_true_x, "../python/vec_true_x_%d_b2048.txt", dim);
	sprintf(fname_vec_b, "../python/vec_b_%d_b2048.txt", dim);	

	read_test_linear_eq(mpfmat, mpf_true_vec, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);

	printf("TD + MPFR(dim = %ld, prec = %ld) ...\n", dim, prec);
	stime = get_real_secv(); //get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	mpf_iterative_refinement_td(mpfx, mpfmat, mpfb);
	etime = get_real_secv(); //get_secv();
//	print_mpfvector(mpfx);

	printf("Iter.Ref(dim=%ld, L=%ld, S=%ld)\n", dim, get_bnc_default_prec(),  get_bnc_default_prec() / 2);
	relerr_mpfvector(mpfrelerr, mpfx, mpf_true_vec, 2);
	relerr_element_mpfvector(mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr, mpfx, mpf_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): "); mpf_log10(mpfrelerr, mpfrelerr); printf("%5.2f\n", mpf_get_d(mpfrelerr));
	printf("lg(Relerr(max))  : "); mpf_log10(mpf_max_relerr, mpf_max_relerr); printf("%5.2f\n", mpf_get_d(mpf_max_relerr));
	printf("lg(Relerr(min))  : "); mpf_log10(mpf_min_relerr, mpf_min_relerr); printf("%5.2f\n", mpf_get_d(mpf_min_relerr));

// QD
	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	sprintf(fname_A, "../python/mat_a_%d_%d_b2048.txt", dim, dim);
	sprintf(fname_true_x, "../python/vec_true_x_%d_b2048.txt", dim);
	sprintf(fname_vec_b, "../python/vec_b_%d_b2048.txt", dim);	

	read_test_linear_eq(mpfmat, mpf_true_vec, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);

	printf("QD + MPFR(dim = %ld, prec = %ld) ...\n", dim, prec);
	stime = get_real_secv(); //get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	mpf_iterative_refinement_qd(mpfx, mpfmat, mpfb);
	etime = get_real_secv(); //get_secv();
//	print_mpfvector(mpfx);

	printf("Iter.Ref(dim=%ld, L=%ld, S=%ld)\n", dim, get_bnc_default_prec(),  get_bnc_default_prec() / 2);
	relerr_mpfvector(mpfrelerr, mpfx, mpf_true_vec, 2);
	relerr_element_mpfvector(mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr, mpfx, mpf_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): "); mpf_log10(mpfrelerr, mpfrelerr); printf("%5.2f\n", mpf_get_d(mpfrelerr));
	printf("lg(Relerr(max))  : "); mpf_log10(mpf_max_relerr, mpf_max_relerr); printf("%5.2f\n", mpf_get_d(mpf_max_relerr));
	printf("lg(Relerr(min))  : "); mpf_log10(mpf_min_relerr, mpf_min_relerr); printf("%5.2f\n", mpf_get_d(mpf_min_relerr));

	//free_mpfmatrix(mpfmat_s);
	//free_mpfvector(mpfb_s);
	//free_mpfvector(mpfx_s);


// DD-TD
	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	tdmat = init_tdmatrix(dim, dim);
	tdx = init_tdvector(dim);
	td_true_vec = init_tdvector(dim);
	tdb = init_tdvector(dim);

	//sprintf(fname_A, "../python/mat_a_%d_%d_b256.txt", dim, dim);
	//sprintf(fname_true_x, "../python/vec_true_x_%d_b256.txt", dim);
	//sprintf(fname_vec_b, "../python/vec_b_%d_b256.txt", dim);	
	sprintf(fname_A, "../python/mat_a_%d_%d_b2048.txt", dim, dim);
	sprintf(fname_true_x, "../python/vec_true_x_%d_b2048.txt", dim);
	sprintf(fname_vec_b, "../python/vec_b_%d_b2048.txt", dim);	

	//read_test_linear_eq_td_test(tdmat, td_true_vec, tdb, (int)dim, fname_A, fname_true_x, fname_vec_b);
	read_test_linear_eq(mpfmat, mpf_true_vec, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);
	subst_tdmatrix_mpfmat(tdmat, mpfmat);
	subst_tdvector_mpfvec(td_true_vec, mpf_true_vec);
	subst_tdvector_mpfvec(tdb, mpfb);

	printf("DD + TD(dim = %ld) ...\n", dim);
	normf_tdmatrix(td_max_relerr, tdmat);
	norm2_tdvector2(td_min_relerr, tdb);
	norm2_tdvector2(td_relerr, td_true_vec);
	printf("||A||_F, ||true_x||_2, ||b||_2 = %15.8e, %15.8e, %15.8e\n", td_max_relerr[0], td_relerr[0], td_min_relerr[0]);
	stime = get_real_secv(); //get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	td_iterative_refinement_dd(tdx, tdmat, tdb);
	etime = get_real_secv(); //get_secv();
//	print_mpfvector(mpfx);

	printf("Iter.Ref(dim=%ld, L=%ld, S=%ld)\n", dim, 159, 106);
	relerr_tdvector(td_relerr, tdx, td_true_vec, 2);
	relerr_element_tdvector(td_max_relerr, td_min_relerr, td_norm_relerr, tdx, td_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): %5.2f\n", log10(td_relerr[0]));
	printf("lg(Relerr(max))  : %5.2f\n", log10(td_max_relerr[0]));
	printf("lg(Relerr(min))  : %5.2f\n", log10(td_min_relerr[0]));

	free_tdmatrix(tdmat);
	free_tdvector(tdx);
	free_tdvector(tdb);

// DD-QD
	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	qdmat = init_qdmatrix(dim, dim);
	qdx = init_qdvector(dim);
	qd_true_vec = init_qdvector(dim);
	qdb = init_qdvector(dim);

	sprintf(fname_A, "../python/mat_a_%d_%d_b256.txt", dim, dim);
	sprintf(fname_true_x, "../python/vec_true_x_%d_b256.txt", dim);
	sprintf(fname_vec_b, "../python/vec_b_%d_b256.txt", dim);	

	read_test_linear_eq_qd(qdmat, qd_true_vec, qdb, (int)dim, fname_A, fname_true_x, fname_vec_b);
	normf_qdmatrix(qd_max_relerr, qdmat);
	norm2_qdvector(qd_min_relerr, qdb);
	printf("||A||_F, ||b||_2 = %15.8e, %15.8e\n", qd_max_relerr[0], qd_min_relerr[0]);

	printf("DD + QD(dim = %ldd) ...\n", dim);
	stime = get_real_secv(); //get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	qd_iterative_refinement_dd(qdx, qdmat, qdb);
	etime = get_real_secv(); //get_secv();
//	print_mpfvector(mpfx);

	printf("Iter.Ref(dim=%ld, L=%ld, S=%ld)\n", dim, 212, 106);
	relerr_qdvector(qd_relerr, qdx, qd_true_vec, 2);
	relerr_element_qdvector(qd_max_relerr, qd_min_relerr, qd_norm_relerr, qdx, qd_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): %5.2f\n", log10(qd_relerr[0]));
	printf("lg(Relerr(max))  : %5.2f\n", log10(qd_max_relerr[0]));
	printf("lg(Relerr(min))  : %5.2f\n", log10(qd_min_relerr[0]));

	free_qdmatrix(qdmat);
	free_qdvector(qdx);
	free_qdvector(qdb);

	mpf_clear(mpfcondest);
	mpf_clear(norm1_mpfmat);
	mpf_clear(mpfcoef_f);
	mpf_clear(mpfrelerr);
	mpf_clear(mpf_max_relerr);
	mpf_clear(mpf_min_relerr);
	mpf_clear(mpf_norm_relerr);
	free_mpfmatrix(mpfmat);
	free_mpfvector(mpfx);
	free_mpfvector(mpfb);
	free_mpfvector(mpf_true_vec);
#endif

	free(row_ch);
	free(col_ch);

}
