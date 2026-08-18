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
	mul_ddmatrix_ddvec(r, mat, x);
	sub_ddvector(r, b, r);
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
	DDLUdecompP(af, af_ch);

	// Apply back-solve in short precision with short precision factors
	SolveDDLSP(xf, af, bf, af_ch);

	// Promote te solution from short precision to long precision
	subst_mpfvector_ddvec(x, xf);

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
		subst_ddvector_mpfvec(resf, res);

		// Back-solve on short precision residual and short precision factors
		SolveDDLSP(zf, af, resf, af_ch);

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
		MPFLUdecomp(a);
		SolveMPFLS(x, a, b);
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
	short_prec = 32;
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

// random

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
#ifdef USE_GMP
	MPFMatrix mpfmat, mpfmat_s;
	MPFVector mpfx, mpfb, mpf_true_vec, mpfx_s, mpfb_s;
	mpf_t norm1_mpfmat, mpfcondest, mpfcoef_f, mpfrelerr, mpf_min_relerr, mpf_max_relerr, mpf_norm_relerr;
#endif

	if(argc <= 1)
	{
		printf("[usage] %s dim [dprec]\n", argv[0]);
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
	set_test_mpfvector(mpfx, 1);

	get_correct_rounded_mpfmatrix(mpfmat, testmatrix);

//	print_mpfmatrix(mpfmat);

	norm1_mpfmatrix(norm1_mpfmat, mpfmat);
//	printf("||A||_1 = %25.17e\n", mpf_get_d(norm1_mpfmat));
	printf("||A||_1 = "); mpf_out_str(stdout, 10, 17, norm1_mpfmat); printf("\n");

	MPFLUdecompC(mpfmat, row_ch, col_ch);

	condest_mpfmatrix_C(mpfcondest, norm1_mpfmat, mpfmat, row_ch, col_ch);

//	printf("Estimated Cond1 = %25.17e\n", mpf_get_d(mpfcondest));
	printf("Estimated Cond1 = "); mpf_out_str(stdout, 10, 17, mpfcondest); printf("\n");
	mpfcoef_f_itr(mpfcoef_f, mpfcondest, get_bnc_default_prec(), 24, dim);
//	mpfcoef_f_itr(mpfcoef_f, mpfcondest, get_bnc_default_prec(), (unsigned long)(get_bnc_default_prec()), dim);
//	printf("Coef_f_itr      = %25.17e\n", mpf_get_d(mpfcoef_f));
	printf("Coef_f_itr      = "); mpf_out_str(stdout, 10, 17, mpfcoef_f); printf("\n");

	mpf_true_vec = init_mpfvector(dim);
	mpfb = init_mpfvector(dim);
	if((mpf_cmp_ui(mpfcoef_f, 0UL) <= 0) || (mpf_cmp_ui(mpfcoef_f, 1UL) > 0))
		goto short_lu;

//iteratige_ref_d:
	/* solve A * x = b */
	set_test_mpfvector(mpfx, 1);
	for(i = 0; i < dim; i++)
		set_mpfvector_i_ui(mpf_true_vec, i, (unsigned long)(i + 1));

	get_correct_rounded_mpfmatrix_mpfvec(mpfmat, mpfb, mpf_true_vec, testmatrix);

	stime = get_secv();
	mpf_iterative_refinement_d(mpfx, mpfmat, mpfb);
	etime = get_secv();
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

//iteratige_ref_dd:
	// Initialize QD library
	fpu_fix_start(NULL);

	/* solve A * x = b */
	set_test_mpfvector(mpfx, 1);
	for(i = 0; i < dim; i++)
		set_mpfvector_i_ui(mpf_true_vec, i, (unsigned long)(i + 1));

	get_correct_rounded_mpfmatrix_mpfvec(mpfmat, mpfb, mpf_true_vec, testmatrix);

	stime = get_secv();
	//mpf_iterative_refinement(mpfx, mpfmat, mpfb);
	mpf_iterative_refinement_dd(mpfx, mpfmat, mpfb);
	etime = get_secv();
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

short_lu:
/* LU decomp (L) */
	/* solve A * x = b */
	for(i = 0; i < dim; i++)
		set_mpfvector_i_ui(mpf_true_vec, i, (unsigned long)(i + 1));

	get_correct_rounded_mpfmatrix_mpfvec(mpfmat, mpfb, mpf_true_vec, testmatrix);

	stime = get_secv();
//	MPFLUdecompC(mpfmat, row_ch, col_ch);
//	SolveMPFLSC(mpfx, mpfmat, mpfb, row_ch, col_ch);
	MPFLUdecompP(mpfmat, row_ch);
	SolveMPFLSP(mpfx, mpfmat, mpfb, row_ch);
	etime = get_secv();
//	print_mpfvector(mpfx);

//	printf("LUdecompC(dim=%d, L=%d)\n", dim, get_bnc_default_prec());
	printf("LUdecompP(dim=%ld, L=%ld)\n", dim, get_bnc_default_prec());
	relerr_mpfvector(mpfrelerr, mpfx, mpf_true_vec, 2);
	relerr_element_mpfvector(mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr, mpfx, mpf_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): "); mpf_log10(mpfrelerr, mpfrelerr); printf("%5.2f\n", mpf_get_d(mpfrelerr));
	printf("lg(Relerr(max))  : "); mpf_log10(mpf_max_relerr, mpf_max_relerr); printf("%5.2f\n", mpf_get_d(mpf_max_relerr));
	printf("lg(Relerr(min))  : "); mpf_log10(mpf_min_relerr, mpf_min_relerr); printf("%5.2f\n", mpf_get_d(mpf_min_relerr));

/* LU decomp (S) */
	/* solve A * x = b */
	for(i = 0; i < dim; i++)
		set_mpfvector_i_ui(mpf_true_vec, i, (unsigned long)(i + 1));

	get_correct_rounded_mpfmatrix_mpfvec(mpfmat, mpfb, mpf_true_vec, testmatrix);

	/* halves prec */
	mpfmat_s = init2_mpfmatrix(dim, dim, get_bnc_default_prec() / 2);
	mpfb_s   = init2_mpfvector(dim, get_bnc_default_prec() / 2);
	mpfx_s   = init2_mpfvector(dim, get_bnc_default_prec() / 2);

	subst_mpfmatrix(mpfmat_s, mpfmat);
	subst_mpfvector(mpfb_s, mpfb);

	stime = get_secv();
//	MPFLUdecompC(mpfmat_s, row_ch, col_ch);
//	SolveMPFLSC(mpfx_s, mpfmat_s, mpfb_s, row_ch, col_ch);
	MPFLUdecompP(mpfmat_s, row_ch);
	SolveMPFLSP(mpfx_s, mpfmat_s, mpfb_s, row_ch);
	etime = get_secv();
//	print_mpfvector(mpfx_s);

//	printf("LUdecompC(dim=%d, S=%d)\n", dim, get_bnc_default_prec() / 2);
	printf("LUdecompP(dim=%ld, S=%ld)\n", dim, get_bnc_default_prec() / 2);
	relerr_mpfvector(mpfrelerr, mpfx_s, mpf_true_vec, 2);
	relerr_element_mpfvector(mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr, mpfx_s, mpf_true_vec, 0);
	printf("time: %f sec\n", etime - stime);
//	printf("Relerr(norm2): "); mpf_out_str(stdout, 10, 5, mpfrelerr); printf("\n");
//	printf("Relerr(max)  : "); mpf_out_str(stdout, 10, 5, mpf_max_relerr); printf("\n");
//	printf("Relerr(min)  : "); mpf_out_str(stdout, 10, 5, mpf_min_relerr); printf("\n");
	printf("lg(Relerr(norm2)): "); mpf_log10(mpfrelerr, mpfrelerr); printf("%5.2f\n", mpf_get_d(mpfrelerr));
	printf("lg(Relerr(max))  : "); mpf_log10(mpf_max_relerr, mpf_max_relerr); printf("%5.2f\n", mpf_get_d(mpf_max_relerr));
	printf("lg(Relerr(min))  : "); mpf_log10(mpf_min_relerr, mpf_min_relerr); printf("%5.2f\n", mpf_get_d(mpf_min_relerr));

	free_mpfmatrix(mpfmat_s);
	free_mpfvector(mpfb_s);
	free_mpfvector(mpfx_s);

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
