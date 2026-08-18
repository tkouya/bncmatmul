/********************************************************************************/
/* diff.c:                                                                      */
/* Copyright (C) 2005-2011 Tomonori Kouya                                       */
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

#define FNM_DIM 128
#ifdef DEBUG
	#define FNM_DIM1 129 // FND_DIM + 1
#endif

double drelerr(double approx, double true_value)
{
	double relerr;

	relerr = fabs(approx - true_value);

	if(true_value =! 0.0)
		relerr /= fabs(true_value);

	return relerr;
}

/* Central Difference (3 points, O(h^2)) */
double dcentral_diff(double x, double (*func)(double), double stepsize)
{
	if(stepsize == 0.0)
	{
		fprintf(stderr, "Divided by 0!(dcentral_diff)\n");
		return 0.0;
	}

	return (0.5 * func(x + stepsize) - 0.5 * func(x - stepsize)) / stepsize;
}

/* Central Difference (3 points, O(h^2)) for extrapolation */
double dcentral_diff_err(double x, double (*func)(double), double stepsize, double *abserr)
{
	double func_p_h, func_m_h, eps_M = 1.11022302462515654e-16; // 2^(-53)

	if(stepsize == 0.0)
	{
		fprintf(stderr, "Divided by 0!(dcentral_diff_err)\n");
		return 0.0;
	}

	/* Erb(T_i1) = eps_M * max(|f(x+h)|, |f(x-h)|) / h */
	func_p_h = func(x + stepsize);
	func_m_h = func(x - stepsize);
	*abserr = dmax(fabs(func_p_h), fabs(func_m_h)) * eps_M / stepsize;

	return 0.5 * (func_p_h - func_m_h) / stepsize;
}


/* Central Difference (5 points, O(h^4)) */
double dcentral_diff15(double x, double (*func)(double), double stepsize)
{
	double ret;

	if(stepsize == 0.0)
	{
		fprintf(stderr, "Divided by 0!(dcentral_diff15)\n");
		return 0.0;
	}

	ret =  -(1.0/12.0) * func(x + 2 * stepsize);
	ret +=  (1.0/12.0) * func(x - 2 * stepsize);
	ret +=  (2.0/ 3.0) * func(x +     stepsize);
	ret += -(2.0/ 3.0) * func(x -     stepsize);
	ret /= stepsize;

	return ret;
}

/* Central Difference (7 points, O(h^6)) */
double dcentral_diff17(double x, double (*func)(double), double stepsize)
{
	double ret;

	if(stepsize == 0.0)
	{
		fprintf(stderr, "Divided by 0!(dcentral_diff17)\n");
		return 0.0;
	}

	ret =   (1.0/60.0) * func(x + 3 * stepsize);
	ret += -(1.0/60.0) * func(x - 3 * stepsize);
	ret += -(3.0/20.0) * func(x + 2 * stepsize);
	ret +=  (3.0/20.0) * func(x - 2 * stepsize);
	ret +=  (3.0/ 4.0) * func(x +     stepsize);
	ret += -(3.0/ 4.0) * func(x -     stepsize);
	ret /= stepsize;

	return ret;
}

/* append "max_stage" argument */
double dfnmdiff(double x, double (*func)(double), double init_stepsize, double rel_tol, double abs_tol, long int max_stage, long int *num_stage)
{
	long i, j;
	double h, power4, abs_est[FNM_DIM];
	double ex_table[FNM_DIM], new_ex, old_ex, correction;

	if(max_stage > FNM_DIM)
	{
		fprintf(stderr, "[@dfnmdiff] max_stage %ld is larger than FNM_DIM %d!\n", max_stage, FNM_DIM);
		return 0.0;
	}

	// h = 0.5; // h = 1/2
	h = init_stepsize; // given by user
	for(i = 0; i < max_stage; i++)
	{
		power4 = 1.0;

		old_ex = dcentral_diff_err(x, func, h, &abs_est[i]);
#ifdef DEBUG
		printf("%2ld %10.3e ", i, old_ex);
#endif
		for(j = 0; j < i; j++)
		{
			power4 *= 4.0;

			// Extrapolation
			correction = (old_ex - ex_table[j]) / (power4 - 1);
			new_ex = old_ex + correction;

			// Check convergence by Y.Fukui
			if((fabs(correction) <= 1.5 / (power4 - 1) * 2 * abs_est[i]) || (fabs(correction) <= rel_tol * fabs(old_ex) + abs_tol))
			{
				*num_stage = i;
				return new_ex;
			}
		
			// To next step
			ex_table[j] = old_ex;
			old_ex = new_ex;
#ifdef DEBUG
			printf("%10.3e ", new_ex);
#endif
		}
		// fix!
		if(i == 0)
			ex_table[0] = old_ex;
		else
			ex_table[i] = new_ex;
#ifdef DEBUG
		printf("\n");
#endif
		h /= 2.0;
	}

	fprintf(stderr, "Warning(dfnmdiff): Not converge!\n");
	return new_ex;
}

/* return D(func(?, y)) */
/* reduce a number of function calls */
/* append "max_stage" argument */
void djfnmdiff(DMatrix jacobi_mat, DVector y, void (*func)(DVector, double, DVector), double init_stepsize, double rel_tol, double abs_tol, long int max_stage)
{
	long i, j, col_index, stage_depth;
	long row_dim, col_dim, vec_dim, *stop_index;
	int init_exp, goto_flag;
	double h, inv_h, power4, xtmp, delta, old_ex_i, ex_table_i, correction, eps_M = 1.11022302462515654e-16;
	DVector ex_table[FNM_DIM], new_ex, old_ex;
	DVector ret_ytmp[2], src_ytmp[2], abst_est[FNM_DIM];

	if(max_stage > FNM_DIM)
	{
		fprintf(stderr, "[@djfnmdiff] max_stage %ld is larger than FNM_DIM %d!\n", max_stage, FNM_DIM);
		return;
	}

	row_dim = jacobi_mat->row_dim;
	col_dim = jacobi_mat->col_dim;
	vec_dim = y->dim;

	/* Initialize variables */
	ret_ytmp[0] = init_dvector(row_dim);
	ret_ytmp[1] = init_dvector(row_dim);
	src_ytmp[0] = init_dvector(col_dim);
	src_ytmp[1] = init_dvector(col_dim);
	for(i = 0; i < max_stage; i++)
	{
		ex_table[i] = init_dvector(vec_dim);
		abst_est[i] = init_dvector(vec_dim);
	}
	new_ex = init_dvector(vec_dim);
	old_ex = init_dvector(vec_dim);
	stop_index = (long int *)malloc(sizeof(long int) * vec_dim);

	/* Calculate Jacobi Matrix of func(x, yvec)                              */
	/*                                                                       */
	/*                                                                       */
	/* ex_table[0]    ex_table[1] ex_table[2] ... ex_table[n-1] ex_table[n]  */
    /*             \                                                         */
	/*      old_ex -> new_ex                                                 */
	/*                                 |                                     */
    /*                                 | old_ex := new_ex                    */
	/*                                 v                                     */
	/* ex_table[0]  ex_table[1]   ex_table[2] ... ex_table[n-1] ex_table[n]  */
    /*       ||                \                                             */
	/*     old_ex        old_ex -> new_ex                                    */
	/*                                                                       */
	for(col_index = 0; col_index < col_dim; col_index++)
	{
		//h = 1.0;
		h = init_stepsize;
		for(i = 0; i < vec_dim; i++)
			stop_index[i] = 1;
		for(stage_depth = 0; stage_depth < max_stage; stage_depth++)
		{
			/* Initial sequence */
			subst_dvector(src_ytmp[0], y);
			subst_dvector(src_ytmp[1], y);

			h /= 2.0; // h = 1/2^(stage_depth + prec/2)
			inv_h = 1.0 / h;

			/* set delta = h * y */
			//delta = h * get_dvector_i(y, col_index) ;
			delta = h;

			// Initial List
			xtmp = get_dvector_i(y, col_index) + delta;
			set_dvector_i(src_ytmp[0], col_index, xtmp);
			xtmp = get_dvector_i(y, col_index) - delta;
			set_dvector_i(src_ytmp[1], col_index, xtmp);

			/* ret[0] = f(x, y + h_i * e_i) */
			/* ret[1] = f(x, y - h_i * e_i) */
			func(ret_ytmp[0], xtmp, src_ytmp[0]);
			func(ret_ytmp[1], xtmp, src_ytmp[1]);
#ifdef DEBUG
			printf("%9.1e - %9.1e = ", get_dvector_i(ret_ytmp[0], 0), get_dvector_i(ret_ytmp[1], 0));
#endif
			/* get abst_est for stopping rule */
			for(i = 0; i < vec_dim; i++)
			{
				set_dvector_i(abst_est[stage_depth], i, inv_h * eps_M * dmax(fabs(get_dvector_i(ret_ytmp[0], i)), fabs(get_dvector_i(ret_ytmp[1], i)) ));
			}

			/* ret[0] = f(x, y + h_i * e_i) - f(x, y - h_i * e_i) */
			sub_dvector(ret_ytmp[0], ret_ytmp[0], ret_ytmp[1]);

			/* old_ex = (f(x, y + h_i * e_i) - f(x, y - h_i * e_i)) / (2 * h) */
			cmul_dvector(old_ex, 0.5 * inv_h, ret_ytmp[0]);

			// Main Loop
			power4 = 1.0;
#ifdef DEBUG
			printf("%10.2e(%8.1e, %8.1e) ", get_dvector_i(old_ex, 0), delta, get_dvector_i(y, 0));
#endif
			for(i = 0; i < stage_depth; i++)
			{
				power4 *= 4.0;

				// Extrapolation
				xtmp = 1.0 / (power4 - 1.0);

				goto_flag = 0;
				subst_dvector(new_ex, old_ex);
				for(j = 0; j < vec_dim; j++)
				{
					if(stop_index[j] == 1)
					{
						goto_flag++;
						old_ex_i = get_dvector_i(old_ex, j);
						ex_table_i = get_dvector_i(ex_table[i], j);
						correction = xtmp * (old_ex_i - ex_table_i);
						if(fabs(correction) <= 1.5 * xtmp * 2 * get_dvector_i(abst_est[stage_depth], j))
							stop_index[j] = 0; // no more calc!
						else
							set_dvector_i(new_ex, j, old_ex_i + correction);
					}
				}
//				printf("%3d ", goto_flag);
				// To next step
				if(goto_flag == 0)
					goto set_val;
#ifdef DEBUG
				printf("%10.2e ", get_dvector_i(new_ex, 0));
#endif
				subst_dvector(ex_table[i], old_ex);
				subst_dvector(old_ex, new_ex);
			}
			// fix! 
			if(stage_depth == 0)
				subst_dvector(ex_table[0], old_ex);
			else
				subst_dvector(ex_table[stage_depth], new_ex);
#ifdef DEBUG
			printf("\n");
#endif
		}
#ifdef DEBUG
		fprintf(stderr, "Warning(djfnmdiff): Not converge!(J[%ld], %e)\n", col_index, norm2_dvector(ret_ytmp[0]));
#endif
set_val:
		for(i = 0; i < row_dim; i++)
			set_dmatrix_ij(jacobi_mat, i, col_index, get_dvector_i(new_ex, i));
	}

	free_dvector(ret_ytmp[0]);
	free_dvector(ret_ytmp[1]);
	free_dvector(src_ytmp[0]);
	free_dvector(src_ytmp[1]);
	for(i = 0; i < max_stage; i++)
	{
		free_dvector(ex_table[i]);
		free_dvector(abst_est[i]);
	}
	free_dvector(new_ex);
	free_dvector(old_ex);
	free(stop_index);
}

#ifdef USE_GMP
void mpfrelerr(mpf_t ret, mpf_t approx, mpf_t true_value)
{
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_sub(ret, approx, true_value);
	mpf_abs(ret, ret);

	if(mpf_cmp_ui(true_value, 0UL) != 0)
	{
		mpf_abs(tmp, true_value);
		mpf_div(ret, ret, tmp);
	}

	mpf_clear(tmp);

	return;
}

/* Central Difference */
void mpf_central_diff(mpf_t ret, mpf_t x, void (*func)(mpf_t, mpf_t), mpf_t stepsize)
{
	unsigned long prec;
	mpf_t tmp_x[2], tmp_func[2];

	if(mpf_cmp_ui(stepsize, 0UL) == 0)
	{
		fprintf(stderr, "Divided by 0!(mpf_central_diff)\n");
		mpf_set_ui(ret, 0UL);
		return;
	}

	prec = mpf_get_prec(ret);

	mpf_init2(tmp_x[0], prec);
	mpf_init2(tmp_x[1], prec);
	mpf_init2(tmp_func[0], prec);
	mpf_init2(tmp_func[1], prec);

	//ret = (0.5 * func(x + h) - 0.5 * func(x - h)) / h;
	mpf_sub(tmp_x[0], x, stepsize); mpf_add(tmp_x[1], x, stepsize);
	func(tmp_func[0], tmp_x[0]); func(tmp_func[1], tmp_x[1]);
	mpf_div_ui(tmp_func[0], tmp_func[0], 2UL);
	mpf_div_ui(tmp_func[1], tmp_func[1], 2UL);
	mpf_sub(ret, tmp_func[1], tmp_func[0]);
	mpf_div(ret, ret, stepsize);

	mpf_clear(tmp_x[0]);
	mpf_clear(tmp_x[1]);
	mpf_clear(tmp_func[0]);
	mpf_clear(tmp_func[1]);

	return;
}

void mpf_ulp(mpf_t ret)
{
	mpf_t tmp;
	unsigned long prec;
	long i;

	prec = mpf_get_prec(ret);

	mpf_init2(tmp, prec);

	mpf_set_ui(ret, 1UL);

	for(i = 0; i < prec; i++)
	{
		mpf_div_ui(ret, ret, 2);
		mpf_add_ui(tmp, ret, 1UL);
//		printf("%10d ", i); mpf_out_str(stdout, 10, 0, tmp); printf("\n");
		if(mpf_cmp_ui(tmp, 1UL) == 0)
		{
			mpf_mul_ui(ret, ret, 2UL);
			break;
		}
	}

	mpf_clear(tmp);
}

/* Central Difference for extrapolation */
void mpf_central_diff_err(mpf_t ret, mpf_t x, void (*func)(mpf_t, mpf_t), mpf_t stepsize, mpf_t abs_err)
{
	unsigned long prec;
	mpf_t tmp_x[2], tmp_func[4], eps_M;

	if(mpf_cmp_ui(stepsize, 0UL) == 0)
	{
		fprintf(stderr, "Divided by 0!(mpf_central_diff_f)\n");
		mpf_set_ui(ret, 0UL);
		return;
	}

	prec = mpf_get_prec(ret);

	mpf_init2(tmp_x[0], prec);
	mpf_init2(tmp_x[1], prec);
	mpf_init2(tmp_func[0], prec);
	mpf_init2(tmp_func[1], prec);
	mpf_init2(tmp_func[2], prec);
	mpf_init2(tmp_func[3], prec);
	mpf_init2(eps_M, prec);

	/* eps_M = 2^(-prec) */
	mpf_ulp(eps_M);

	mpf_sub(tmp_x[0], x, stepsize);
	mpf_add(tmp_x[1], x, stepsize);
	func(tmp_func[0], tmp_x[0]);
	func(tmp_func[1], tmp_x[1]);
	mpf_abs(tmp_func[3], tmp_func[0]);
	mpf_abs(tmp_func[4], tmp_func[1]);

	/* Erb(T_i1) = eps_M * max(|f(x+h)|, |f(x-h)|) / h */
	mpf_set(abs_err, mpf_max(tmp_func[3], tmp_func[4]));
	mpf_mul(abs_err, abs_err, eps_M);
	mpf_div(abs_err, abs_err, stepsize);

	//ret = (0.5 * func(x + h) - 0.5 * func(x - h)) / h;
	mpf_sub(ret, tmp_func[1], tmp_func[0]);
	mpf_div(ret, ret, stepsize);
	mpf_div_ui(ret, ret, 2UL);

	mpf_clear(tmp_x[0]);
	mpf_clear(tmp_x[1]);
	mpf_clear(tmp_func[0]);
	mpf_clear(tmp_func[1]);
	mpf_clear(tmp_func[2]);
	mpf_clear(tmp_func[3]);
	mpf_clear(eps_M);

	return;
}
/* Central Difference (5 points, O(h^4))*/
void mpf_central_diff15(mpf_t ret, mpf_t x, void (*func)(mpf_t, mpf_t), mpf_t stepsize)
{
	unsigned long prec;
	long int i;
	mpf_t tmp_x[4], tmp_func[4], tmp_coef[4], tmp_stepsize;

	if(mpf_cmp_ui(stepsize, 0UL) == 0)
	{
		fprintf(stderr, "Divided by 0!(mpf_central_diff15)\n");
		mpf_set_ui(ret, 0UL);
		return;
	}

	prec = mpf_get_prec(ret);

	for(i = 0; i < 4; i++)
	{
		mpf_init2(tmp_x[i], prec);
		mpf_init2(tmp_func[i], prec);
		mpf_init2(tmp_coef[i], prec);
	}
	mpf_init2(tmp_stepsize, prec);

	/* set coef */
	mpf_set_ui(tmp_coef[3], 1UL);
	mpf_div_ui(tmp_coef[3], tmp_coef[3], 12UL);
	mpf_neg(tmp_coef[0], tmp_coef[3]);
	mpf_set_ui(tmp_coef[1], 2UL);
	mpf_div_ui(tmp_coef[1], tmp_coef[1], 3UL);
	mpf_neg(tmp_coef[2], tmp_coef[1]);

	mpf_mul_ui(tmp_stepsize, stepsize, 2UL);
	mpf_add(tmp_x[0], x, tmp_stepsize);
	mpf_sub(tmp_x[3], x, tmp_stepsize);
	mpf_add(tmp_x[1], x, stepsize);
	mpf_sub(tmp_x[2], x, stepsize);

	func(tmp_func[0], tmp_x[0]);
	func(tmp_func[1], tmp_x[1]);
	func(tmp_func[2], tmp_x[2]);
	func(tmp_func[3], tmp_x[3]);

	mpf_mul(tmp_func[0], tmp_func[0], tmp_coef[0]);
	mpf_mul(tmp_func[1], tmp_func[1], tmp_coef[1]);
	mpf_mul(tmp_func[2], tmp_func[2], tmp_coef[2]);
	mpf_mul(tmp_func[3], tmp_func[3], tmp_coef[3]);

	mpf_set(ret, tmp_func[0]);
	mpf_add(ret, ret, tmp_func[1]);
	mpf_add(ret, ret, tmp_func[2]);
	mpf_add(ret, ret, tmp_func[3]);

	mpf_div(ret, ret, stepsize);

	for(i = 0; i < 4; i++)
	{
		mpf_clear(tmp_x[i]);
		mpf_clear(tmp_func[i]);
		mpf_clear(tmp_coef[i]);
	}
	mpf_clear(tmp_stepsize);

	return;
}

/* Central Difference (7 points, O(h^6))*/
void mpf_central_diff17(mpf_t ret, mpf_t x, void (*func)(mpf_t, mpf_t), mpf_t stepsize)
{
	unsigned long prec;
	long int i;
	mpf_t tmp_x[6], tmp_func[6], tmp_coef[6], tmp_stepsize;

	if(mpf_cmp_ui(stepsize, 0UL) == 0)
	{
		fprintf(stderr, "Divided by 0!(mpf_central_diff15)\n");
		mpf_set_ui(ret, 0UL);
		return;
	}

	prec = mpf_get_prec(ret);

	for(i = 0; i < 6; i++)
	{
		mpf_init2(tmp_x[i], prec);
		mpf_init2(tmp_func[i], prec);
		mpf_init2(tmp_coef[i], prec);
	}
	mpf_init2(tmp_stepsize, prec);

	/* set coef */
	mpf_set_ui(tmp_coef[0], 1UL);
	mpf_div_ui(tmp_coef[0], tmp_coef[0], 60UL);
	mpf_neg(tmp_coef[5], tmp_coef[0]);
	mpf_set_ui(tmp_coef[4], 3UL);
	mpf_div_ui(tmp_coef[4], tmp_coef[4], 20UL);
	mpf_neg(tmp_coef[1], tmp_coef[4]);
	mpf_set_ui(tmp_coef[2], 3UL);
	mpf_div_ui(tmp_coef[2], tmp_coef[2], 4UL);
	mpf_neg(tmp_coef[3], tmp_coef[2]);

	mpf_mul_ui(tmp_stepsize, stepsize, 3UL);
	mpf_add(tmp_x[0], x, tmp_stepsize);
	mpf_sub(tmp_x[5], x, tmp_stepsize);
	mpf_mul_ui(tmp_stepsize, stepsize, 2UL);
	mpf_add(tmp_x[1], x, tmp_stepsize);
	mpf_sub(tmp_x[4], x, tmp_stepsize);
	mpf_add(tmp_x[2], x, stepsize);
	mpf_sub(tmp_x[3], x, stepsize);

	func(tmp_func[0], tmp_x[0]);
	func(tmp_func[1], tmp_x[1]);
	func(tmp_func[2], tmp_x[2]);
	func(tmp_func[3], tmp_x[3]);
	func(tmp_func[4], tmp_x[4]);
	func(tmp_func[5], tmp_x[5]);

	mpf_mul(tmp_func[0], tmp_func[0], tmp_coef[0]);
	mpf_mul(tmp_func[1], tmp_func[1], tmp_coef[1]);
	mpf_mul(tmp_func[2], tmp_func[2], tmp_coef[2]);
	mpf_mul(tmp_func[3], tmp_func[3], tmp_coef[3]);
	mpf_mul(tmp_func[4], tmp_func[4], tmp_coef[4]);
	mpf_mul(tmp_func[5], tmp_func[5], tmp_coef[5]);

	mpf_set(ret, tmp_func[0]);
	mpf_add(ret, ret, tmp_func[1]);
	mpf_add(ret, ret, tmp_func[2]);
	mpf_add(ret, ret, tmp_func[3]);
	mpf_add(ret, ret, tmp_func[4]);
	mpf_add(ret, ret, tmp_func[5]);

	mpf_div(ret, ret, stepsize);

	for(i = 0; i < 6; i++)
	{
		mpf_clear(tmp_x[i]);
		mpf_clear(tmp_func[i]);
		mpf_clear(tmp_coef[i]);
	}
	mpf_clear(tmp_stepsize);

	return;
}

/* append "max_stage" argument */
void mpffnmdiff(mpf_t ret, mpf_t x, void (*func)(mpf_t, mpf_t), mpf_t init_stepsize, mpf_t rel_tol, mpf_t abs_tol, long int max_stage, long int *num_stage)
{
	long i, j;
	unsigned long prec;
	mpf_t h, power4, tmp, tmp1, abst_est[FNM_DIM];
	mpf_t ex_table[FNM_DIM], new_ex, old_ex, correction;
	mpf_t tmp_func[2], tmp_x[2];
	mpf_t ex_table_all[FNM_DIM][FNM_DIM];

	if(max_stage > FNM_DIM)
	{
		fprintf(stderr, "[@mpfnmdiff] max_stage %ld is larger than FNM_DIM %d!\n", max_stage, FNM_DIM);
		return;
	}

	// Initialize
	prec = mpf_get_prec(ret);
	mpf_init2(power4, prec);
	mpf_init2(h, prec);
	mpf_init2(new_ex, prec);
	mpf_init2(old_ex, prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp_func[0], prec);
	mpf_init2(tmp_func[1], prec);
	mpf_init2(tmp_x[0], prec);
	mpf_init2(tmp_x[1], prec);
	mpf_init2(correction, prec);
	for(i = 0; i < max_stage; i++)
	{
		mpf_init2(ex_table[i], prec);
		mpf_init2(abst_est[i], prec);
	}

	// Initial Sequence
	// mpf_set_ui(h, 1UL); mpf_div_ui(h, h, 2UL); // h = 1/2
	mpf_set(h, init_stepsize); // given by user
	for(i = 0; i < max_stage; i++)
	{
		mpf_set_ui(power4, 1UL);

		mpf_central_diff_err(old_ex, x, func, h, abst_est[i]);

		for(j = 0; j < i; j++)
		{
			mpf_mul_ui(power4, power4, 4UL);

			// Extrapolation
			// correction = (old_ex - ex_table[j]) / (power4 - 1);
			mpf_sub_ui(tmp1, power4, 1UL);
			mpf_sub(correction, old_ex, ex_table[j]);
			mpf_div(correction, correction, tmp1);

			// new_ex = old_ex + correction;
			mpf_add(new_ex, old_ex, correction);

			// Check convergence by Y.Fukui
			/* tmp  := fabs(correction) */
			/* tmp1 := 1.5 / (power4 - 1) * 2 * abst_est */
			mpf_abs(tmp, correction);
			mpf_div(tmp1, abst_est[i], tmp1);
			mpf_mul_ui(tmp1, tmp1, 3UL);
			if(mpf_cmp(tmp, tmp1) <= 0)
			{
				*num_stage = i;
				goto mpfnmdiff_end;
			}

			/* tmp := fabs(correction) */
			/* tmp1 := rel_tol * fabs(old_ex) + abs_tol) */
			mpf_abs(tmp1, old_ex);
			mpf_mul(tmp1, tmp1, rel_tol);
			mpf_add(tmp1, tmp1, abs_tol);
			if(mpf_cmp(tmp, tmp1) <= 0)
			{
				*num_stage = i;
				goto mpfnmdiff_end;
			}
		
			// To next step
			mpf_set(ex_table[j], old_ex);
			mpf_set(old_ex, new_ex);
		}
		// fix!
		if(i == 0)
			mpf_set(ex_table[0], old_ex);
		else
			mpf_set(ex_table[i], new_ex);

		// h /= 2.0;
		mpf_div_ui(h, h, 2UL);
	}

	fprintf(stderr, "Warning(mpffnmdiff): Not converge!\n");

mpfnmdiff_end:

	mpf_set(ret, new_ex);

	// clear
	mpf_clear(power4);
	mpf_clear(h);
	mpf_clear(new_ex);
	mpf_clear(old_ex);
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(tmp_func[0]);
	mpf_clear(tmp_func[1]);
	mpf_clear(tmp_x[0]);
	mpf_clear(tmp_x[1]);
	mpf_clear(correction);
	for(i = 0; i < max_stage; i++)
	{
		mpf_clear(ex_table[i]);
		mpf_clear(abst_est[i]);
	}

	return;
}



/* return D(func(?, y)) */
/* reduce a number of function calls */
/* append "max_stage" argument */
void mpf_jfnmdiff(MPFMatrix jacobi_mat, MPFVector y, void (*func)(MPFVector, mpf_t, MPFVector), mpf_t init_stepsize, mpf_t rel_tol, mpf_t abs_tol, long int max_stage)
{
	long i, j, col_index, stage_depth;
	long row_dim, col_dim, vec_dim, *stop_index, goto_flag;
	int init_exp;
	unsigned long prec;
	mpf_t h, inv_h, power4, xtmp, delta, tmp[3];
	mpf_t eps_M, correction, ex_table_i, old_ex_i;
	MPFVector ex_table[FNM_DIM], new_ex, old_ex, abst_est[FNM_DIM];
	MPFVector ret_ytmp[2], src_ytmp[2];

	if(max_stage > FNM_DIM)
	{
		fprintf(stderr, "[@mpf_jfnmdiff] max_stage %ld is larger than FNM_DIM %d!\n", max_stage, FNM_DIM);
		return;
	}

	row_dim = jacobi_mat->row_dim;
	col_dim = jacobi_mat->col_dim;
	vec_dim = y->dim;
	prec = jacobi_mat->prec;

	/* Initialize variables */
	mpf_init2(h, prec);
	mpf_init2(inv_h, prec);
	mpf_init2(power4, prec);
	mpf_init2(xtmp, prec);
	mpf_init2(delta, prec);
	mpf_init2(tmp[0], prec);
	mpf_init2(tmp[1], prec);
	mpf_init2(tmp[2], prec);
	mpf_init2(eps_M, prec);
	mpf_init2(ex_table_i, prec);
	mpf_init2(old_ex_i, prec);
	mpf_init2(correction, prec);

	mpf_ulp(eps_M);

	ret_ytmp[0] = init2_mpfvector(row_dim, prec);
	ret_ytmp[1] = init2_mpfvector(row_dim, prec);
	src_ytmp[0] = init2_mpfvector(col_dim, prec);
	src_ytmp[1] = init2_mpfvector(col_dim, prec);
	for(i = 0; i < max_stage; i++)
	{
		ex_table[i] = init2_mpfvector(vec_dim, prec);
		abst_est[i] = init2_mpfvector(vec_dim, prec);
	}
	new_ex = init2_mpfvector(vec_dim, prec);
	old_ex = init2_mpfvector(vec_dim, prec);

	stop_index = (long int *)malloc(sizeof(long int) * vec_dim);

	/* Calculate Jacobi Matrix of func(x, yvec)                              */
	/*                                                                       */
	/*                                                                       */
	/* ex_table[0]    ex_table[1] ex_table[2] ... ex_table[n-1] ex_table[n]  */
    /*             \                                                         */
	/*      old_ex -> new_ex                                                 */
	/*                                 |                                     */
    /*                                 | old_ex := new_ex                    */
	/*                                 v                                     */
	/* ex_table[0]  ex_table[1]   ex_table[2] ... ex_table[n-1] ex_table[n]  */
    /*       ||                \                                             */
	/*     old_ex        old_ex -> new_ex                                    */
	/*                                                                       */
	for(col_index = 0; col_index < col_dim; col_index++)
	{
		//mpf_set_ui(h, 1UL);
		mpf_set(h, init_stepsize);
		for(i = 0; i < vec_dim; i++)
			stop_index[i] = 1;
		for(stage_depth = 0; stage_depth < max_stage; stage_depth++)
		{
			subst_mpfvector(src_ytmp[0], y);
			subst_mpfvector(src_ytmp[1], y);

			mpf_div_ui(h, h, 2); // h = 1
			mpf_ui_div(inv_h, 1UL, h);

			/* set delta = h * y */
			mpf_set(delta, h);

			// Initial List
			mpf_add(xtmp, get_mpfvector_i(y, col_index), delta);
			set_mpfvector_i(src_ytmp[0], col_index, xtmp);
			mpf_sub(xtmp, get_mpfvector_i(y, col_index), delta);
			set_mpfvector_i(src_ytmp[1], col_index, xtmp);

			func(ret_ytmp[0], xtmp, src_ytmp[0]);
			func(ret_ytmp[1], xtmp, src_ytmp[1]);

			/* get abst_est for stopping rule */
			for(i = 0; i < vec_dim; i++)
			{
				mpf_abs(tmp[0], get_mpfvector_i(ret_ytmp[0], i));
				mpf_abs(tmp[1], get_mpfvector_i(ret_ytmp[1], i));
				mpf_set(xtmp, mpf_max(tmp[0], tmp[1]));
				mpf_mul(xtmp, eps_M, xtmp);
				mpf_mul(xtmp, xtmp, inv_h);
				set_mpfvector_i(abst_est[stage_depth], i, xtmp);
			}
#ifdef DEBUG
			printf("%9.1e - %9.1e = ", mpf_get_d(get_mpfvector_i(ret_ytmp[0], 0)), mpf_get_d(get_mpfvector_i(ret_ytmp[1], 0)));
#endif
			/* ret[0] = f(x, y + h_i * e_i) - f(x, y - h_i * e_i) */
			sub_mpfvector(ret_ytmp[0], ret_ytmp[0], ret_ytmp[1]);

			/* old_ex = (f(x, y + h_i * e_i) - f(x, y - h_i * e_i)) / (2 * h) */
			mpf_set_ui(xtmp, 1UL); mpf_div_ui(xtmp, xtmp, 2UL);
			mpf_mul(xtmp, xtmp, inv_h);
			cmul_mpfvector(old_ex, xtmp, ret_ytmp[0]);

			if(stage_depth == 0)
				subst_mpfvector(ex_table[0], old_ex);

			// Main Loop
			mpf_set_ui(power4, 1UL);
#ifdef DEBUG
			printf("%10.2e(%8.1e, %8.1e) ", mpf_get_d(get_mpfvector_i(old_ex, 0)), mpf_get_d(delta), mpf_get_d(get_mpfvector_i(y, 0)));
#endif
			for(i = 0; i < stage_depth; i++)
			{
				mpf_mul_ui(power4, power4, 4UL);

				// Extrapolation
				mpf_sub_ui(xtmp, power4, 1UL);
				mpf_ui_div(xtmp, 1UL, xtmp);

				goto_flag = 0;
				subst_mpfvector(new_ex, old_ex);
				for(j = 0; j < vec_dim; j++)
				{
					if(stop_index[j] == 1)
					{
						goto_flag++;
						mpf_set(old_ex_i, get_mpfvector_i(old_ex, j));
						mpf_set(ex_table_i, get_mpfvector_i(ex_table[i], j));
						mpf_sub(correction, old_ex_i, ex_table_i);
						mpf_mul(correction, correction, xtmp);

						mpf_abs(tmp[0], correction);
						mpf_set(tmp[1], get_mpfvector_i(abst_est[stage_depth], j));
						mpf_mul(tmp[1], tmp[1], xtmp);
						mpf_mul_ui(tmp[1], tmp[1], 3UL);
						// fix! 2006.02/17
						 mpf_mul(tmp[2], old_ex_i, rel_tol);
						 mpf_abs(tmp[2], tmp[2]);
						 mpf_add(tmp[2], tmp[2], abs_tol);
						 //printf("e_r|J|+e_a: %g, est: %g -> ", mpf_get_d(tmp[2]), mpf_get_d(tmp[1]));
						 mpf_set(tmp[1], mpf_max(tmp[2], tmp[1]));
						 //printf("%g\n", mpf_get_d(tmp[1]));
						if(mpf_cmp(tmp[0], tmp[1]) <= 0)
							stop_index[j] = 0; // no more calc!
						else
						{
							mpf_add(tmp[0], old_ex_i, correction);
							set_mpfvector_i(new_ex, j, tmp[0]);
						}
					}
				}

				// To next step
				if(goto_flag == 0)
					goto set_val;
#ifdef DEBUG
				printf("%10.2e ", mpf_get_d(get_mpfvector_i(new_ex, 0)));
#endif
				subst_mpfvector(ex_table[i], old_ex);
				subst_mpfvector(old_ex, new_ex);
			}
			subst_mpfvector(ex_table[stage_depth], new_ex);
#ifdef DEBUG
			printf("\n");
#endif
		}
		norm2_mpfvector(tmp[0], ret_ytmp[0]);
#ifdef DEBUG
		fprintf(stderr, "Warning(mpf_jfnmdiff): Not converge!(J[%ld], %e)\n", col_index, mpf_get_d(tmp[0]));
#endif
set_val:
		printf("con_index: %ld, num_of_stage: %ld (rel_tol: %g, abs_tol: %g)\n", col_index, stage_depth, mpf_get_d(rel_tol), mpf_get_d(abs_tol));
		for(i = 0; i < row_dim; i++)
			set_mpfmatrix_ij(jacobi_mat, i, col_index, get_mpfvector_i(new_ex, i));
	}

	/* free */
	mpf_clear(h);
	mpf_clear(inv_h);
	mpf_clear(xtmp);
	mpf_clear(power4);
	mpf_clear(delta);
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	mpf_clear(eps_M);
	mpf_clear(old_ex_i);
	mpf_clear(ex_table_i);
	mpf_clear(correction);

	free_mpfvector(ret_ytmp[0]);
	free_mpfvector(ret_ytmp[1]);
	free_mpfvector(src_ytmp[0]);
	free_mpfvector(src_ytmp[1]);
	for(i = 0; i < max_stage; i++)
	{
		free_mpfvector(ex_table[i]);
		free_mpfvector(abst_est[i]);
	}
	free_mpfvector(new_ex);
	free_mpfvector(old_ex);
	free(stop_index);

	return;
}
#endif
