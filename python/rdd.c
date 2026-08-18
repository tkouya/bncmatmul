#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define USE_RDD_FUNCTIONS
#define USE_RTD_FUNCTIONS
#define USE_RQD_FUNCTIONS
#include "rdd.h"

#ifndef USE_GMP
	#define USE_GMP
#endif // USE_GMP
#ifndef USE_MPFR
	#define USE_MPFR
#endif // USE_MPFR
#include "mpfr_dtq_sd.h" // 2025-12-19(Fri) T.Kouya


// DD print(no appending CR)
extern void rdd_out_str_base(FILE *fp, int base, int length, double val[DDSIZE]);

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rdd_cmp(double a[DDSIZE], double b[DDSIZE]);

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rdd_cmp_d(double a[DDSIZE], double b);

// DD sqrt_d
//extern void rdd_sqrt_d(double ret[DDSIZE], double a);
void rdd_sqrt_d(double ret[DDSIZE], double a) { rdd_sqrt_d_mpfr(ret, a); };

// DD fma
// ret = a * b + c
extern void rdd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE]);

// DD pow
// ret = base^power = exp(power * log(base))
extern void rdd_pow(double ret[DDSIZE], double base[DDSIZE], double power[DDSIZE]);

//#define RDD_CMP(a, b) rdd_cmp(a, b)
//#define RDD_CMP_D(a, b) rdd_cmp_d(a, b)

//#define RDD_CMP_UI(a, b) rdd_cmp_d(a, (double)(b))
int rdd_cmp_ui(double a[DDSIZE], unsigned long b) { return RDD_CMP_UI(a, b); }

//#define RDD_SQRT_D(ret, a) rdd_sqrt_d(ret, a)

//#define RDD_SQRT_UI(ret, a) rdd_sqrt_d(ret, (double)(a))
void rdd_sqrt_ui(double a[DDSIZE], unsigned long b) { rdd_sqrt_d(a, (double)b); }

// QD print(no appending CR)
extern void rqd_out_str_base(FILE *fp, int base, int length, double val[QDSIZE]);

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rqd_cmp(double a[QDSIZE], double b[QDSIZE]);

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rqd_cmp_d(double a[QDSIZE], double b);

// QD sqrt_d
//extern void rqd_sqrt_d(double ret[QDSIZE], double a);
void rqd_sqrt_d(double ret[QDSIZE], double a) { rqd_sqrt_d_mpfr(ret, a); }


// QD fma
// ret = a * b + c
extern void rqd_fma(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE], double c[QDSIZE]);

// QD pow
// ret = base^power = exp(power * log(base))
extern void rqd_pow(double ret[QDSIZE], double base[QDSIZE], double power[QDSIZE]);



// DD
void set0_dd(double val[DDSIZE]) { val[0] = (double)0.0; val[1] = (double)0.0; }
void rdd_set0(double val[DDSIZE]) { set0_dd(val); }
void rdd_add(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_ADD(ret, a, b); }
void rdd_sub(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_SUB(ret, a, b); }
void rdd_mul(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_MUL(ret, a, b); }
void rdd_div(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_DIV(ret, a, b); }
void rdd_sqrt(double ret[DDSIZE], double a[DDSIZE]) { RDD_SQRT(ret, a); }
extern void rdd_sqrt_d(double ret[DDSIZE], double a);
//void rdd_set_str(char *str, double a[DDSIZE]) { RDD_SET_STR(str, a); }
//void rdd_get_str(double a[DDSIZE], char *str) { RDD_GET_STR(a, str); }
void rdd_get_d(double a[DDSIZE]) { RDD_GET_D(a); }
void rdd_set_d(double ret[DDSIZE], double d) { RDD_SET_D(ret, d); }
void rdd_set_ui(double ret[DDSIZE], double d) { RDD_SET_UI(ret, d); }
void rdd_set(double ret[DDSIZE], double org[DDSIZE]) { RDD_SET(ret, org); }
void rdd_neg(double ret[DDSIZE], double a[DDSIZE]) { RDD_NEG(ret, a); }
void rdd_abs(double ret[DDSIZE], double a[DDSIZE]) { RDD_ABS(ret, a); }
extern int rdd_cmp_ui(double a[DDSIZE], unsigned long b);
void rdd_ui_div(double ret[DDSIZE], unsigned long a, double b[DDSIZE]) { RDD_UI_DIV(ret, a, b); }
void rdd_ui_sub(double ret[DDSIZE], unsigned long a, double b[DDSIZE]) { RDD_UI_SUB(ret, a, b); }
void rdd_div_d(double ret[DDSIZE], double a[DDSIZE], double b) { RDD_DIV_D(ret, a, b); }
void rdd_add_d(double ret[DDSIZE], double a[DDSIZE], double b) { RDD_ADD_D(ret, a, b); }
void rdd_sub_d(double ret[DDSIZE], double a[DDSIZE], double b) { RDD_SUB_D(ret, a, b); }
void rdd_mul_d(double ret[DDSIZE], double a[DDSIZE], double b) { RDD_MUL_D(ret, a, b); }
void rdd_div_ui(double ret[DDSIZE], double a[DDSIZE], unsigned long b) { RDD_DIV_UI(ret, a, b); }
void rdd_add_ui(double ret[DDSIZE], double a[DDSIZE], unsigned long b) { RDD_ADD_UI(ret, a, b); }
void rdd_sub_ui(double ret[DDSIZE], double a[DDSIZE], unsigned long b) { RDD_SUB_UI(ret, a, b); }
void rdd_mul_ui(double ret[DDSIZE], double a[DDSIZE], unsigned long b) { RDD_MUL_UI(ret, a, b); }

double *dd_vec_init(int dim)
{
	double *ret = NULL;

	ret = (double *)calloc(dim * DDSIZE, sizeof(double));
	return ret;
}

double *dd_mat_init(int row_dim, int col_dim)
{
	double *ret = NULL;

	ret = (double *)calloc(row_dim * col_dim * DDSIZE, sizeof(double));
	return ret;
}

// ret := mat_a * vec_b
void dd_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
{
	int i, j, ij_index, i_index;
	double ab[DDSIZE];

	for(i = 0; i < a_row_dim; i++)
	{
		i_index = i * DDSIZE;
		rdd_set0(&ret[i_index]);
		for(j = 0; j < a_col_dim; j++)
		{
			ij_index = (i * a_col_dim + j) * DDSIZE;
			rdd_mul(ab, &a[ij_index], &b[j * DDSIZE]);
			rdd_add(&ret[i_index], &ret[i_index], ab);
		}
	}
}

// ret := mat_a * mat_b
void dd_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
{
	int i, j, k;
	int ij_index, ik_index, kj_index;
	double ab[DDSIZE];

	for(i = 0; i < ret_row_dim; i++)
	{
		for(j = 0; j < ret_col_dim; j++)
		{
			ij_index = (i * ret_col_dim + j) * DDSIZE;
			rdd_set0(&ret[ij_index]);

			for(k = 0; k < a_col_dim; k++)
			{
				ik_index = (i * a_col_dim + k) * DDSIZE;
				kj_index = (k * b_col_dim + j) * DDSIZE;
				//ret[ij_index] += a[ik_index] * b[kj_index];
				rdd_mul(ab, &a[ik_index], &b[kj_index]);
				rdd_add(&ret[ij_index], &ret[ij_index], ab);
			}
		}
	}
}

// TD
void set0_td(double val[TDSIZE]) { SET0_TD(val); }
void rtd_set0(double val[TDSIZE]) { SET0_TD(val); }
void rtd_add(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_ADD(ret, a, b); }
void rtd_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_SUB(ret, a, b); }
void rtd_mul(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_MUL(ret, a, b); }
void rtd_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_DIV(ret, a, b); }
void rtd_sqrt(double ret[TDSIZE], double a[TDSIZE]) { RTD_SQRT(ret, a); }
//void rtd_sqrt_d(double ret[TDSIZE], double a) { RTD_SQRT_D(ret, a); }
void rtd_sqrt_ui(double ret[TDSIZE], unsigned long a) { RTD_SQRT_UI(ret, a); }
//void rtd_set_str(char *str, double a[TDSIZE]) { RTD_SET_STR(str, a); }
//void rtd_get_str(double a[TDSIZE], char *str) { RTD_GET_STR(a, str); }
void rtd_get_d(double a[TDSIZE]) { RTD_GET_D(a); }
void rtd_set_d(double ret[TDSIZE], double d) { RTD_SET_D(ret, d); }
void rtd_set_ui(double ret[TDSIZE], unsigned long d) { RTD_SET_UI(ret, d); }
void rtd_set(double ret[TDSIZE], double org[TDSIZE]) { RTD_SET(ret, org); }
void rtd_neg(double ret[TDSIZE], double a[TDSIZE]) { RTD_NEG(ret, a); }
void rtd_abs(double ret[TDSIZE], double a[TDSIZE]) { RTD_ABS(ret, a); }
void rtd_cmp_ui(double a[TDSIZE], unsigned long b) { RTD_CMP_UI(a, b); }
void rtd_ui_div(double ret[TDSIZE], unsigned long a, double b[TDSIZE]) { RTD_UI_DIV(ret, a, b); }
void rtd_ui_sub(double ret[TDSIZE], unsigned long a, double b[TDSIZE]) { RTD_UI_SUB(ret, a, b); }
void rtd_div_d(double ret[TDSIZE], double a[TDSIZE], double b) { RTD_DIV_D(ret, a, b); }
void rtd_add_d(double ret[TDSIZE], double a[TDSIZE], double b) { RTD_ADD_D(ret, a, b); }
void rtd_sub_d(double ret[TDSIZE], double a[TDSIZE], double b) { RTD_SUB_D(ret, a, b); }
void rtd_mul_d(double ret[TDSIZE], double a[TDSIZE], double b) { RTD_MUL_D(ret, a, b); }
void rtd_div_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RTD_DIV_UI(ret, a, b); }
void rtd_add_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RTD_ADD_UI(ret, a, b); }
void rtd_sub_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RTD_SUB_UI(ret, a, b); }
void rtd_mul_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RTD_MUL_UI(ret, a, b); }

void rtd_pi(double ret[TDSIZE]) { RTD_PI(ret); }
//void rtd_exp(double ret[TDSIZE], double x[TDSIZE]) { RTD_EXP(ret, x); }
//void rtd_sin(double ret[TDSIZE], double x[TDSIZE]) { RTD_SIN(ret, x); }
//void rtd_cos(double ret[TDSIZE], double x[TDSIZE]) { RTD_COS(ret, x); }
//void rtd_log(double ret[TDSIZE], double x[TDSIZE]) { RTD_LOG(ret, x); }
//void rtd_asin(double ret[TDSIZE], double x[TDSIZE]) { RTD_ASIN(ret, x); }
//void rtd_acos(double ret[TDSIZE], double x[TDSIZE]) { RTD_ACOS(ret, x); }

double *td_vec_init(int dim)
{
	double *ret = NULL;

	ret = (double *)calloc(dim * TDSIZE, sizeof(double));
	return ret;
}

double *td_mat_init(int row_dim, int col_dim)
{
	double *ret = NULL;

	ret = (double *)calloc(row_dim * col_dim * TDSIZE, sizeof(double));
	return ret;
}

// ret := mat_a * vec_b
void td_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
{
	int i, j, ij_index, i_index;
	double ab[TDSIZE];

	for(i = 0; i < a_row_dim; i++)
	{
		i_index = i * TDSIZE;
		rtd_set0(&ret[i_index]);
		for(j = 0; j < a_col_dim; j++)
		{
			ij_index = (i * a_col_dim + j) * TDSIZE;
			rtd_mul(ab, &a[ij_index], &b[j * TDSIZE]);
			rtd_add(&ret[i_index], &ret[i_index], ab);
		}
	}
}

void td_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
{
	int i, j, k;
	int ij_index, ik_index, kj_index;
	double ab[TDSIZE];

	for(i = 0; i < ret_row_dim; i++)
	{
		for(j = 0; j < ret_col_dim; j++)
		{
			ij_index = (i * ret_col_dim + j) * TDSIZE;
			rtd_set0(&ret[ij_index]);

			for(k = 0; k < a_col_dim; k++)
			{
				ik_index = (i * a_col_dim + k) * TDSIZE;
				kj_index = (k * b_col_dim + j) * TDSIZE;
				//ret[ij_index] += a[ik_index] * b[kj_index];
				rtd_mul(ab, &a[ik_index], &b[kj_index]);
				rtd_add(&ret[ij_index], &ret[ij_index], ab);
			}
		}
	}
}


// QD
void set0_qd(double val[TDSIZE]) { SET0_QD(val); }
void rqd_set0(double val[TDSIZE]) { SET0_QD(val); }
void rqd_add(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_ADD(ret, a, b); }
void rqd_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_SUB(ret, a, b); }
void rqd_mul(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_MUL(ret, a, b); }
void rqd_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_DIV(ret, a, b); }
void rqd_sqrt(double ret[TDSIZE], double a[TDSIZE]) { RQD_SQRT(ret, a); }
//void rqd_sqrt_d(double ret[TDSIZE], double a) { RQD_SQRT_D(ret, a); }
void rqd_sqrt_ui(double ret[TDSIZE], unsigned long a) { RQD_SQRT_UI(ret, a); }
//void rqd_out_str(double a[TDSIZE]) { RQD_OUT_STR(a); }
//void rqd_set_str(char *str, double a[TDSIZE]) { RQD_SET_STR(str, a); }
//void rqd_get_str(double a[TDSIZE], char *str) { RQD_GET_STR(a, str); }
void rqd_get_d(double a[TDSIZE]) { RQD_GET_D(a); }
void rqd_set_d(double ret[TDSIZE], double d) { RQD_SET_D(ret, d); }
void rqd_set_ui(double ret[TDSIZE], unsigned long d) { RQD_SET_UI(ret, d); }
void rqd_set(double ret[TDSIZE], double org[TDSIZE]) { RQD_SET(ret, org); }
void rqd_neg(double ret[TDSIZE], double a[TDSIZE]) { RQD_NEG(ret, a); }
void rqd_abs(double ret[TDSIZE], double a[TDSIZE]) { RQD_ABS(ret, a); }
void rqd_cmp_ui(double a[TDSIZE], unsigned long b) { RQD_CMP_UI(a, b); }
void rqd_ui_div(double ret[TDSIZE], unsigned long a, double b[TDSIZE]) { RQD_UI_DIV(ret, a, b); }
void rqd_ui_sub(double ret[TDSIZE], unsigned long a, double b[TDSIZE]) { RQD_UI_SUB(ret, a, b); }
void rqd_div_d(double ret[TDSIZE], double a[TDSIZE], double b) { RQD_DIV_D(ret, a, b); }
void rqd_add_d(double ret[TDSIZE], double a[TDSIZE], double b) { RQD_ADD_D(ret, a, b); }
void rqd_sub_d(double ret[TDSIZE], double a[TDSIZE], double b) { RQD_SUB_D(ret, a, b); }
void rqd_mul_d(double ret[TDSIZE], double a[TDSIZE], double b) { RQD_MUL_D(ret, a, b); }
void rqd_div_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RQD_DIV_UI(ret, a, b); }
void rqd_add_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RQD_ADD_UI(ret, a, b); }
void rqd_sub_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RQD_SUB_UI(ret, a, b); }
void rqd_mul_ui(double ret[TDSIZE], double a[TDSIZE], unsigned long b) { RQD_MUL_UI(ret, a, b); }

void rqd_pi(double ret[TDSIZE]) { RQD_PI(ret); }
void rqd_exp(double ret[TDSIZE], double x[TDSIZE]) { RQD_EXP(ret, x); }
void rqd_sin(double ret[TDSIZE], double x[TDSIZE]) { RQD_SIN(ret, x); }
void rqd_cos(double ret[TDSIZE], double x[TDSIZE]) { RQD_COS(ret, x); }
void rqd_log(double ret[TDSIZE], double x[TDSIZE]) { RQD_LOG(ret, x); }
void rqd_asin(double ret[TDSIZE], double x[TDSIZE]) { RQD_ASIN(ret, x); }
void rqd_acos(double ret[TDSIZE], double x[TDSIZE]) { RQD_ACOS(ret, x); }

double *qd_vec_init(int dim)
{
	double *ret = NULL;

	ret = (double *)calloc(dim * QDSIZE, sizeof(double));
	return ret;
}

double *qd_mat_init(int row_dim, int col_dim)
{
	double *ret = NULL;

	ret = (double *)calloc(row_dim * col_dim * QDSIZE, sizeof(double));
	return ret;
}

// ret := mat_a * vec_b
void qd_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
{
	int i, j, ij_index, i_index;
	double ab[QDSIZE];

	for(i = 0; i < a_row_dim; i++)
	{
		i_index = i * QDSIZE;
		rqd_set0(&ret[i_index]);
		for(j = 0; j < a_col_dim; j++)
		{
			ij_index = (i * a_col_dim + j) * QDSIZE;
			rqd_mul(ab, &a[ij_index], &b[j * QDSIZE]);
			rqd_add(&ret[i_index], &ret[i_index], ab);
		}
	}
}

void qd_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
{
	int i, j, k;
	int ij_index, ik_index, kj_index;
	double ab[QDSIZE];

	for(i = 0; i < ret_row_dim; i++)
	{
		for(j = 0; j < ret_col_dim; j++)
		{
			ij_index = (i * ret_col_dim + j) * QDSIZE;
			rqd_set0(&ret[ij_index]);

			for(k = 0; k < a_col_dim; k++)
			{
				ik_index = (i * a_col_dim + k) * QDSIZE;
				kj_index = (k * b_col_dim + j) * QDSIZE;
				//ret[ij_index] += a[ik_index] * b[kj_index];
				rqd_mul(ab, &a[ik_index], &b[kj_index]);
				rqd_add(&ret[ij_index], &ret[ij_index], ab);
			}
		}
	}
}

unsigned int rdd_old_cw;

// Initialize rdd library
void rdd_start(void)
{
	fpu_fix_start(&rdd_old_cw);
}

// Finalize rdd library
void rdd_end(void)
{
	fpu_fix_end(&rdd_old_cw);
}

