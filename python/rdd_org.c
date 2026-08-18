#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdd.h"

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
extern void rdd_sqrt_d(double ret[DDSIZE], double a);

// DD fma
// ret = a * b + c
extern void rdd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE]);

// DD pow
// ret = base^power = exp(power * log(base))
extern void rdd_pow(double ret[DDSIZE], double base[DDSIZE], double power[DDSIZE]);

//#define RDD_CMP(a, b) rdd_cmp(a, b)
//#define RDD_CMP_D(a, b) rdd_cmp_d(a, b)

//#define RDD_CMP_UI(a, b) rdd_cmp_d(a, (double)(b))
int rdd_cmp_ui(double a[DDSIZE], unsigned long b) { return rdd_cmp_d(a, (double)b); }

//#define RDD_SQRT_D(ret, a) rdd_sqrt_d(ret, a)

//#define RDD_SQRT_UI(ret, a) rdd_sqrt_d(ret, (double)(a))
int rdd_sqrt_ui(double a[DDSIZE], unsigned long b) { return rdd_sqrt_d(a, (double)b); }

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
extern void rqd_sqrt_d(double ret[QDSIZE], double a);

// QD fma
// ret = a * b + c
extern void rqd_fma(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE], double c[QDSIZE]);

// QD pow
// ret = base^power = exp(power * log(base))
extern void rqd_pow(double ret[QDSIZE], double base[QDSIZE], double power[QDSIZE]);

//#define RQD_CMP(a, b) rqd_cmp(a, b)
//#define RQD_CMP_D(a, b) rqd_cmp_d(a, b)

//#define RQD_CMP_UI(a, b) rqd_cmp_d(a, (double)(b))
int rqd_cmp_ui(double a[QDSIZE], unsigned long b) { return rqd_cmp_d(a, (double)b); }

//#define RQD_SQRT_D(ret, a) rqd_sqrt_d(ret, a)

//#define RQD_SQRT_UI(ret, a) rqd_sqrt_d(ret, (double)(a))
int rqd_sqrt_ui(double a[QDSIZE], unsigned long b) { return rdd_sqrt_d(a, (double)b); }

// TD print(no appending CR)
extern void rtd_out_str_base(FILE *fp, int base, int length, double val[TDSIZE]);

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rtd_cmp(double a[TDSIZE], double b[TDSIZE]);

// TD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
extern int rtd_cmp_d(double a[TDSIZE], double b);

// TD sqrt_d
extern void rtd_sqrt_d(double ret[TDSIZE], double a);

// TD fma
// ret = a * b + c
extern void rtd_fma(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE], double c[TDSIZE]);

// TD pow
// ret = base^power = exp(power * log(base))
extern void rtd_pow(double ret[TDSIZE], double base[TDSIZE], double power[TDSIZE]);

//#define RTD_CMP_UI(a, b) rtd_cmp_d(a, (double)(b))
int rtd_cmp_ui(double a[TDSIZE], double b) { return rtd_cmp_d(a, (double)b); }

//#define RTD_SQRT_UI(ret, a) rtd_sqrt_d(ret, (double)(a))
int rtd_sqrt_ui(double a[TDSIZE], unsigned long b) { return rtd_sqrt_d(a, (double)b); }


// DD
void set0_dd(double val[DDSIZE]) { SET0_DD(val); }
void rdd_add(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_ADD(ret, a, b); }
void rdd_sub(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_SUB(ret, a, b); }
void rdd_mul(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_MUL(ret, a, b); }
void rdd_div(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_DIV(ret, a, b); }
void rdd_sqrt(double ret[DDSIZE], double a[DDSIZE]) { RDD_SQRT(ret, a); }
void rdd_sqrt_d(double ret[DDSIZE], double a[DDSIZE]) { RDD_SQRT_D(ret, a); }
void rdd_sqrt_ui(double ret[DDSIZE], double a[DDSIZE]) { RDD_SQRT_UI(ret, a); }
void rdd_set_str(char *str, double a[DDSIZE]) { RDD_SET_STR(str, a); }
void rdd_get_str(double a[DDSIZE], char *str) { RDD_GET_STR(a, str); }
void rdd_get_d(double a[DDSIZE]) { RDD_GET_D(a); }
void rdd_set_d(double ret[DDSIZE], double d) { RDD_SET_D(ret, d); }
void rdd_set_ui(double ret[DDSIZE], double d) { RDD_SET_UI(ret, d); }
void rdd_set(double ret[DDSIZE], double org[DDSIZE]) { RDD_SET(ret, org); }
void rdd_neg(double ret[DDSIZE], double a[DDSIZE]) { RDD_NEG(ret, a); }
void rdd_abs(double ret[DDSIZE], double a[DDSIZE]) { RDD_ABS(ret, a); }
void rdd_cmp_ui(double a[DDSIZE], double b[DDSIZE]) { RDD_CMP_UI(a, b); }
void rdd_ui_div(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_UI_DIV(ret, a, b); }
void rdd_ui_sub(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_UI_SUB(ret, a, b); }
void rdd_div_d(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_DIV_D(ret, a, b); }
void rdd_add_d(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_ADD_D(ret, a, b); }
void rdd_sub_d(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_SUB_D(ret, a, b); }
void rdd_mul_d(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_MUL_D(ret, a, b); }
void rdd_div_ui(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_DIV_UI(ret, a, b); }
void rdd_add_ui(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_ADD_UI(ret, a, b); }
void rdd_sub_ui(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_SUB_UI(ret, a, b); }
void rdd_mul_ui(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE]) { RDD_MUL_UI(ret, a, b); }

void rdd_pi(double ret[DDSIZE]) { RDD_PI(ret); }
void rdd_exp(double ret[DDSIZE], double x[DDSIZE]) { RDD_EXP(ret, x); }
void rdd_sin(double ret[DDSIZE], double x[DDSIZE]) { RDD_SIN(ret, x); }
void rdd_cos(double ret[DDSIZE], double x[DDSIZE]) { RDD_COS(ret, x); }
void rdd_log(double ret[DDSIZE], double x[DDSIZE]) { RDD_LOG(ret, x); }
void rdd_asin(double ret[DDSIZE], double x[DDSIZE]) { RDD_ASIN(ret, x); }
void rdd_acos(double ret[DDSIZE], double x[DDSIZE]) { RDD_ACOS(ret, x); }

// TD
void set0_td(double val[TDSIZE]) { SET0_TD(val); }
void rtd_add(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_ADD(ret, a, b); }
void rtd_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_SUB(ret, a, b); }
void rtd_mul(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_MUL(ret, a, b); }
void rtd_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_DIV(ret, a, b); }
void rtd_sqrt(double ret[TDSIZE], double a[TDSIZE]) { RTD_SQRT(ret, a); }
void rtd_sqrt_d(double ret[TDSIZE], double a[TDSIZE]) { RTD_SQRT_D(ret, a); }
void rtd_sqrt_ui(double ret[TDSIZE], double a[TDSIZE]) { RTD_SQRT_UI(ret, a); }
void rtd_set_str(char *str, double a[TDSIZE]) { RTD_SET_STR(str, a); }
void rtd_get_str(double a[TDSIZE], char *str) { RTD_GET_STR(a, str); }
void rtd_get_d(a[TDSIZE]) { RTD_GET_D(a); }
void rtd_set_d(double ret[TDSIZE], double d) { RTD_SET_D(ret, d); }
void rtd_set_ui(double ret[TDSIZE], unsigned long d) { RTD_SET_UI(ret, d); }
void rtd_set(double ret[TDSIZE], double org[TDSIZE]) { RTD_SET(ret, org); }
void rtd_neg(double ret[TDSIZE], double a[TDSIZE]) { RTD_NEG(ret, a); }
void rtd_abs(double ret[TDSIZE], double a[TDSIZE]) { RTD_ABS(ret, a); }
void rtd_cmp_ui(double a[TDSIZE], unsigned long b) { RTD_CMP_UI(a, b); }
void rtd_ui_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_UI_DIV(ret, a, b); }
void rtd_ui_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_UI_SUB(ret, a, b); }
void rtd_div_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_DIV_D(ret, a, b); }
void rtd_add_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_ADD_D(ret, a, b); }
void rtd_sub_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_SUB_D(ret, a, b); }
void rtd_mul_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_MUL_D(ret, a, b); }
void rtd_div_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_DIV_UI(ret, a, b); }
void rtd_add_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_ADD_UI(ret, a, b); }
void rtd_sub_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_SUB_UI(ret, a, b); }
void rtd_mul_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RTD_MUL_UI(ret, a, b); }

void rtd_pi(double ret[TDSIZE]) { RTD_PI(ret); }
void rtd_exp(double ret[TDSIZE], double x[TDSIZE]) { RTD_EXP(ret, x); }
void rtd_sin(double ret[TDSIZE], double x[TDSIZE]) { RTD_SIN(ret, x); }
void rtd_cos(double ret[TDSIZE], double x[TDSIZE]) { RTD_COS(ret, x); }
void rtd_log(double ret[TDSIZE], double x[TDSIZE]) { RTD_LOG(ret, x); }
void rtd_asin(double ret[TDSIZE], double x[TDSIZE]) { RTD_ASIN(ret, x); }
void rtd_acos(double ret[TDSIZE], double x[TDSIZE]) { RTD_ACOS(ret, x); }

// QD
void set0_qd(double val[TDSIZE]) { SET0_QD(val); }
void rqd_add(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_ADD(ret, a, b); }
void rqd_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_SUB(ret, a, b); }
void rqd_mul(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_MUL(ret, a, b); }
void rqd_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_DIV(ret, a, b); }
void rqd_sqrt(double ret[TDSIZE], double a[TDSIZE]) { RQD_SQRT(ret, a); }
void rqd_sqrt_d(double ret[TDSIZE], double a) { RQD_SQRT_D(ret, a); }
void rqd_sqrt_ui(double ret[TDSIZE], unsigned long a) { RQD_SQRT_UI(ret, a); }
void rqd_out_str(double a[TDSIZE]) { RQD_OUT_STR(a); }
void rqd_set_str(char *str, double a[TDSIZE]) { RQD_SET_STR(str, a); 
void rqd_get_str(double a[TDSIZE], char *str) { RQD_GET_STR(a, str); 
void rqd_get_d(double a[TDSIZE]) { RQD_GET_D(a); }
void rqd_set_d(double ret[TDSIZE], double d) { RQD_SET_D(ret, d); }
void rqd_set_ui(double ret[TDSIZE], unsigned long d) { RQD_SET_UI(ret, d); }
void rqd_set(double ret[TDSIZE], double org[TDSIZE]) { RQD_SET(ret, org); }
void rqd_neg(double ret[TDSIZE], double a[TDSIZE]) { RQD_NEG(ret, a); }
void rqd_abs(double ret[TDSIZE], double a[TDSIZE]) { RQD_ABS(ret, a); }
void rqd_cmp_ui(double a[TDSIZE], unsigned long b) { RQD_CMP_UI(a, b); }
void rqd_ui_div(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_UI_DIV(ret, a, b); }
void rqd_ui_sub(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_UI_SUB(ret, a, b); }
void rqd_div_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_DIV_D(ret, a, b); }
void rqd_add_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_ADD_D(ret, a, b); }
void rqd_sub_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_SUB_D(ret, a, b); }
void rqd_mul_d(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_MUL_D(ret, a, b); }
void rqd_div_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_DIV_UI(ret, a, b); }
void rqd_add_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_ADD_UI(ret, a, b); }
void rqd_sub_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_SUB_UI(ret, a, b); }
void rqd_mul_ui(double ret[TDSIZE], double a[TDSIZE], double b[TDSIZE]) { RQD_MUL_UI(ret, a, b); }

void rqd_pi(double ret[TDSIZE]) { RQD_PI(ret); }
void rqd_exp(double ret[TDSIZE], double x[TDSIZE]) { RQD_EXP(ret, x); }
void rqd_sin(double ret[TDSIZE], double x[TDSIZE]) { RQD_SIN(ret, x); }
void rqd_cos(double ret[TDSIZE], double x[TDSIZE]) { RQD_COS(ret, x); }
void rqd_log(double ret[TDSIZE], double x[TDSIZE]) { RQD_LOG(ret, x); }
void rqd_asin(double ret[TDSIZE], double x[TDSIZE]) { RQD_ASIN(ret, x); }
void rqd_acos(double ret[TDSIZE], double x[TDSIZE]) { RQD_ACOS(ret, x); }
