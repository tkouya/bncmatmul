/********************************************************************************/
/* poly.h: Algebraic Equations and Polynomials                                  */
/* copyright (c) 2002-2025 Tomonori Kouya                                       */
/*                                                                              */
/* Ver. 0.4 2025-06-18: append CDPoly                                           */
/* Ver. 0.3 2025-02-03: append DDPoly, TDPoly, and QDPoly                       */
/* Ver. 0.2 2025-01-10: separate polynomial and DKA                             */
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
#ifndef __POLY_H_
#define __POLY_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdio.h>
#include <stdlib.h>
#include <complex.h> // double _Complex
#include "bnc_common.h"

// [dd, td, qd]float
#include "rdd.h"

// [cdd, ctd, cqd]float
#include "rcdd.h"

/*************************************************/
/* Polynomial Type: FPoly, DPoly, MPFPoly        */
/*                  CFPoly, CDPoly, CMPFPoly     */
/*************************************************/
typedef struct{
	float *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} fpoly;

typedef fpoly *FPoly;

typedef struct{
	fcmplx *coef;
    //float _Complex *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} cfpoly;

typedef cfpoly *CFPoly;

typedef struct{
	double *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} dpoly;

typedef dpoly *DPoly;

typedef struct{
	//dcmplx *coef;
    double _Complex *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} cdpoly;

typedef cdpoly *CDPoly;

// DDPoly, CDDPoly
typedef struct {
	ddfloat *coef;
	ddfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} ddpoly;
typedef ddpoly *DDPoly;

typedef struct {
	cddfloat *coef;
	cddfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} cddpoly;
typedef cddpoly *CDDPoly;

// TDPoly, CTDPoly
typedef struct {
	tdfloat *coef;
	tdfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} tdpoly;
typedef tdpoly *TDPoly;

typedef struct {
	ctdfloat *coef;
	ctdfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} ctdpoly;
typedef ctdpoly *CTDPoly;

// QDPoly, CQDPoly
typedef struct {
	qdfloat *coef;
	qdfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} qdpoly;
typedef qdpoly *QDPoly;

typedef struct {
	cqdfloat *coef;
	cqdfloat zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len;
} cqdpoly;
typedef cqdpoly *CQDPoly;

#ifdef USE_GMP

typedef struct{
	unsigned long int prec;
	mpf_t *coef;
	mpf_t zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
//	mpf_ptr *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} mpfpoly;

typedef mpfpoly *MPFPoly;

// Old implementation on MPFCmplx
typedef struct{
	unsigned long int prec;
	mpfcmplx *coef;
	//mpc_t *coef;
	long int deg;
	long int max_len; /* coef[max_len] */
} _bncold_cmpfpoly;

typedef _bncold_cmpfpoly *_bncold_CMPFPoly;

// New implementation on MPC
typedef struct{
	unsigned long int prec;
	//mpfcmplx *coef;
	mpc_t *coef;
	mpc_t zero; /* ... + 0 x^(deg+2) + 0 x^(deg+1) + coef[i] x^deg + ... */
	long int deg;
	long int max_len; /* coef[max_len] */
} cmpfpoly;

typedef cmpfpoly *CMPFPoly;
#endif // USE_GMP

/*************************************************/
/* Array                                         */
/*************************************************/
/* farray */
typedef struct {
	float *array; /* array */
	long int size; /* Length of array */
} farray;

typedef farray *FArray;

/* cfarray */
typedef struct {
	fcmplx *array; /* array */
	//float _Complex *array; /* array */
	long int size; /* Length of array */
} cfarray;

typedef cfarray *CFArray;

/* darray */
typedef struct {
	double *array; /* array */
	long int size; /* Length of array */
} darray;

typedef darray *DArray;

/* cdarray */
typedef struct {
	dcmplx *array; /* array */
    //double _Complex *array; /* array */
	long int size; /* Length of array */
} cdarray;

typedef cdarray *CDArray;

#ifdef USE_GMP
/* mpfarray */
typedef struct {
	unsigned long prec; /* preision */
	mpf_t *array; /* array */
	long int size; /* Length of array */
} mpfarray;

typedef mpfarray *MPFArray;

/* Old cmpfarray */
typedef struct {
	unsigned long prec; /* preision */
	mpfcmplx *array; /* array */
	//mpc_t *array; /* array */
	long int size; /* Length of array */
} _bncold_cmpfarray;

typedef _bncold_cmpfarray *_bncold_CMPFArray;

/* cmpfarray */
typedef struct {
	unsigned long prec; /* preision */
	//mpfcmplx *array; /* array */
	mpc_t *array; /* array */
	long int size; /* Length of array */
} cmpfarray;

typedef cmpfarray *CMPFArray;
#endif // USE_GMP

/*************************************************/
/* poly.c: Algebraic equations and Polynomial    */
/*************************************************/
// Float
FPoly init_fpoly(long int);
void free_fpoly(FPoly);
float get_fpoly_i(FPoly, long int);
long int setdegree_fpoly(FPoly);
void set_fpoly_i(FPoly, long int, float);
#define gfpi get_fpoly_i
#define sfpi set_fpoly_i
void print_fpoly(FPoly);
void add_fpoly(FPoly, FPoly, FPoly);
void sub_fpoly(FPoly, FPoly, FPoly);
void cmul_fpoly(FPoly, float, FPoly);
void subst_fpoly(FPoly, FPoly);
void set0_fpoly(FPoly);
long int max_abscoef_fpoly(FPoly);
long int num_nonzero_fpoly(FPoly);
void diff_fpoly(FPoly);
float eval_fpoly(FPoly, float);
float eval_diff_fpoly(FPoly, float);
void ceval_fpoly(FCmplx, FPoly, FCmplx);
void ceval_diff_fpoly(FCmplx, FPoly, FCmplx);
//void ceval_fpoly(float _Complex *, FPoly, float _Complex *);
//void ceval_diff_fpoly(float _Complex *, FPoly, float _Complex *);

// Double
DPoly init_dpoly(long int);
void free_dpoly(DPoly);
DPoly init_set_dpoly(DPoly);
double get_dpoly_i(DPoly, long int);
long int setdegree_dpoly(DPoly);
void set_dpoly_i(DPoly, long int, double);
#define gdpi get_dpoly_i
#define sdpi set_dpoly_i
void print_dpoly(DPoly);
void add_dpoly(DPoly, DPoly, DPoly);
void sub_dpoly(DPoly, DPoly, DPoly);
void add2_dpoly(DPoly, DPoly);
void sub2_dpoly(DPoly, DPoly);
void mul_dpoly(DPoly, DPoly, DPoly);
void cmul_dpoly(DPoly, double, DPoly);
void subst_dpoly(DPoly, DPoly);
void set0_dpoly(DPoly);
long int max_abscoef_dpoly(DPoly);
long int num_nonzero_dpoly(DPoly);
void diff_dpoly(DPoly);
double eval_dpoly_horner(DPoly, double);
double eval_dpoly_estrin(DPoly, double);
#define eval_dpoly eval_dpoly_horner
void div_dpoly(DPoly, DPoly, DPoly, DPoly);
void xpow_mul_dpoly(DPoly, long int, DPoly);
double eval_diff_dpoly(DPoly, double);
void div_dpoly(DPoly quo, DPoly rem, DPoly a, DPoly b);
void ceval_dpoly_horner(DCmplx, DPoly, DCmplx);
void ceval_dpoly_estrin(DCmplx, DPoly, DCmplx);
#define ceval_dpoly ceval_dpoly_horner
void ceval_diff_dpoly(DCmplx, DPoly, DCmplx);
//void ceval_dpoly(double _Complex *, DPoly, double _Complex *);
//void ceval_diff_dpoly(double _Complex *, DPoly, double _Complex *);
// AVX2
#include "dlinear.h"
#include "cdlinear.h"
double _bncavx2_eval_dpoly_estrin(DPoly a, double x);
void _bncavx2_ceval_dpoly_estrin(DCmplx ret, DPoly a, DCmplx x);

// Double Complex
CDPoly init_cdpoly(long int);
void free_cdpoly(CDPoly);
CDPoly init_set_cdpoly(CDPoly);
CDPoly init_set_cdpoly_dpoly(DPoly);
void set_cdpoly_dpoly(CDPoly, DPoly);
double _Complex get_cdpoly_i(CDPoly, long int);
long int setdegree_cdpoly(CDPoly);
void set_cdpoly_i(CDPoly, long int, double _Complex);
void set_cdpoly_i_d(CDPoly, long int, double);
#define gcdpi get_cdpoly_i
#define scdpi set_cdpoly_i
void print_cdpoly(CDPoly);
void add_cdpoly(CDPoly, CDPoly, CDPoly);
void sub_cdpoly(CDPoly, CDPoly, CDPoly);
void add2_cdpoly(CDPoly, CDPoly);
void sub2_cdpoly(CDPoly, CDPoly);
void mul_cdpoly(CDPoly, CDPoly, CDPoly);
void mul2_cdpoly(CDPoly, CDPoly);
void cmul_cdpoly(CDPoly, double _Complex, CDPoly);
void subst_cdpoly(CDPoly, CDPoly);
void set0_cdpoly(CDPoly);
long int max_abscoef_cdpoly(CDPoly);
long int num_nonzero_cdpoly(CDPoly);
void diff_cdpoly(CDPoly);
double _Complex eval_cdpoly_horner(CDPoly a, double _Complex x);
double _Complex eval_cdpoly_estrin(CDPoly a, double _Complex x);
#define eval_cdpoly eval_cdpoly_horner
double _Complex eval_diff_cdpoly(CDPoly a, double _Complex x);
void div_cdpoly(CDPoly, CDPoly, CDPoly, CDPoly);
void xpow_mul_cdpoly(CDPoly, long int, CDPoly);
double _Complex _bncavx2_eval_cdpoly_estrin(CDPoly a, double _Complex x);

// DD
DDPoly init_ddpoly(long int);
void free_ddpoly(DDPoly);
double *get_ddpoly_i(DDPoly, long int);
long int setdegree_ddpoly(DDPoly);
void set_ddpoly_i(DDPoly, long int, double *);
void set_ddpoly_i_si(DDPoly pol, long int index, long val);
void set_ddpoly_i_ui(DDPoly pol, long int index, unsigned long val);
void set_ddpoly_i_d(DDPoly pol, long int index, double val);
void set_ddpoly_i_str(DDPoly pol, long int index, const char *str, int base);
#define gddpi get_ddpoly_i
#define sddpi set_ddpoly_i
void print_ddpoly(DDPoly);
void add_ddpoly(DDPoly, DDPoly, DDPoly);
void sub_ddpoly(DDPoly, DDPoly, DDPoly);
void mul_ddpoly(DDPoly, DDPoly, DDPoly);
void cmul_ddpoly(DDPoly, double *, DDPoly);
void subst_ddpoly(DDPoly, DDPoly);
void set0_ddpoly(DDPoly);
long int max_abscoef_ddpoly(DDPoly);
long int num_nonzero_ddpoly(DDPoly);
void diff_ddpoly(DDPoly);
void eval_ddpoly_horner(double *, DDPoly, double *);
void eval_ddpoly_estrin(double *, DDPoly, double *);
#define eval_ddpoly eval_ddpoly_horner
void eval_diff_ddpoly(double *, DDPoly, double *);
void ceval_ddpoly_horner(cddfloat *, DDPoly, cddfloat *);
void ceval_ddpoly_estrin(cddfloat *, DDPoly, cddfloat *);
#define ceval_ddpoly ceval_ddpoly_horner
void ceval_diff_ddpoly(cddfloat *, DDPoly, cddfloat *);
// AVX2
#include "ddlinear.h"
#include "cddlinear.h"
void _bncavx2_eval_ddpoly_estrin(double ret[DDSIZE], DDPoly a, double x[DDSIZE]);
void _bncavx2_ceval_ddpoly_estrin(cddfloat *ret, DDPoly a, cddfloat *x);
//void _bncavx2_eval_cddpoly_estrin(cddfloat *ret, CDDPoly a, cddfloat *x);

// CDD
CDDPoly init_cddpoly(long int);
void free_cddpoly(CDDPoly);
CDDPoly init_set_cddpoly(CDDPoly);
CDDPoly init_set_cddpoly_ddpoly(DDPoly);
cddfloat* get_cddpoly_i(CDDPoly, long int);
long int setdegree_cddpoly(CDDPoly);
void set_cddpoly_i(CDDPoly, long int, cddfloat *);
#define gcddpi get_cddpoly_i
#define scddpi set_cddpoly_i
void print_cddpoly(CDDPoly);
void add_cddpoly(CDDPoly, CDDPoly, CDDPoly);
void sub_cddpoly(CDDPoly, CDDPoly, CDDPoly);
void mul_cddpoly(CDDPoly, CDDPoly, CDDPoly);
void cmul_ddpoly(DDPoly, double *, DDPoly);
void subst_cddpoly(CDDPoly, CDDPoly);
void set0_cddpoly(CDDPoly);
long int max_abscoef_cddpoly(CDDPoly);
long int num_nonzero_cddpoly(CDDPoly);
void diff_cddpoly(CDDPoly);
void eval_cddpoly_horner(cddfloat *, CDDPoly, cddfloat *);
void eval_cddpoly_estrin(cddfloat *, CDDPoly, cddfloat *);
#define eval_cddpoly eval_cddpoly_horner
void eval_diff_cddpoly(cddfloat *, CDDPoly, cddfloat *);
void _bncavx2_eval_cddpoly_estrin(cddfloat *ret, CDDPoly a, cddfloat *x);

// TD
TDPoly init_tdpoly(long int max_length);
void free_tdpoly(TDPoly pol);
tdfloat get_tdpoly_i_float(TDPoly pol, long int index);
double *get_tdpoly_i(TDPoly pol, long int index);
long int setdegree_tdpoly(TDPoly pol);
void set_tdpoly_i(TDPoly pol, long int index, double val[TDSIZE]);
void set_tdpoly_i_si(TDPoly pol, long int index, long val);
void set_tdpoly_i_ui(TDPoly pol, long int index, unsigned long val);
void set_tdpoly_i_d(TDPoly pol, long int index, double val);
void set_tdpoly_i_str(TDPoly pol, long int index, const char *str, int base);
#define gtdpi get_tdpoly_i
#define stdpi set_tdpoly_i
void print_tdpoly(TDPoly pol);
/* c = a + b */
void add_tdpoly(TDPoly c, TDPoly a, TDPoly b);
/* c += a */
void add2_tdpoly(TDPoly c, TDPoly a);
/* c = a - b */
void sub_tdpoly(TDPoly c, TDPoly a, TDPoly b);
/* c -= a */
void sub2_tdpoly(TDPoly c, TDPoly a);
/* c = a * b */
void mul_tdpoly(TDPoly c, TDPoly a, TDPoly b);
/* c = val * a */
void cmul_tdpoly(TDPoly c, double val[TDSIZE], TDPoly a);
/* c *= val */
void cmul2_tdpoly(TDPoly c, double val[TDSIZE]);
/* c := a */
void subst_tdpoly(TDPoly c, TDPoly a);
/* c := 0 */
void set0_tdpoly(TDPoly c);
/* number of nonzero coef */
long int num_nonzero_tdpoly(TDPoly c);
/* get maximum |coef| */
long int max_abscoef_tdpoly(TDPoly c);
/* a := a'(x) */
void diff_tdpoly(TDPoly a);
/* value of a(x) */
// Based on Horner method
void eval_tdpoly_horner(double ret[TDSIZE], TDPoly a, double x[TDSIZE]);
/* value of a(x) */
// Based on Estrin method
void eval_tdpoly_estrin(double ret[TDSIZE], TDPoly a, double x[TDSIZE]);
#define eval_tdpoly eval_tdpoly_horner
/* value of a'(x) */
// Based on Horner method
void eval_diff_tdpoly(double ret[TDSIZE], TDPoly a, double x[TDSIZE]);
/* complex value of a(x) */
// Based on Horner method
void ceval_tdpoly_horner(ctdfloat *ret, TDPoly a, ctdfloat *x);
/* value of a(x) */
// Based on Estrin method
void ceval_tdpoly_estrin(ctdfloat *ret, TDPoly a, ctdfloat *x);
#define ceval_tdpoly ceval_tdpoly_horner
/* value of a'(x) */
// Based on Horner method
void ceval_diff_tdpoly(ctdfloat *ret, TDPoly a, ctdfloat *x);
// AVX2
#include "tdlinear.h"
#include "ctdlinear.h"
void _bncavx2_eval_tdpoly_estrin(double ret[TDSIZE], TDPoly a, double x[TDSIZE]);
void _bncavx2_ceval_tdpoly_estrin(ctdfloat *ret, TDPoly a, ctdfloat *x);

// CTD
// Initialize
CTDPoly init_ctdpoly(long int max_length);

// Free
void free_ctdpoly(CTDPoly pol);

// Get & Set Values
ctdfloat *get_ctdpoly_i(CTDPoly pol, long int index);
long int setdegree_ctdpoly(CTDPoly pol);
void set_ctdpoly_i(CTDPoly pol, long int index, ctdfloat *val);

#ifdef USE_GMP
void set_ctdpoly_i_mpc(CTDPoly pol, long int index, mpc_t val);
CTDPoly init_set_ctdpoly_cmpfpoly(CMPFPoly org_pol);
#endif // USE_GMP

// Initialize and substitute
CTDPoly init_set_ctdpoly(CTDPoly org_pol);
void subst_ctdpoly(CTDPoly pol, CTDPoly org_pol);

// Set all coefficients to zero
void set0_ctdpoly(CTDPoly pol);

// Output
void print_ctdpoly(CTDPoly pol);
//void fprint_ctdpoly(FILE *fp, CTDPoly pol);

// Evaluation
void eval_ctdpoly(ctdfloat *ret, CTDPoly pol, ctdfloat *x);

// Polynomial operations
void add_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2);
void sub_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2);
void scal_ctdpoly(CTDPoly ret, ctdfloat *alpha, CTDPoly pol);
void mul_ctdpoly(CTDPoly ret, CTDPoly pol1, CTDPoly pol2);
void div_ctdpoly(CTDPoly quot, CTDPoly rem, CTDPoly pol1, CTDPoly pol2);
void diff_ctdpoly(CTDPoly ret, CTDPoly pol);
void eval_ctdpoly_horner(ctdfloat *ret, CTDPoly a, ctdfloat *x);
void eval_diff_ctdpoly(ctdfloat *ret, CTDPoly a, ctdfloat *x);
void eval_ctdpoly_estrin(ctdfloat *ret, CTDPoly a, ctdfloat *x);
void _bncavx2_eval_ctdpoly_estrin(ctdfloat *ret, CTDPoly a, ctdfloat *x);

// QD
QDPoly init_qdpoly(long int max_length);
void free_qdpoly(QDPoly pol);
qdfloat get_qdpoly_i_float(QDPoly pol, long int index);
double *get_qdpoly_i(QDPoly pol, long int index);
long int setdegree_qdpoly(QDPoly pol);
void set_qdpoly_i(QDPoly pol, long int index, double val[QDSIZE]);
void set_qdpoly_i_si(QDPoly pol, long int index, long val);
void set_qdpoly_i_ui(QDPoly pol, long int index, unsigned long val);
void set_qdpoly_i_d(QDPoly pol, long int index, double val);
void set_qdpoly_i_str(QDPoly pol, long int index, const char *str, int base);
#define gqdpi get_qdpoly_i
#define sqdpi set_qdpoly_i
void print_qdpoly(QDPoly pol);
/* c = a + b */
void add_qdpoly(QDPoly c, QDPoly a, QDPoly b);
/* c += a */
void add2_qdpoly(QDPoly c, QDPoly a);
/* c = a - b */
void sub_qdpoly(QDPoly c, QDPoly a, QDPoly b);
/* c -= a */
void sub2_qdpoly(QDPoly c, QDPoly a);
/* c = a * b */
void mul_qdpoly(QDPoly c, QDPoly a, QDPoly b);
/* c = val * a */
void cmul_qdpoly(QDPoly c, double val[QDSIZE], QDPoly a);
/* c *= val */
void cmul2_qdpoly(QDPoly c, double val[QDSIZE]);
/* c := a */
void subst_qdpoly(QDPoly c, QDPoly a);
/* c := 0 */
void set0_qdpoly(QDPoly c);
/* number of nonzero coef */
long int num_nonzero_qdpoly(QDPoly c);
/* get maximum |coef| */
long int max_abscoef_qdpoly(QDPoly c);
/* a := a'(x) */
void diff_qdpoly(QDPoly a);
/* value of a(x) */
// Based on Horner method
void eval_qdpoly_horner(double ret[QDSIZE], QDPoly a, double x[QDSIZE]);
/* value of a(x) */
// Based on Estrin method
void eval_qdpoly_estrin(double ret[QDSIZE], QDPoly a, double x[QDSIZE]);
#define eval_qdpoly eval_qdpoly_horner
/* value of a'(x) */
// Based on Horner method
void eval_diff_qdpoly(double ret[QDSIZE], QDPoly a, double x[QDSIZE]);
/* complex value of a(x) */
// Based on Horner method
void ceval_qdpoly_horner(cqdfloat *ret, QDPoly a, cqdfloat *x);
/* value of a(x) */
// Based on Estrin method
void ceval_qdpoly_estrin(cqdfloat *ret, QDPoly a, cqdfloat *x);
#define ceval_qdpoly ceval_qdpoly_horner
/* value of a'(x) */
// Based on Horner method
void ceval_diff_qdpoly(cqdfloat *ret, QDPoly a, cqdfloat *x);

// CQD

// Initialize
CQDPoly init_cqdpoly(long int max_length);

// Free
void free_cqdpoly(CQDPoly pol);

// Get & Set Values
cqdfloat *get_cqdpoly_i(CQDPoly pol, long int index);
long int setdegree_cqdpoly(CQDPoly pol);
void set_cqdpoly_i(CQDPoly pol, long int index, cqdfloat *val);

#ifdef USE_GMP
void set_cqdpoly_i_mpc(CQDPoly pol, long int index, mpc_t val);
CQDPoly init_set_cqdpoly_cmpfpoly(CMPFPoly org_pol);
#endif // USE_GMP

// Initialize and substitute
CQDPoly init_set_cqdpoly(CQDPoly org_pol);
void subst_cqdpoly(CQDPoly pol, CQDPoly org_pol);

// Set all coefficients to zero
void set0_cqdpoly(CQDPoly pol);

// Output
void print_cqdpoly(CQDPoly pol);
//void fprint_cqdpoly(FILE *fp, CQDPoly pol);

// Evaluation
void eval_cqdpoly(cqdfloat *ret, CQDPoly pol, cqdfloat *x);

// Polynomial operations
void add_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2);
void sub_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2);
void scal_cqdpoly(CQDPoly ret, cqdfloat *alpha, CQDPoly pol);
void mul_cqdpoly(CQDPoly ret, CQDPoly pol1, CQDPoly pol2);
void div_cqdpoly(CQDPoly quot, CQDPoly rem, CQDPoly pol1, CQDPoly pol2);
void diff_cqdpoly(CQDPoly ret, CQDPoly pol);

void eval_cqdpoly_horner(cqdfloat *ret, CQDPoly a, cqdfloat *x);
void eval_cqdpoly_estrin(cqdfloat *ret, CQDPoly a, cqdfloat *x);
void eval_diff_cqdpoly(cqdfloat *ret, CQDPoly a, cqdfloat *x);

#ifdef USE_GMP
// Set coefs from mpf_t val
void set_ddpoly_i_mpf(DDPoly pol, long int index, mpf_t val);
void set_tdpoly_i_mpf(TDPoly pol, long int index, mpf_t val);
void set_qdpoly_i_mpf(QDPoly pol, long int index, mpf_t val);
void set_cddpoly_i_mpc(CDDPoly pol, long int index, mpc_t val);
void set_ctdpoly_i_mpc(CTDPoly pol, long int index, mpc_t val);
void set_cqdpoly_i_mpc(CQDPoly pol, long int index, mpc_t val);

// Initialize and substitute polynomial from org_pol
DDPoly init_set_ddpoly_mpfpoly(MPFPoly org_pol);
TDPoly init_set_tdpoly_mpfpoly(MPFPoly org_pol);
QDPoly init_set_qdpoly_mpfpoly(MPFPoly org_pol);
CDDPoly init_set_cddpoly_cmpfpoly(CMPFPoly org_pol);
CTDPoly init_set_ctdpoly_cmpfpoly(CMPFPoly org_pol);
CQDPoly init_set_cqdpoly_cmpfpoly(CMPFPoly org_pol);
#endif // USE_GMP

// AVX2
#include "qdlinear.h"
#include "cqdlinear.h"
void _bncavx2_eval_qdpoly_estrin(double ret[QDSIZE], QDPoly a, double x[QDSIZE]);
void _bncavx2_ceval_qdpoly_estrin(cqdfloat *ret, QDPoly a, cqdfloat *x);
void _bncavx2_eval_cqdpoly_estrin(cqdfloat *ret, CQDPoly a, cqdfloat *x);
//#ifdef USE_QDMATRIX
// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
void ceval_hes_qdmatrix(cqdfloat *ret, QDMatrix hes, cqdfloat *x);
//#endif // USE_QDMATRIX

// GMP and MPFR
#ifdef USE_GMP
MPFPoly init_mpfpoly(long int);
MPFPoly init2_mpfpoly(long int, unsigned long);
// init_set_mpfpoly
MPFPoly init_set_mpfpoly(MPFPoly org_pol);
void free_mpfpoly(MPFPoly);
mpf_ptr get_mpfpoly_i(MPFPoly, long int);
void set_mpfpoly_i(MPFPoly, long int, mpf_t);
void set_mpfpoly_i_d(MPFPoly, long int, double);
void set_mpfpoly_i_si(MPFPoly, long int, long);
void set_mpfpoly_i_ui(MPFPoly, long int, unsigned long);
void set_mpfpoly_i_str(MPFPoly, long int, const char *, int);
#define gmpfpi get_mpfpoly_i
#define smpfpi set_mpfpoly_i
void print_mpfpoly(MPFPoly);
void print_fdmpfpoly(FPoly, DPoly, MPFPoly);
void add_mpfpoly(MPFPoly, MPFPoly, MPFPoly);
void sub_mpfpoly(MPFPoly, MPFPoly, MPFPoly);
void add2_mpfpoly(MPFPoly, MPFPoly);
void sub2_mpfpoly(MPFPoly, MPFPoly);
void mul_mpfpoly(MPFPoly, MPFPoly, MPFPoly);
void cmul_mpfpoly(MPFPoly, mpf_t, MPFPoly);
void subst_mpfpoly(MPFPoly, MPFPoly);
void set0_mpfpoly(MPFPoly);
long int max_abscoef_mpfpoly(MPFPoly);
long int num_nonzero_mpfpoly(MPFPoly);
void diff_mpfpoly(MPFPoly);
void eval_mpfpoly(mpf_t, MPFPoly, mpf_t);
void eval_mpfpoly_horner(mpf_t, MPFPoly, mpf_t);
void eval_mpfpoly_estrin(mpf_t, MPFPoly, mpf_t);
//#define eval_mpfpoly eval_mpfpoly_estrin
#define eval_mpfpoly eval_mpfpoly_horner
void eval_diff_mpfpoly(mpf_t, MPFPoly, mpf_t);
// Old implementation
void _bncold_ceval_mpfpoly(MPFCmplx, MPFPoly, MPFCmplx);
void _bncold_ceval_diff_mpfpoly(MPFCmplx, MPFPoly, MPFCmplx);
// New implementation
void ceval_mpfpoly_horner(mpc_t, MPFPoly, mpc_t);
void ceval_mpfpoly_estrin(mpc_t, MPFPoly, mpc_t);
//#define ceval_mpfpoly ceval_mpfpoly_estrin
#define ceval_mpfpoly ceval_mpfpoly_horner
void ceval_diff_mpfpoly(mpc_t, MPFPoly, mpc_t);

// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
void ceval_hes_mpfmatrix(mpc_t ret, MPFMatrix hes, mpc_t x);

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_mpfpoly(MPFPoly pol);

#endif // USE_GMP

// MPC: complex
#ifdef USE_GMP
CMPFPoly init_cmpfpoly(long int);
CMPFPoly init2_cmpfpoly(long int, unsigned long);
void free_cmpfpoly(CMPFPoly);
mpc_ptr get_cmpfpoly_i(CMPFPoly, long int);
void set_cmpfpoly_i(CMPFPoly, long int, mpc_t);
void set_cmpfpoly_i_cd(CMPFPoly, long int, double _Complex);
void set_cmpfpoly_i_d(CMPFPoly, long int, double);
void set_cmpfpoly_i_si_si(CMPFPoly, long int, long, long);
void set_cmpfpoly_i_si(CMPFPoly, long int, long);
void set_cmpfpoly_i_ui(CMPFPoly, long int, unsigned long);
void set_cmpfpoly_i_ui_ui(CMPFPoly, long int, unsigned long, unsigned long);
void set_cmpfpoly_i_str(CMPFPoly, long int, const char *, int);
#define gcmpfpi get_cmpfpoly_i
#define scmpfpi set_cmpfpoly_i
void print_cmpfpoly(CMPFPoly);
void subst_cmpfpoly_mpfpoly(CMPFPoly ret, MPFPoly pol);
CMPFPoly init_set_cmpfpoly_mpfpoly(MPFPoly org);
CMPFPoly init_set_cmpfpoly(CMPFPoly org);
void add_cmpfpoly(CMPFPoly, CMPFPoly, CMPFPoly);
void sub_cmpfpoly(CMPFPoly, CMPFPoly, CMPFPoly);
void add2_cmpfpoly(CMPFPoly, CMPFPoly);
void sub2_cmpfpoly(CMPFPoly, CMPFPoly);
void mul_cmpfpoly(CMPFPoly, CMPFPoly, CMPFPoly);
void cmul_cmpfpoly(CMPFPoly, mpc_t, CMPFPoly);
void subst_cmpfpoly(CMPFPoly, CMPFPoly);
void set0_cmpfpoly(CMPFPoly);
long int max_abscoef_cmpfpoly(CMPFPoly);
long int num_nonzero_cmpfpoly(CMPFPoly);
void diff_cmpfpoly(CMPFPoly);
void eval_cmpfpoly(mpc_t, CMPFPoly, mpc_t);
void eval_cmpfpoly_horner(mpc_t, CMPFPoly, mpc_t);
void eval_cmpfpoly_estrin(mpc_t, CMPFPoly, mpc_t);
//#define eval_mpfpoly eval_mpfpoly_estrin
#define eval_cmpfpoly eval_cmpfpoly_horner
void eval_diff_cmpfpoly(mpc_t, CMPFPoly, mpc_t);
// New implementation
void ceval_cmpfpoly_horner(mpc_t, CMPFPoly, mpc_t);
void ceval_cmpfpoly_estrin(mpc_t, CMPFPoly, mpc_t);
//#define ceval_mpfpoly ceval_mpfpoly_estrin
#define ceval_mpfpoly ceval_mpfpoly_horner
void ceval_diff_cmpfpoly(mpc_t, CMPFPoly, mpc_t);

// eval based on Hessenberg matrix
// mphes: Upper Hessenberg matrix
void ceval_hes_cmpfmatrix(mpc_t ret, CMPFMatrix hes, mpc_t x);

/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cmpfpoly(CMPFPoly pol);

#endif // USE_GMP

/*************************************************/
/* array.c: array                                */
/*************************************************/
FArray init_farray(long int);
void free_farray(FArray);
float get_farray_i(FArray, long int);
void set_farray_i(FArray, long int, float);
void subst_farray(FArray, FArray);
void print_farray(FArray);
#define gfai get_farray_i
#define sfai set_farray_i

CFArray init_cfarray(long int);
void free_cfarray(CFArray);
FCmplx get_cfarray_i(CFArray, long int);
//float _Complex *get_cfarray_i(CFArray, long int);
void set_cfarray_i(CFArray, long int, FCmplx);
//void set_cfarray_i(CFArray, long int, float _Complex *);
void subst_cfarray(CFArray, CFArray);
void print_cfarray(CFArray);
#define gcfai get_cfarray_i
#define scfai set_cfarray_i

DArray init_darray(long int);
void free_darray(DArray);
double get_darray_i(DArray, long int);
void set_darray_i(DArray, long int, double);
void subst_darray(DArray, DArray);
void print_darray(DArray);
#define gdai get_darray_i
#define sdai set_darray_i

CDArray init_cdarray(long int);
void free_cdarray(CDArray);
DCmplx get_cdarray_i(CDArray, long int);
void set_cdarray_i(CDArray, long int, DCmplx);
//double _Complex *get_cdarray_i(CDArray, long int);
//void set_cdarray_i(CDArray, long int, double _Complex *);
void subst_cdarray(CDArray, CDArray);
void print_cdarray(CDArray);
#define gcdai get_cdarray_i
#define scdai set_cdarray_i

#ifdef USE_GMP
MPFArray init_mpfarray(long int);
MPFArray init2_mpfarray(long int, unsigned long);
void free_mpfarray(MPFArray);
mpf_ptr get_mpfarray_i(MPFArray, long int);
void set_mpfarray_i(MPFArray, long int, mpf_t);
void set_mpfarray_i_d(MPFArray, long int, double);
void set_mpfarray_i_ui(MPFArray, long int, unsigned long);
void subst_mpfarray(MPFArray, MPFArray);
void print_mpfarray(MPFArray);
#define gmpfai get_mpfarray_i
#define smpfai set_mpfarray_i

// Old implementation
_bncold_CMPFArray _bncold_init_cmpfarray(long int);
_bncold_CMPFArray _bncold_init2_cmpfarray(long int, unsigned long);
void _bncold_free_cmpfarray(_bncold_CMPFArray);
MPFCmplx _bncold_get_cmpfarray_i(_bncold_CMPFArray, long int);
//mpc_ptr get_cmpfarray_i(CMPFArray, long int);
void _bncold_set_cmpfarray_i(_bncold_CMPFArray, long int, MPFCmplx);
//void set_cmpfarray_i(CMPFArray, long int, mpc_t);
void _bncold_subst_cmpfarray(_bncold_CMPFArray, _bncold_CMPFArray);
void _bncold_print_cmpfarray(_bncold_CMPFArray);

// New implementation
CMPFArray init_cmpfarray(long int);
CMPFArray init2_cmpfarray(long int, unsigned long);
void free_cmpfarray(CMPFArray);
//MPFCmplx get_cmpfarray_i(CMPFArray, long int);
mpc_ptr get_cmpfarray_i(CMPFArray, long int);
//void set_cmpfarray_i(CMPFArray, long int, MPFCmplx);
void set_cmpfarray_i(CMPFArray, long int, mpc_t);
void set_cmpfarray_i_d(CMPFArray, long int, double);
void set_cmpfarray_i_ui(CMPFArray, long int, unsigned long);
void set_cmpfarray_i_real(CMPFArray, long int, mpf_t);
void subst_cmpfarray(CMPFArray, CMPFArray);
void print_cmpfarray(CMPFArray);
#define gcmpfai get_cmpfarray_i
#define scmpfai set_cmpfarray_i
#endif // USE_GMP

/*************************************************/
/* dka.c: DKA Methods                            */
/*************************************************/
float fdka_center(FPoly);
float fdka_radius(FPoly);
void fdka_init(CFArray, FPoly);
long int fdka(CFArray, CFArray, FPoly, long int, float, float);

double ddka_center(DPoly);
double _Complex cddka_center(CDPoly);
double ddka_radius(DPoly);
double cddka_radius(CDPoly);
void ddka_init(CDArray, DPoly);
void cddka_init(double _Complex[], CDPoly);
void cddka_init_cdarray(CDArray, CDPoly);
long int ddka(CDArray, CDArray, DPoly, long int, double, double);
long int cddka(double _Complex[], double _Complex[], CDPoly, long int, double, double);
/*************************************************/
/* dd_dka.c: DKA Methods                         */
/*************************************************/
void dd_dka_center(double ret[DDSIZE], DDPoly func);
void dd_dka_radius(double ret[DDSIZE], DDPoly func);
// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void dd_dka_ozawa_radius(double ret[DDSIZE], DDPoly func);
void dd_dka_init(CDDVector, DDPoly);
long int dd_dka(CDDVector, CDDVector, DDPoly, long int, double *, double*);
void dd_dka_init2(CDDVector x_init, DDPoly func, void (* get_radius)(double *, DDPoly), void (* get_center)(double *, DDPoly));
long int dd_dka_mod(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE]);
long int dd_petckovic(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE]);
long int dd_aberth(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE]);
// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void dd_dka_ozawa_radius(double ret[DDSIZE], DDPoly func);
/*************************************************/
/* td_dka.c: DKA Methods                         */
/*************************************************/
void td_dka_center(double ret[TDSIZE], TDPoly func);
void td_dka_radius(double ret[TDSIZE], TDPoly func);
// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void td_dka_ozawa_radius(double ret[TDSIZE], TDPoly func);
void td_dka_init(CTDVector, TDPoly);
long int td_dka(CTDVector, CTDVector, TDPoly, long int, double *, double*);
void td_dka_init2(CTDVector x_init, TDPoly func, void (* get_radius)(double *, TDPoly), void (* get_center)(double *, TDPoly));
long int td_dka_mod(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE]);
long int td_petckovic(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE]);
long int td_aberth(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE]);

/*************************************************/
/* qd_dka.c: DKA Methods                         */
/*************************************************/
void qd_dka_center(double ret[QDSIZE], QDPoly func);
void qd_dka_radius(double ret[QDSIZE], QDPoly func);
// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void qd_dka_ozawa_radius(double ret[QDSIZE], QDPoly func);
void qd_dka_init(CQDVector, QDPoly);
long int qd_dka(CQDVector, CQDVector, QDPoly, long int, double *, double*);
void qd_dka_init2(CQDVector x_init, QDPoly func, void (* get_radius)(double *, QDPoly), void (* get_center)(double *, QDPoly));
long int qd_dka_mod(CQDVector, CQDVector, QDPoly, long int, double *, double*);
long int qd_petckovic(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE]);
long int qd_aberth(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE]);
/*************************************************/
/* mpf_dka.c: DKA Methods                        */
/*************************************************/
#ifdef USE_GMP
void mpf_dka_center(mpf_t, MPFPoly);
void mpf_dka_radius(mpf_t, MPFPoly);

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void mpf_dka_ozawa_radius(mpf_t ret, MPFPoly func);

// Old implementation
void _bncold_mpf_dka_init(_bncold_CMPFArray, MPFPoly);
long int _bncold_mpf_dka(_bncold_CMPFArray, _bncold_CMPFArray, MPFPoly, long int, mpf_t, mpf_t);
/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _bncold_mpf_dka_init2(_bncold_CMPFArray x_init, MPFPoly func, void (* get_radius)(mpf_t, MPFPoly), void (* get_center)(mpf_t, MPFPoly));

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncold_mpf_dka_mod(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);
/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncold_mpf_petckovic(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncold_mpf_aberth(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

// New implementation
void mpf_dka_init(CMPFArray, MPFPoly);
long int mpf_dka(CMPFArray, CMPFArray, MPFPoly, long int, mpf_t, mpf_t);

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpf_dka_init2(CMPFArray x_init, MPFPoly func, void (* get_radius)(mpf_t, MPFPoly), void (* get_center)(mpf_t, MPFPoly));

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_dka_mod(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_petckovic(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_aberth(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

#endif // USE_GMP

/*************************************************/
/* mpc_dka.c: DKA Methods                        */
/*************************************************/
#ifdef USE_GMP
void mpc_dka_center(mpc_t, CMPFPoly);
void mpc_dka_radius(mpf_t, CMPFPoly);

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void mpc_dka_ozawa_radius(mpf_t ret, CMPFPoly func);

// New implementation
void mpc_dka_init(CMPFArray, CMPFPoly);
long int mpc_dka(CMPFArray, CMPFArray, CMPFPoly, long int, mpf_t, mpf_t);

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpc_dka_init2(CMPFArray x_init, CMPFPoly func, void (* get_radius)(mpf_t, CMPFPoly), void (* get_center)(mpc_t, CMPFPoly));

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_dka_mod(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_petckovic(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_aberth(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

#endif // USE_GMP

/*************************************************/
/* hirano.c                                      */
/*************************************************/
// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
double _Complex dhorner(double _Complex x, DPoly pol);
double _Complex cdhorner(double _Complex x, CDPoly pol);
// Coef of p(x + d)
// 
// based on Horner method
void dcoef_horner(double _Complex ret_coef[], double _Complex x, DPoly pol);
void cdcoef_horner(double _Complex ret_coef[], double _Complex x, CDPoly pol);
// return j and max|a_j x^j| from p(x)
long int absmax_dpoly(double *absmax_anxn, double _Complex x, DPoly pol);
long int absmax_cdpoly(double *absmax_anxn, double _Complex x, CDPoly pol);
// get_plus_arg
// return arg(x) in [0, 2 PI]
double dget_plus_arg(double _Complex x);
// get_nearest_int
long int dget_nearest_int(double real_x);
// get_min_branch
double _Complex dget_min_branch(double _Complex x, double mu, double _Complex coef[], long int i_num, long int i_den);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
//long int dhirano(double _Complex *ret, double _Complex init_x, double coef[], long int deg, double reps, double aeps, long int maxtimes)
long int dhirano(double _Complex *ret, double _Complex init_x, DPoly pol, double reps, double aeps, long int maxtimes);
long int cdhirano(double _Complex *ret, double _Complex init_x, CDPoly pol, double reps, double aeps, long int maxtimes);

#ifdef USE_GMP

// Coef of p(x + d)
// 
// based on Horner method
void mpf_coef_horner(CMPFArray ret_coef, mpc_t x, MPFPoly poly);
void mpc_coef_horner(CMPFArray ret_coef, mpc_t x, CMPFPoly poly);
// return j and max|a_j x^j| from p(x)
long int absmax_mpfpoly(mpf_t absmax_anxn, mpc_t x, MPFPoly poly);
long int absmax_cmpfpoly(mpf_t absmax_anxn, mpc_t x, CMPFPoly poly);
// get_plus_arg
// return arg(x) in [0, 2 PI]
void mpf_get_plus_arg(mpf_t ret, mpc_t x);
// get_nearest_int
void mpf_get_nearest_int(mpf_t ret, mpf_t real_x);
// get_min_branch
void mpf_get_min_branch(mpc_t ret, mpc_t x, mpf_t mu, CMPFArray coef, long int i_num, long int i_den);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int mpf_hirano(mpc_t ret, mpc_t init_x, MPFPoly poly, mpf_t reps, mpf_t aeps, long int maxtimes);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int mpc_hirano(mpc_t ret, mpc_t init_x, CMPFPoly poly, mpf_t reps, mpf_t aeps, long int maxtimes);
// Ono's Problem
void ono_poly(MPFPoly poly, long int deg);
// Wilkinson's example: (x - 1)(x - 2) ... (x - n) = 0
void wilkinson_poly(MPFPoly ret, long int n);
// Deflation of polynomial
// p(x) / (x - r)
void deflation_cmpfpoly(CMPFPoly ret, CMPFPoly pol, mpc_t root);
#endif // USE_GMP

/*************************************************/
/* dd_hirano.c                                   */
/*************************************************/

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void dd_horner(cddfloat *ret, cddfloat *x, DDPoly poly);
void dd_coef_horner(cddfloat ret_coef[], cddfloat *x, DDPoly poly);
void cdd_horner(cddfloat *ret, cddfloat *x, CDDPoly poly);
void cdd_coef_horner(cddfloat ret_coef[], cddfloat *x, CDDPoly poly);
long int absmax_ddpoly(double absmax_anxn[DDSIZE], cddfloat *x, DDPoly poly);
long int absmax_cddpoly(double absmax_anxn[DDSIZE], cddfloat *x, CDDPoly poly);
void dd_get_plus_arg(double ret[DDSIZE], cddfloat *x);
void dd_get_nearest_int(double ret[DDSIZE], double real_x[DDSIZE]);
void dd_get_min_branch(cddfloat *ret, cddfloat *x, double mu[DDSIZE], cddfloat coef[], long int i_num, long int i_den);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int dd_hirano(cddfloat *ret, cddfloat *init_x, DDPoly poly, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cdd_hirano(cddfloat *ret, cddfloat *init_x, CDDPoly poly, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Deflation of polynomial
// p(x) / (x - r)
void deflation_cddpoly(CDDPoly ret, CDDPoly pol, cddfloat *root);

/*************************************************/
/* td_hirano.h                                   */
/*************************************************/

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void td_horner(ctdfloat *ret, ctdfloat *x, TDPoly poly);
void td_coef_horner(ctdfloat ret_coef[], ctdfloat *x, TDPoly poly);
void ctd_horner(ctdfloat *ret, ctdfloat *x, CTDPoly poly);
void ctd_coef_horner(ctdfloat ret_coef[], ctdfloat *x, CTDPoly poly);
long int absmax_tdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, TDPoly poly);
long int absmax_ctdpoly(double absmax_anxn[TDSIZE], ctdfloat *x, CTDPoly poly);
void td_get_plus_arg(double ret[TDSIZE], ctdfloat *x);
void td_get_nearest_int(double ret[TDSIZE], double real_x[TDSIZE]);
void td_get_min_branch(ctdfloat *ret, ctdfloat *x, double mu[TDSIZE], ctdfloat coef[], long int i_num, long int i_den);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int td_hirano(ctdfloat *ret, ctdfloat *init_x, TDPoly poly, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int ctd_hirano(ctdfloat *ret, ctdfloat *init_x, CTDPoly poly, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Deflation of polynomial
// p(x) / (x - r)
void deflation_ctdpoly(CTDPoly ret, CTDPoly pol, ctdfloat *root);

/*************************************************/
/* qd_hirano.h                                   */
/*************************************************/

// Horner method
// return p(x)
// coef[0] + coef[1] * x + ... + coef[deg] * x^deg
void qd_horner(cqdfloat *ret, cqdfloat *x, QDPoly poly);
void qd_coef_horner(cqdfloat ret_coef[], cqdfloat *x, QDPoly poly);
void cqd_horner(cqdfloat *ret, cqdfloat *x, CQDPoly poly);
void cqd_coef_horner(cqdfloat ret_coef[], cqdfloat *x, CQDPoly poly);
long int absmax_qdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, QDPoly poly);
long int absmax_cqdpoly(double absmax_anxn[QDSIZE], cqdfloat *x, CQDPoly poly);
void qd_get_plus_arg(double ret[QDSIZE], cqdfloat *x);
void qd_get_nearest_int(double ret[QDSIZE], double real_x[QDSIZE]);
void qd_get_min_branch(cqdfloat *ret, cqdfloat *x, double mu[QDSIZE], cqdfloat coef[], long int i_num, long int i_den);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int qd_hirano(cqdfloat *ret, cqdfloat *init_x, QDPoly poly, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Hirano method
// |p(x)| < max|a_j x^j| * reps + aeps -> ret
long int cqd_hirano(cqdfloat *ret, cqdfloat *init_x, CQDPoly poly, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Deflation of polynomial
// p(x) / (x - r)
void deflation_cqdpoly(CQDPoly ret, CQDPoly pol, cqdfloat *root);

/*************************************************/
/* fread_write_complex.c                         */
/*************************************************/
#ifndef BNC_IO_MAX_DEC_DIGITS
#define BNC_IO_MAX_DEC_DIGITS 65536
#endif // BNC_IO_MAX_DEC_DIGITS

void fread_cmpfarray(FILE *fp, CMPFArray array);
void fread_cmpfarray_fname(const char *fname, CMPFArray array);
void fwrite_cmpfarray(FILE *fp, CMPFArray array);
void fwrite_cmpfarray_fname(const char *fname, CMPFArray array);
void fread_mpfpolycoef(FILE *fp, MPFPoly p, long int maxdeg);




#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // ifndef __POLY_H_
