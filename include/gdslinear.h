/********************************************************************************/
/* gdslinear.h: Double-double and Quadruple precision                           */
/*               Linear Computation Library with GQS and CUDA                   */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#ifndef __BNC_GDSLINEAR_H__
  #define __BNC_GDSLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>   /* pull in libstdc++ <iosfwd>/<string> at global scope BEFORE any
                         extern "C"-wrapped bncmatmul header drags in gmp.h's <iosfwd> */
#include <string>

// on CPU
#include "rds.h"
#include "dslinear.h"
#include "qslinear.h"   // QSVector / QSMatrix used by GQS prototypes below

// on GPU
#include "cuda_runtime.h" // cudaMemcpy
#include "gqd_type.h" // gdtq-0.0.2: single header declares gdd/gtd/gqd AND gds/gts/gqs types
#include "vector_types.h" // for GQS type
#ifdef __CUDACC__
	#ifndef __INCLUDE_GQS_CU__
		#include "gqs.cuh"
	#else
		#include "gqs.cu"
	#endif // __INCLUDE_GQS_CU__
#endif // __CUDACC__

// Constants
static gds_real _const_gds_zero = {0.0, 0.0}; // zero
static gds_real _const_gds_one = {1.0, 0.0}; // 1
static gqs_real _const_gqs_zero = {0.0, 0.0, 0.0, 0.0}; // zero
static gqs_real _const_gqs_one = {1.0, 0.0, 0.0, 0.0}; // 1

// convert gds on GPU to dd on CPU
//Ref: http://homepages.math.uic.edu/~jan/mcs572/quad_double_cuda.pdf
inline dsfloat gds_get_ds(gds_real gval)
{
	dsfloat ret;

	ret.val[0] = gval.x;
	ret.val[1] = gval.y;

	return ret;
}

// convert gds on GPU to qd on CPU
//Ref: http://homepages.math.uic.edu/~jan/mcs572/quad_double_cuda.pdf
inline qsfloat gqs_get_qs(gqs_real gval)
{
	qsfloat ret;

	ret.val[0] = gval.x;
	ret.val[1] = gval.y;
	ret.val[2] = gval.z;
	ret.val[3] = gval.w;

	return ret;
}
// dsval := gdsval
inline void gds2ds(dsfloat *dsval, gds_real *gdsval)
{
	dsval->val[0] = gdsval->x;
	dsval->val[1] = gdsval->y;
}

// qsval := gqsval
inline void gqs2qs(qsfloat *qsval, gqs_real *gqsval)
{
	qsval->val[0] = gqsval->x;
	qsval->val[1] = gqsval->y;
	qsval->val[2] = gqsval->z;
	qsval->val[3] = gqsval->w;
}

// gdsval := dsval
inline void ds2gds(gds_real *gdsval, dsfloat *dsval)
{
	gdsval->x = dsval->val[0];
	gdsval->y = dsval->val[1];
}

// gqsval := qsval
inline void qs2gqs(gqs_real *gqsval, qsfloat *qsval)
{
	gqsval->x = qsval->val[0];
	gqsval->y = qsval->val[1];
	gqsval->z = qsval->val[2];
	gqsval->w = qsval->val[3];
}

// gdsval on GPU := dsval
inline void ds2gds_dev(gds_real *gdsval_dev, dsfloat *dsval)
{
	cudaMemcpy((void *)&(gdsval_dev->x), (void *)&(dsval->val[0]), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gdsval_dev->y), (void *)&(dsval->val[1]), sizeof(float), cudaMemcpyHostToDevice);
}

// dsval := gdsval on GPU
inline void gds2ds_dev(dsfloat *dsval, gds_real *gdsval_dev)
{
	cudaMemcpy((void *)&(dsval->val[0]), (void *)&(gdsval_dev->x), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(dsval->val[1]), (void *)&(gdsval_dev->y), sizeof(float), cudaMemcpyDeviceToHost);
}

// gdsval := gdsval on GPU
inline void gds2gds_dev(gds_real *gdsval, gds_real *gdsval_dev)
{
	cudaMemcpy((void *)&(gdsval->x), (void *)&(gdsval_dev->x), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gdsval->y), (void *)&(gdsval_dev->y), sizeof(float), cudaMemcpyDeviceToHost);
}

// gdsval on GPU := gdsval on HOST
inline void gds_dev2gds(gds_real *gdsval_dev, gds_real *gdsval)
{
	cudaMemcpy((void *)&(gdsval_dev->x), (void *)&(gdsval->x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gdsval_dev->y), (void *)&(gdsval->y), sizeof(float), cudaMemcpyHostToDevice);
}

// gqsval on GPU := qsval
inline void qs2gqs_dev(gqs_real *gqsval_dev, qsfloat *qsval)
{
	cudaMemcpy((void *)&(gqsval_dev->x), (void *)&(qsval->val[0]), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->y), (void *)&(qsval->val[1]), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->z), (void *)&(qsval->val[2]), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->w), (void *)&(qsval->val[3]), sizeof(float), cudaMemcpyHostToDevice);
}

// qsval := gqsval on GPU
inline void gqs2qs_dev(qsfloat *qsval, gqs_real *gqsval_dev)
{
	cudaMemcpy((void *)&(qsval->val[0]), (void *)&(gqsval_dev->x), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qsval->val[1]), (void *)&(gqsval_dev->y), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qsval->val[2]), (void *)&(gqsval_dev->z), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qsval->val[3]), (void *)&(gqsval_dev->w), sizeof(float), cudaMemcpyDeviceToHost);
}

// gqsval on GPU := gqsval on HOST
inline void gqs_dev2gqs(gqs_real *gqsval_dev, gqs_real *gqsval)
{
	cudaMemcpy((void *)&(gqsval_dev->x), (void *)&(gqsval->x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->y), (void *)&(gqsval->y), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->z), (void *)&(gqsval->z), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->w), (void *)&(gqsval->w), sizeof(float), cudaMemcpyHostToDevice);
}

// gdsval := gdsval on GPU
inline void gqs2gqs_dev(gqs_real *gqsval, gqs_real *gqsval_dev)
{
	cudaMemcpy((void *)&(gqsval->x), (void *)&(gqsval_dev->x), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqsval->y), (void *)&(gqsval_dev->y), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqsval->z), (void *)&(gqsval_dev->z), sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqsval->w), (void *)&(gqsval_dev->w), sizeof(float), cudaMemcpyDeviceToHost);
}


// C++ Macros
#if 0 

#define RGDS_ADD(ret, a, b) { ret = a + b; }
#define RGDS_SUB(ret, a, b) { ret = a - b; }
#define RGDS_MUL(ret, a, b) { ret = a * b; }
#define RGDS_DIV(ret, a, b) { ret = a / b; }
#define RGDS_SQRT(ret, a) { ret = sqrt(a); }
//#define RGDS_OUT_STR(a) c_dd_write(a)
#define RGDS_OUT_STR(a) { std::cout << dsfloat(a.x, a.y); }
//#define RGDS_SET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RGDS_GET_STR(a, str) c_dd_read(str, a)
#define RGDS_GET_D(a) ((double)a.x)
#define RGDS_SET_D(ret, d) { ret = (gds_real)make_float2((double)d, 0.0); }
#define RGDS_SET_UI(ret, org) { ret = (double)(org); }
#define RGDS_SET(ret, org) { ret = org; }
#define RGDS_NEG(ret, a) { ret.x = -a.x; ret.y = -a.y; }
#define RGDS_ABS(ret, a) { ret = abs(a); }
#define RGDS_UI_DIV(ret, a, b) { ret = (gds_real)make_float2(a, 0.0) / b; }
#define RGDS_UI_SUB(ret, a, b) { ret = (gds_real)make_float2(a, 0.0) - b; }
#define RGDS_DIV_D(ret, a, b) { ret = a / (gds_real)make_float2(b, 0.0); }
#define RGDS_ADD_D(ret, a, b) { ret = a + (gds_real)make_float2(b, 0.0); }
#define RGDS_SUB_D(ret, a, b) { ret = a - (gds_real)make_float2(b, 0.0); }
#define RGDS_MUL_D(ret, a, b) { ret = a * (gds_real)make_float2(b, 0.0); }
#define RGDS_DIV_UI(ret, a, b) { ret = a / (gds_real)make_float2(b, 0.0)); }
#define RGDS_ADD_UI(ret, a, b) { ret = a + (gds_real)make_float2(b, 0.0); }
#define RGDS_SUB_UI(ret, a, b) { ret = a - (gds_real)make_float2(b, 0.0); }
#define RGDS_MUL_UI(ret, a, b) { ret = a * (gds_real)make_float2(b, 0.0); }

#define RGQS_ADD(ret, a, b) { ret = a + b; }
#define RGQS_SUB(ret, a, b) { ret = a - b; }
#define RGQS_MUL(ret, a, b) { ret = a * b; }
#define RGQS_DIV(ret, a, b) { ret = a / b; }
#define RGQS_SQRT(ret, a) { ret = sqrt(a); }
//#define RGQS_OUT_STR(a) c_qd_write(a)
#define RGQS_OUT_STR(a) { std::cout << a; }
//#define RGQS_SET_STR(str, a) c_qd_swrite(a, 66, str, 84)
//#define RGQS_GET_STR(a, str) c_qd_read(str, a)
#define RGQS_GET_D(a) ((double)a)
#define RGQS_SET_D(ret, d) { ret = (double)d; }
#define RGQS_SET_UI(ret, org) { ret = (double)(org); }
#define RGQS_SET(ret, org) { ret = org; }
#define RGQS_NEG(ret, a) { ret = -a; }
#define RGQS_ABS(ret, a) { ret = abs(a); }
#define RGQS_UI_DIV(ret, a, b) { ret = (double)a / b; }
#define RGQS_UI_SUB(ret, a, b) { ret = (double)a - b; }
#define RGQS_DIV_D(ret, a, b) { ret = a / (gqs_real)b; }
#define RGQS_ADD_D(ret, a, b) { ret = a + (gqs_real)b; }
#define RGQS_SUB_D(ret, a, b) { ret = a - (gqs_real)b; }
#define RGQS_MUL_D(ret, a, b) { ret = a * (gqs_real)b; }
#define RGQS_DIV_UI(ret, a, b) { ret = a /(double)(b); }
#define RGQS_ADD_UI(ret, a, b) { ret = a +(double)(b); }
#define RGQS_SUB_UI(ret, a, b) { ret = a -(double)(b); }
#define RGQS_MUL_UI(ret, a, b) { ret = a *(double)(b); }
 
// DD print(no appending CR)
//void rgds_out_str_base(FILE *fp, int base, int length, double val[DSSIZE]);

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgds_cmp(gds_real a, gds_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgds_cmp_d(gds_real a, double b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD sqrt_d
//#define rgds_sqrt_d(ret, a) { ret = sqrt((gds_real)a); }
inline void rgds_sqrt_d(gds_real &ret, double a)
{
	dsfloat ret_tmp, a_tmp = (dsfloat)a;
//	gds_real a_tmp = (gds_real)make_float2(a, 0.0);
	ret_tmp = sqrt(a_tmp);
	ds2gds(&ret, &ret_tmp);
}

#define RGDS_CMP(a, b) rgds_cmp(a, b)
#define RGDS_CMP_D(a, b) rgds_cmp_d(a, b)
#define RGDS_CMP_UI(a, b) rgds_cmp_d(a, (double)(b))
#define RGDS_SQRT_D(ret, a) rgds_sqrt_d(ret, a)
#define RGDS_SQRT_UI(ret, a) rgds_sqrt_d(ret, (double)(a))

// QD print(no appending CR)
void rgqs_out_str_base(FILE *fp, int base, int length, double val[QSSIZE]);

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgqs_cmp(gqs_real a, gqs_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgqs_cmp_d(gqs_real a, double b)
{
	gqs_real b_tmp = (gqs_real)make_float4(b, 0.0, 0.0, 0.0);
	if(a > b_tmp) return 1;
	else if(a == b_tmp) return 0;
	else return -1;
}

// DD sqrt_d
//#define rgqs_sqrt_d(ret, a) { ret = sqrt((gqs_real)a); }
inline void rgqs_sqrt_d(gqs_real &ret, double a)
{
	//gqs_real a_tmp = (gqs_real)make_float4(a, 0.0, 0.0, 0.0);
	qsfloat ret_tmp, a_tmp = (qsfloat)a;
	ret_tmp = sqrt(a_tmp);
	qs2gqs(&ret, &ret_tmp);
}

#define RGQS_CMP(a, b) rgqs_cmp(a, b)
#define RGQS_CMP_D(a, b) rgqs_cmp_d(a, b)
#define RGQS_CMP_UI(a, b) rgqs_cmp_d(a, (double)(b))
#define RGQS_SQRT_D(ret, a) rgqs_sqrt_d(ret, a)
#define RGQS_SQRT_UI(ret, a) rgqs_sqrt_d(ret, (double)(a))

#define rgds_add(ret, a, b) RGDS_ADD(ret, a, b)
#define rgds_sub(ret, a, b) RGDS_SUB(ret, a, b)
#define rgds_mul(ret, a, b) RGDS_MUL(ret, a, b)
#define rgds_div(ret, a, b) RGDS_DIV(ret, a, b)
#define rgds_sqrt(ret, a) RGDS_SQRT(ret, a)
#define rgds_sqrt_d(ret, a) RGDS_SQRT_D(ret, a)
#define rgds_sqrt_ui(ret, a) RGDS_SQRT_UI(ret, a)
#define rgds_out_str(a) RGDS_OUT_STR(a)
#define rgds_set_str(str, a) RGDS_SET_STR(str, a)
#define rgds_get_str(a, str) RGDS_GET_STR(a, str)
#define rgds_get_d(a) RGDS_GET_D(a)
#define rgds_set_d(ret, d) RGDS_SET_D(ret, d)
#define rgds_set_ui(ret, d) RGDS_SET_UI(ret, d)
#define rgds_set(ret, org) RGDS_SET(ret, org)
#define rgds_neg(ret, a) RGDS_NEG(ret, a)
#define rgds_abs(ret, a) RGDS_ABS(ret, a)
#define rgds_cmp_ui(a, b) RGDS_CMP_UI(a, b)
#define rgds_ui_div(ret, a, b) RGDS_UI_DIV(ret, a, b)
#define rgds_ui_sub(ret, a, b) RGDS_UI_SUB(ret, a, b)
#define rgds_div_d(ret, a, b) RGDS_DIV_D(ret, a, b)
#define rgds_add_d(ret, a, b) RGDS_ADD_D(ret, a, b)
#define rgds_sub_d(ret, a, b) RGDS_SUB_D(ret, a, b)
#define rgds_mul_d(ret, a, b) RGDS_MUL_D(ret, a, b)
#define rgds_div_ui(ret, a, b) RGDS_DIV_UI(ret, a, b)
#define rgds_add_ui(ret, a, b) RGDS_ADD_UI(ret, a, b)
#define rgds_sub_ui(ret, a, b) RGDS_SUB_UI(ret, a, b)
#define rgds_mul_ui(ret, a, b) RGDS_MUL_UI(ret, a, b)

#define rgqs_add(ret, a, b) RGQS_ADD(ret, a, b)
#define rgqs_sub(ret, a, b) RGQS_SUB(ret, a, b)
#define rgqs_mul(ret, a, b) RGQS_MUL(ret, a, b)
#define rgqs_div(ret, a, b) RGQS_DIV(ret, a, b)
#define rgqs_sqrt(ret, a) RGQS_SQRT(ret, a)
#define rgqs_sqrt_d(ret, a) RGQS_SQRT_D(ret, a)
#define rgqs_sqrt_ui(ret, a) RGQS_SQRT_UI(ret, a)
#define rgqs_out_str(a) RGQS_OUT_STR(a)
#define rgqs_set_str(str, a) RGQS_SET_STR(str, a)
#define rgqs_get_str(a, str) RGQS_GET_STR(a, str)
#define rgqs_set_d(ret, d) RGQS_SET_D(ret, d)
#define rgqs_set_ui(ret, d) RGQS_SET_UI(ret, d)
#define rgqs_set(ret, org) RGQS_SET(ret, org)
#define rgqs_neg(ret, a) RGQS_NEG(ret, a)
#define rgqs_abs(ret, a) RGQS_ABS(ret, a)
#define rgqs_cmp_ui(a, b) RGQS_CMP_UI(a, b)
#define rgqs_ui_div(ret, a, b) RGQS_UI_DIV(ret, a, b)
#define rgqs_ui_sub(ret, a, b) RGQS_UI_SUB(ret, a, b)
#define rgqs_div_d(ret, a, b) RGQS_DIV_D(ret, a, b)
#define rgqs_add_d(ret, a, b) RGQS_ADD_D(ret, a, b)
#define rgqs_sub_d(ret, a, b) RGQS_SUB_D(ret, a, b)
#define rgqs_mul_d(ret, a, b) RGQS_MUL_D(ret, a, b)
#define rgqs_div_ui(ret, a, b) RGQS_DIV_UI(ret, a, b)
#define rgqs_add_ui(ret, a, b) RGQS_ADD_UI(ret, a, b)
#define rgqs_sub_ui(ret, a, b) RGQS_SUB_UI(ret, a, b)
#define rgqs_mul_ui(ret, a, b) RGQS_MUL_UI(ret, a, b)
#endif // 0

// constants on host
#define SET0_GDS(val) {(val).x = 0.0; (val).y = 0.0;}
#define SET1_GDS(val) {(val).x = 1.0; (val).y = 0.0;}
#define SET0_GQS(val) {(val).x = 0.0; (val).y = 0.0;(val).z = 0.0; (val).w = 0.0;}
#define SET1_GQS(val) {(val).x = 1.0; (val).y = 0.0;(val).z = 0.0; (val).w = 0.0;}

#define set0_gds(val) SET0_GDS(val)
#define set1_gds(val) SET1_GDS(val)
#define set0_gqs(val) SET0_GQS(val)
#define set1_gqs(val) SET1_GQS(val)

//inline gds_real const_gds_zero(&x = 0.0, y = 0.0)

// gdsval_dev := 0
inline void set0_gds_dev(gds_real *gdsval_dev)
{
	SET0_GDS(_const_gds_zero);

	cudaMemcpy((void *)&(gdsval_dev->x), (void *)&(_const_gds_zero.x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gdsval_dev->y), (void *)&(_const_gds_zero.y), sizeof(float), cudaMemcpyHostToDevice);
}

// gdsval_dev := 1
inline void set1_gds_dev(gds_real *gdsval_dev)
{
	SET1_GDS(_const_gds_one);

	cudaMemcpy((void *)&(gdsval_dev->x), (void *)&(_const_gds_one.x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gdsval_dev->y), (void *)&(_const_gds_one.y), sizeof(float), cudaMemcpyHostToDevice);
}

// gqsval_dev := 0
inline void set0_gqs_dev(gqs_real *gqsval_dev)
{
	SET0_GQS(_const_gqs_zero);

	cudaMemcpy((void *)&(gqsval_dev->x), (void *)&(_const_gqs_zero.x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->y), (void *)&(_const_gqs_zero.y), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->z), (void *)&(_const_gqs_zero.z), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->w), (void *)&(_const_gqs_zero.w), sizeof(float), cudaMemcpyHostToDevice);
}

// gqsval_dev := 1
inline void set1_gqs_dev(gqs_real *gqsval_dev)
{
	SET1_GQS(_const_gqs_one);

	cudaMemcpy((void *)&(gqsval_dev->x), (void *)&(_const_gqs_one.x), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->y), (void *)&(_const_gqs_one.y), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->z), (void *)&(_const_gqs_one.z), sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqsval_dev->w), (void *)&(_const_gqs_one.w), sizeof(float), cudaMemcpyHostToDevice);
}

// defined as macros
#define SET0_GDS_DEV(val) set0_gds_dev(val)
#define SET1_GDS_DEV(val) set1_gds_dev(val)
#define SET0_GQS_DEV(val) set0_gqs_dev(val)
#define SET1_GQS_DEV(val) set1_gqs_dev(val)

// reverse sign
#define NEG_GDS(ret, val) {(ret).x = -((val).x); (ret).y = -((val).y);}
#define NEG_GQS(ret, val) {(ret).x = -((val).x); (ret).y = -((val).y); (ret).z = -((val).z); (ret).w = -((val).w);}

// init_gds on GPU
gds_real * _bncuda_init_gds(void);

// free_gds_array on GPU
void _bncuda_free_gds(gds_real *dev_val);

// init_gds_array on GPU
gds_real * _bncuda_init_gds_array(int array_num);

// free_gds_array on GPU
void _bncuda_free_gds_array(gds_real *dev_array);

// init_gqs on GPU
__host__ gqs_real * _bncuda_init_gqs(void);

// free_gqs_array on GPU
void _bncuda_free_gqs(gqs_real *dev_val);

// init_gds_array on GPU
__host__ gqs_real * _bncuda_init_gqs_array(int array_num);

// free_gds_array on GPU
void _bncuda_free_gqs_array(gqs_real *dev_array);

// definition
#define MAX_NUM_THREADS_PER_BLOCK 128
#define MAX_NUM_BLOCKS_PER_GRID 128

// C++ Macros
#define SET_GDSVECTOR_I(vec, index, value) {\
	vec->element[index] = value;\
}
#define SET_GDSVECTOR_I_UI(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET_GDSVECTOR_I_D(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET0_GDSVECTOR_I(vec, index) {\
	vec->element[index] = (gds_real)0.0;\
}

#define GET_GDSVECTOR_I(vec, index) (vec->element[index])

#define set_gdsvector_i(vec, index, value) SET_GDSVECTOR_I(vec, index, value)
#define set_gdsvector_i_d(vec, index, value) SET_GDSVECTOR_I_D(vec, index, value)
#define set_gdsvector_i_ui(vec, index, value) SET_GDSVECTOR_I_UI(vec, index, value)
#define set0_gdsvector_i(vec, index) SET_GDSVECTOR_I(vec, index)
#define get_gdsvector_i(vec, index) GET_GDSVECTOR_I(vec, index)

// GDS vector
typedef struct
{
	long int dim;
	gds_real *element;
} gdsvector; // on CPU or GPU

typedef gdsvector *GDSVector;

// initialize gdsvector on CPU(host)
 GDSVector init_gdsvector(long int dim);

// free gdsvector on CPU(HOST)
void free_gdsvector(GDSVector vec);

// initialize gdsvector
GDSVector init_gdsvector_dev(long int dim);

// free dsvector
void free_gdsvector_dev(GDSVector vec);

// copy DSVector to GDSVector on GPU
// gdsvec on GPU := dsvec
void subst_gdsvector_dev_dsvec(GDSVector gdsvec_dev, DSVector dsvec);

// copy GDSVector on GPU to DSVector
// dsvec := gdsvec_dev on GPU
void subst_dsvector_gdsvec_dev(DSVector dsvec, GDSVector gdsvec_dev);

// print dsvector
void print_gdsvector_dev(GDSVector dev_vec);

/* c = a + b */
void add_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a - b */
void sub_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = val * a */
void cmul_gdsvector_dev(GDSVector c_dev, gds_real val, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := vec */
void subst_gdsvector_dev(GDSVector ret_dev, GDSVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gdsvector_dev(GDSVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);

/* (a, b) */
void ip_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm2
void norm2_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm1
void norm1_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm_inf
void normi_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// GDS matrix

// C++ Macros
#define SET_GDSMATRIX_IJ(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = value;\
}
#define SET_GDSMATRIX_IJ_D(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET_GDSMATRIX_IJ_UI(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET0_GDSMATRIX_IJ(mat, row_index, col_index) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (gds_real)0.0;\
}

#define GET_GDSMATRIX_IJ(mat, row_index, col_index) (mat->element[(row_index) * (mat->col_dim) + (col_index)])

#define set_gdsmatrix_ij(mat, row_index, col_index, value) SET_GDSMATRIX_IJ(mat, row_index, col_index, value)
#define set_gdsmatrix_ij_d(mat, row_index, col_index, value) SET_GDSMATRIX_IJ_D(mat, row_index, col_index, value)
#define set_gdsmatrix_ij_ui(mat, row_index, col_index, value) SET_GDSMATRIX_IJ_UI(mat, row_index, col_index, value)
#define set0_gdsmatrix_ij(mat, row_index, col_index) SET0_GDSMATRIX_IJ(mat, row_index, col_index)
#define get_gdsmatrix_ij(mat, row_index, col_index) GET_GDSMATRIX_IJ(mat, row_index, col_index)

typedef struct
{
	long int row_dim, col_dim;
	gds_real *element;
} gdsmatrix;

typedef gdsmatrix *GDSMatrix;

// set a zero matrix
void set0_gdsmatrix(GDSMatrix mat, int num_blocks_per_grid, int num_threads_per_block);

// initialize dsvector
GDSMatrix init_gdsmatrix(long int row_dim, long int col_dim);

// free dsvector
void free_gdsmatrix(GDSMatrix mat);

// initialize dsvector
GDSMatrix init_gdsmatrix_dev(long int row_dim, long int col_dim);

// free dsvector
void free_gdsmatrix_dev(GDSMatrix mat);

// print dsvector
void print_gdsmatrix_dev(GDSMatrix mat);

// matrix multiplication
// ret := A * B
void mul_gdsmatrix_dev(GDSMatrix ret_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// copy DSMatrix to GDSMatrix on GPU
// gdsvec on GPU := dsvec
void subst_gdsmatrix_dev_dsmat(GDSMatrix gdsmat_dev, DSMatrix dsmat);

// copy GDSMatrix on GPU to DSMatrix
// dsvec := gdsmat_dev on GPU
void subst_dsmatrix_gdsmat_dev(DSMatrix dsmat, GDSMatrix gdsmat_dev);

// Frobenius norm
void normf_gdsmatrix_dev(gds_real *ret_dev, GDSMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a - b */
void sub_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := sc * a */
void cmul_gdsmatrix_dev(GDSMatrix c_dev, gds_real sc, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a^T */
void transpose_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a */
void subst_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gdsmatrix_dev(GDSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* set (i, j)-element on GPU  */
void set_gdsmatrix_ij_dev(GDSMatrix mat_dev, long int row_index, long int col_index, gds_real val);

/* set (i, j)-element on GPU from dsfloat */
void set_gdsmatrix_ij_ds_dev(GDSMatrix mat_dev, long int row_index, long int col_index, dsfloat val);

/* c := I */
void setI_gdsmatrix_dev(GDSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* v := a * vb */
void mul_gdsmatrix_gdsvec(GDSVector v, GDSMatrix a, GDSVector vb, int num_blocks_per_grid, int num_threads_per_block);

/* v := a^T * vb */
void mul_gdsmatrixt_gdsvec(GDSVector v, GDSMatrix a, GDSVector vb, int num_blocks_per_grid, int num_threads_per_block);

// C++ Macros for GQS type

#define SET_GQSVECTOR_I(vec, index, value) {\
	vec->element[index] = value;\
}
#define SET_GQSVECTOR_I_UI(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET_GQSVECTOR_I_D(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET0_GQSVECTOR_I(vec, index) {\
	vec->element[index] = (gqs_real)0.0;\
}

#define GET_GQSVECTOR_I(vec, index) (vec->element[index])

#define set_gqsvector_i(vec, index, value) SET_GQSVECTOR_I(vec, index, value)
#define set_gqsvector_i_d(vec, index, value) SET_GQSVECTOR_I_D(vec, index, value)
#define set_gqsvector_i_ui(vec, index, value) SET_GQSVECTOR_I_UI(vec, index, value)
#define set0_gqsvector_i(vec, index) SET_GQSVECTOR_I(vec, index)
#define get_gqsvector_i(vec, index) GET_GQSVECTOR_I(vec, index)

// GQS vector
typedef struct
{
	long int dim;
	gqs_real *element;
} gqsvector; // on CPU or GPU

typedef gqsvector *GQSVector;

// initialize gqsvector on CPU(host)
GQSVector init_gqsvector(long int dim);

// free gqsvector on CPU(HOST)
void free_gqsvector(GQSVector vec);

// initialize gqsvector
GQSVector init_gqsvector_dev(long int dim);

// free qsvector
void free_gqsvector_dev(GQSVector vec);

// copy QSVector to GQSVector on GPU
// gqsvec on GPU := qsvec
void subst_gqsvector_dev_qsvec(GQSVector gqsvec_dev, QSVector qsvec);

// copy GQSVector on GPU to QSVector
// qsvec := gqsvec_dev on GPU
void subst_qsvector_gqsvec_dev(QSVector qsvec, GQSVector gqsvec_dev);

// print qsvector
void print_gqsvector_dev(GQSVector dev_vec);

/* c = a + b */
void add_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a - b */
void sub_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = val * a */
void cmul_gqsvector_dev(GQSVector c_dev, gqs_real val, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := vec */
void subst_gqsvector_dev(GQSVector ret_dev, GQSVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gqsvector_dev(GQSVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);

/* (a, b) */
void ip_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm2
void norm2_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm1
void norm1_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm_i
void normi_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// GQS matrix

// C++ Macros
#define SET_GQSMATRIX_IJ(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = value;\
}
#define SET_GQSMATRIX_IJ_D(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET_GQSMATRIX_IJ_UI(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET0_GQSMATRIX_IJ(mat, row_index, col_index) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (gqs_real)0.0;\
}

#define GET_GQSMATRIX_IJ(mat, row_index, col_index) (mat->element[(row_index) * (mat->col_dim) + (col_index)])

#define set_gqsmatrix_ij(mat, row_index, col_index, value) SET_GQSMATRIX_IJ(mat, row_index, col_index, value)
#define set_gqsmatrix_ij_d(mat, row_index, col_index, value) SET_GQSMATRIX_IJ_D(mat, row_index, col_index, value)
#define set_gqsmatrix_ij_ui(mat, row_index, col_index, value) SET_GQSMATRIX_IJ_UI(mat, row_index, col_index, value)
#define set0_gqsmatrix_ij(mat, row_index, col_index) SET0_GQSMATRIX_IJ(mat, row_index, col_index)
#define get_gqsmatrix_ij(mat, row_index, col_index) GET_GQSMATRIX_IJ(mat, row_index, col_index)

typedef struct
{
	long int row_dim, col_dim;
	gqs_real *element;
} gqsmatrix;

typedef gqsmatrix *GQSMatrix;

// set a zero matrix
void set0_gqsmatrix(GQSMatrix mat, int num_blocks_per_grid, int num_threads_per_block);

// initialize qsvector
GQSMatrix init_gqsmatrix(long int row_dim, long int col_dim);

// free qsvector
void free_gqsmatrix(GQSMatrix mat);

// initialize qsvector
GQSMatrix init_gqsmatrix_dev(long int row_dim, long int col_dim);

// free qsvector
void free_gqsmatrix_dev(GQSMatrix mat);

// print qsvector
void print_gqsmatrix_dev(GQSMatrix mat);

// matrix multiplication
// ret := A * B
void mul_gqsmatrix_dev(GQSMatrix ret_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// copy QSMatrix to GQSMatrix on GPU
// gqsvec on GPU := qsvec
void subst_gqsmatrix_dev_qsmat(GQSMatrix gqsmat_dev, QSMatrix qsmat);

// copy GQSMatrix on GPU to QSMatrix
// qsvec := gqsmat_dev on GPU
void subst_qsmatrix_gqsmat_dev(QSMatrix qsmat, GQSMatrix gqsmat_dev);

// Frobenius norm
void normf_gqsmatrix_dev(gqs_real *ret_dev, GQSMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a - b */
void sub_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := sc * a */
void cmul_gqsmatrix_dev(GQSMatrix c_dev, gqs_real sc, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a^T */
void transpose_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a */
void subst_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gqsmatrix_dev(GQSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* set (i, j)-element on GPU from qsfloat */
void set_gqsmatrix_ij_qs_dev(GQSMatrix mat_dev, long int row_index, long int col_index, qsfloat val);

/* c := I */
void setI_gqsmatrix_dev(GQSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* v := a * vb */
void mul_gqsmatrix_gqsvec(GQSVector v, GQSMatrix a, GQSVector vb, int num_blocks_per_grid, int num_threads_per_block);

/* v := a^T * vb */
void mul_gqsmatrixt_gqsvec(GQSVector v, GQSMatrix a, GQSVector vb, int num_blocks_per_grid, int num_threads_per_block);

#endif // __BNC_GDSLINEAR_H__
