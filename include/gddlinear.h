/********************************************************************************/
/* gddlinear.h: Double-double and Quadruple precision                           */
/*               Linear Computation Library with GQD and CUDA                   */
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
#ifndef __BNC_GDDLINEAR_H__
  #define __BNC_GDDLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>

// on CPU
#include "qd/qd_real.h"
#include "ddlinear.h"
#include "qdlinear.h"   // QDVector / QDMatrix used by GQD prototypes below

// on GPU
#include "cuda_runtime.h" // cudaMemcpy
#include "gqd_type.h" // defines gdd_real, gqd_real
#include "vector_types.h" // for GQD type
#ifdef __CUDACC__
	#ifndef __INCLUDE_GQD_CU__
		#include "gqd.cuh"
	#else
		#include "gqd.cu"
	#endif // __INCLUDE_GQD_CU__
#endif // __CUDACC__

// Constants
static gdd_real _const_gdd_zero = {0.0, 0.0}; // zero
static gdd_real _const_gdd_one = {1.0, 0.0}; // 1
static gqd_real _const_gqd_zero = {0.0, 0.0, 0.0, 0.0}; // zero
static gqd_real _const_gqd_one = {1.0, 0.0, 0.0, 0.0}; // 1

// convert gdd on GPU to dd on CPU
//Ref: http://homepages.math.uic.edu/~jan/mcs572/quad_double_cuda.pdf
inline dd_real gdd_get_dd(gdd_real gval)
{
	dd_real ret;

	ret.x[0] = gval.x;
	ret.x[1] = gval.y;

	return ret;
}

// convert gdd on GPU to qd on CPU
//Ref: http://homepages.math.uic.edu/~jan/mcs572/quad_double_cuda.pdf
inline qd_real gqd_get_qd(gqd_real gval)
{
	qd_real ret;

	ret.x[0] = gval.x;
	ret.x[1] = gval.y;
	ret.x[2] = gval.z;
	ret.x[3] = gval.w;

	return ret;
}
// ddval := gddval
inline void gdd2dd(dd_real *ddval, gdd_real *gddval)
{
	ddval->x[0] = gddval->x;
	ddval->x[1] = gddval->y;
}

// qdval := gqdval
inline void gqd2qd(qd_real *qdval, gqd_real *gqdval)
{
	qdval->x[0] = gqdval->x;
	qdval->x[1] = gqdval->y;
	qdval->x[2] = gqdval->z;
	qdval->x[3] = gqdval->w;
}

// gddval := ddval
inline void dd2gdd(gdd_real *gddval, dd_real *ddval)
{
	gddval->x = ddval->x[0];
	gddval->y = ddval->x[1];
}

// gqdval := qdval
inline void qd2gqd(gqd_real *gqdval, qd_real *qdval)
{
	gqdval->x = qdval->x[0];
	gqdval->y = qdval->x[1];
	gqdval->z = qdval->x[2];
	gqdval->w = qdval->x[3];
}

// gddval on GPU := ddval
inline void dd2gdd_dev(gdd_real *gddval_dev, dd_real *ddval)
{
	cudaMemcpy((void *)&(gddval_dev->x), (void *)&(ddval->x[0]), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gddval_dev->y), (void *)&(ddval->x[1]), sizeof(double), cudaMemcpyHostToDevice);
}

// ddval := gddval on GPU
inline void gdd2dd_dev(dd_real *ddval, gdd_real *gddval_dev)
{
	cudaMemcpy((void *)&(ddval->x[0]), (void *)&(gddval_dev->x), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(ddval->x[1]), (void *)&(gddval_dev->y), sizeof(double), cudaMemcpyDeviceToHost);
}

// gddval := gddval on GPU
inline void gdd2gdd_dev(gdd_real *gddval, gdd_real *gddval_dev)
{
	cudaMemcpy((void *)&(gddval->x), (void *)&(gddval_dev->x), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gddval->y), (void *)&(gddval_dev->y), sizeof(double), cudaMemcpyDeviceToHost);
}

// gddval on GPU := gddval on HOST
inline void gdd_dev2gdd(gdd_real *gddval_dev, gdd_real *gddval)
{
	cudaMemcpy((void *)&(gddval_dev->x), (void *)&(gddval->x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gddval_dev->y), (void *)&(gddval->y), sizeof(double), cudaMemcpyHostToDevice);
}

// gqdval on GPU := qdval
inline void qd2gqd_dev(gqd_real *gqdval_dev, qd_real *qdval)
{
	cudaMemcpy((void *)&(gqdval_dev->x), (void *)&(qdval->x[0]), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->y), (void *)&(qdval->x[1]), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->z), (void *)&(qdval->x[2]), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->w), (void *)&(qdval->x[3]), sizeof(double), cudaMemcpyHostToDevice);
}

// qdval := gqdval on GPU
inline void gqd2qd_dev(qd_real *qdval, gqd_real *gqdval_dev)
{
	cudaMemcpy((void *)&(qdval->x[0]), (void *)&(gqdval_dev->x), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qdval->x[1]), (void *)&(gqdval_dev->y), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qdval->x[2]), (void *)&(gqdval_dev->z), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(qdval->x[3]), (void *)&(gqdval_dev->w), sizeof(double), cudaMemcpyDeviceToHost);
}

// gqdval on GPU := gqdval on HOST
inline void gqd_dev2gqd(gqd_real *gqdval_dev, gqd_real *gqdval)
{
	cudaMemcpy((void *)&(gqdval_dev->x), (void *)&(gqdval->x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->y), (void *)&(gqdval->y), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->z), (void *)&(gqdval->z), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->w), (void *)&(gqdval->w), sizeof(double), cudaMemcpyHostToDevice);
}

// gddval := gddval on GPU
inline void gqd2gqd_dev(gqd_real *gqdval, gqd_real *gqdval_dev)
{
	cudaMemcpy((void *)&(gqdval->x), (void *)&(gqdval_dev->x), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqdval->y), (void *)&(gqdval_dev->y), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqdval->z), (void *)&(gqdval_dev->z), sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&(gqdval->w), (void *)&(gqdval_dev->w), sizeof(double), cudaMemcpyDeviceToHost);
}


// C++ Macros
#if 0 

#define RGDD_ADD(ret, a, b) { ret = a + b; }
#define RGDD_SUB(ret, a, b) { ret = a - b; }
#define RGDD_MUL(ret, a, b) { ret = a * b; }
#define RGDD_DIV(ret, a, b) { ret = a / b; }
#define RGDD_SQRT(ret, a) { ret = sqrt(a); }
//#define RGDD_OUT_STR(a) c_dd_write(a)
#define RGDD_OUT_STR(a) { std::cout << dd_real(a.x, a.y); }
//#define RGDD_SET_STR(str, a) c_dd_swrite(a, 33, str, 48)
//#define RGDD_GET_STR(a, str) c_dd_read(str, a)
#define RGDD_GET_D(a) ((double)a.x)
#define RGDD_SET_D(ret, d) { ret = (gdd_real)make_double2((double)d, 0.0); }
#define RGDD_SET_UI(ret, org) { ret = (double)(org); }
#define RGDD_SET(ret, org) { ret = org; }
#define RGDD_NEG(ret, a) { ret.x = -a.x; ret.y = -a.y; }
#define RGDD_ABS(ret, a) { ret = abs(a); }
#define RGDD_UI_DIV(ret, a, b) { ret = (gdd_real)make_double2(a, 0.0) / b; }
#define RGDD_UI_SUB(ret, a, b) { ret = (gdd_real)make_double2(a, 0.0) - b; }
#define RGDD_DIV_D(ret, a, b) { ret = a / (gdd_real)make_double2(b, 0.0); }
#define RGDD_ADD_D(ret, a, b) { ret = a + (gdd_real)make_double2(b, 0.0); }
#define RGDD_SUB_D(ret, a, b) { ret = a - (gdd_real)make_double2(b, 0.0); }
#define RGDD_MUL_D(ret, a, b) { ret = a * (gdd_real)make_double2(b, 0.0); }
#define RGDD_DIV_UI(ret, a, b) { ret = a / (gdd_real)make_double2(b, 0.0)); }
#define RGDD_ADD_UI(ret, a, b) { ret = a + (gdd_real)make_double2(b, 0.0); }
#define RGDD_SUB_UI(ret, a, b) { ret = a - (gdd_real)make_double2(b, 0.0); }
#define RGDD_MUL_UI(ret, a, b) { ret = a * (gdd_real)make_double2(b, 0.0); }

#define RGQD_ADD(ret, a, b) { ret = a + b; }
#define RGQD_SUB(ret, a, b) { ret = a - b; }
#define RGQD_MUL(ret, a, b) { ret = a * b; }
#define RGQD_DIV(ret, a, b) { ret = a / b; }
#define RGQD_SQRT(ret, a) { ret = sqrt(a); }
//#define RGQD_OUT_STR(a) c_qd_write(a)
#define RGQD_OUT_STR(a) { std::cout << a; }
//#define RGQD_SET_STR(str, a) c_qd_swrite(a, 66, str, 84)
//#define RGQD_GET_STR(a, str) c_qd_read(str, a)
#define RGQD_GET_D(a) ((double)a)
#define RGQD_SET_D(ret, d) { ret = (double)d; }
#define RGQD_SET_UI(ret, org) { ret = (double)(org); }
#define RGQD_SET(ret, org) { ret = org; }
#define RGQD_NEG(ret, a) { ret = -a; }
#define RGQD_ABS(ret, a) { ret = abs(a); }
#define RGQD_UI_DIV(ret, a, b) { ret = (double)a / b; }
#define RGQD_UI_SUB(ret, a, b) { ret = (double)a - b; }
#define RGQD_DIV_D(ret, a, b) { ret = a / (gqd_real)b; }
#define RGQD_ADD_D(ret, a, b) { ret = a + (gqd_real)b; }
#define RGQD_SUB_D(ret, a, b) { ret = a - (gqd_real)b; }
#define RGQD_MUL_D(ret, a, b) { ret = a * (gqd_real)b; }
#define RGQD_DIV_UI(ret, a, b) { ret = a /(double)(b); }
#define RGQD_ADD_UI(ret, a, b) { ret = a +(double)(b); }
#define RGQD_SUB_UI(ret, a, b) { ret = a -(double)(b); }
#define RGQD_MUL_UI(ret, a, b) { ret = a *(double)(b); }
 
// DD print(no appending CR)
//void rgdd_out_str_base(FILE *fp, int base, int length, double val[DDSIZE]);

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgdd_cmp(gdd_real a, gdd_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgdd_cmp_d(gdd_real a, double b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// DD sqrt_d
//#define rgdd_sqrt_d(ret, a) { ret = sqrt((gdd_real)a); }
inline void rgdd_sqrt_d(gdd_real &ret, double a)
{
	dd_real ret_tmp, a_tmp = (dd_real)a;
//	gdd_real a_tmp = (gdd_real)make_double2(a, 0.0);
	ret_tmp = sqrt(a_tmp);
	dd2gdd(&ret, &ret_tmp);
}

#define RGDD_CMP(a, b) rgdd_cmp(a, b)
#define RGDD_CMP_D(a, b) rgdd_cmp_d(a, b)
#define RGDD_CMP_UI(a, b) rgdd_cmp_d(a, (double)(b))
#define RGDD_SQRT_D(ret, a) rgdd_sqrt_d(ret, a)
#define RGDD_SQRT_UI(ret, a) rgdd_sqrt_d(ret, (double)(a))

// QD print(no appending CR)
void rgqd_out_str_base(FILE *fp, int base, int length, double val[QDSIZE]);

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgqd_cmp(gqd_real a, gqd_real b) { if(a > b) return 1; else if(a == b) return 0; else return -1; }

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
inline int rgqd_cmp_d(gqd_real a, double b)
{
	gqd_real b_tmp = (gqd_real)make_double4(b, 0.0, 0.0, 0.0);
	if(a > b_tmp) return 1;
	else if(a == b_tmp) return 0;
	else return -1;
}

// DD sqrt_d
//#define rgqd_sqrt_d(ret, a) { ret = sqrt((gqd_real)a); }
inline void rgqd_sqrt_d(gqd_real &ret, double a)
{
	//gqd_real a_tmp = (gqd_real)make_double4(a, 0.0, 0.0, 0.0);
	qd_real ret_tmp, a_tmp = (qd_real)a;
	ret_tmp = sqrt(a_tmp);
	qd2gqd(&ret, &ret_tmp);
}

#define RGQD_CMP(a, b) rgqd_cmp(a, b)
#define RGQD_CMP_D(a, b) rgqd_cmp_d(a, b)
#define RGQD_CMP_UI(a, b) rgqd_cmp_d(a, (double)(b))
#define RGQD_SQRT_D(ret, a) rgqd_sqrt_d(ret, a)
#define RGQD_SQRT_UI(ret, a) rgqd_sqrt_d(ret, (double)(a))

#define rgdd_add(ret, a, b) RGDD_ADD(ret, a, b)
#define rgdd_sub(ret, a, b) RGDD_SUB(ret, a, b)
#define rgdd_mul(ret, a, b) RGDD_MUL(ret, a, b)
#define rgdd_div(ret, a, b) RGDD_DIV(ret, a, b)
#define rgdd_sqrt(ret, a) RGDD_SQRT(ret, a)
#define rgdd_sqrt_d(ret, a) RGDD_SQRT_D(ret, a)
#define rgdd_sqrt_ui(ret, a) RGDD_SQRT_UI(ret, a)
#define rgdd_out_str(a) RGDD_OUT_STR(a)
#define rgdd_set_str(str, a) RGDD_SET_STR(str, a)
#define rgdd_get_str(a, str) RGDD_GET_STR(a, str)
#define rgdd_get_d(a) RGDD_GET_D(a)
#define rgdd_set_d(ret, d) RGDD_SET_D(ret, d)
#define rgdd_set_ui(ret, d) RGDD_SET_UI(ret, d)
#define rgdd_set(ret, org) RGDD_SET(ret, org)
#define rgdd_neg(ret, a) RGDD_NEG(ret, a)
#define rgdd_abs(ret, a) RGDD_ABS(ret, a)
#define rgdd_cmp_ui(a, b) RGDD_CMP_UI(a, b)
#define rgdd_ui_div(ret, a, b) RGDD_UI_DIV(ret, a, b)
#define rgdd_ui_sub(ret, a, b) RGDD_UI_SUB(ret, a, b)
#define rgdd_div_d(ret, a, b) RGDD_DIV_D(ret, a, b)
#define rgdd_add_d(ret, a, b) RGDD_ADD_D(ret, a, b)
#define rgdd_sub_d(ret, a, b) RGDD_SUB_D(ret, a, b)
#define rgdd_mul_d(ret, a, b) RGDD_MUL_D(ret, a, b)
#define rgdd_div_ui(ret, a, b) RGDD_DIV_UI(ret, a, b)
#define rgdd_add_ui(ret, a, b) RGDD_ADD_UI(ret, a, b)
#define rgdd_sub_ui(ret, a, b) RGDD_SUB_UI(ret, a, b)
#define rgdd_mul_ui(ret, a, b) RGDD_MUL_UI(ret, a, b)

#define rgqd_add(ret, a, b) RGQD_ADD(ret, a, b)
#define rgqd_sub(ret, a, b) RGQD_SUB(ret, a, b)
#define rgqd_mul(ret, a, b) RGQD_MUL(ret, a, b)
#define rgqd_div(ret, a, b) RGQD_DIV(ret, a, b)
#define rgqd_sqrt(ret, a) RGQD_SQRT(ret, a)
#define rgqd_sqrt_d(ret, a) RGQD_SQRT_D(ret, a)
#define rgqd_sqrt_ui(ret, a) RGQD_SQRT_UI(ret, a)
#define rgqd_out_str(a) RGQD_OUT_STR(a)
#define rgqd_set_str(str, a) RGQD_SET_STR(str, a)
#define rgqd_get_str(a, str) RGQD_GET_STR(a, str)
#define rgqd_set_d(ret, d) RGQD_SET_D(ret, d)
#define rgqd_set_ui(ret, d) RGQD_SET_UI(ret, d)
#define rgqd_set(ret, org) RGQD_SET(ret, org)
#define rgqd_neg(ret, a) RGQD_NEG(ret, a)
#define rgqd_abs(ret, a) RGQD_ABS(ret, a)
#define rgqd_cmp_ui(a, b) RGQD_CMP_UI(a, b)
#define rgqd_ui_div(ret, a, b) RGQD_UI_DIV(ret, a, b)
#define rgqd_ui_sub(ret, a, b) RGQD_UI_SUB(ret, a, b)
#define rgqd_div_d(ret, a, b) RGQD_DIV_D(ret, a, b)
#define rgqd_add_d(ret, a, b) RGQD_ADD_D(ret, a, b)
#define rgqd_sub_d(ret, a, b) RGQD_SUB_D(ret, a, b)
#define rgqd_mul_d(ret, a, b) RGQD_MUL_D(ret, a, b)
#define rgqd_div_ui(ret, a, b) RGQD_DIV_UI(ret, a, b)
#define rgqd_add_ui(ret, a, b) RGQD_ADD_UI(ret, a, b)
#define rgqd_sub_ui(ret, a, b) RGQD_SUB_UI(ret, a, b)
#define rgqd_mul_ui(ret, a, b) RGQD_MUL_UI(ret, a, b)
#endif // 0

// constants on host
#define SET0_GDD(val) {(val).x = 0.0; (val).y = 0.0;}
#define SET1_GDD(val) {(val).x = 1.0; (val).y = 0.0;}
#define SET0_GQD(val) {(val).x = 0.0; (val).y = 0.0;(val).z = 0.0; (val).w = 0.0;}
#define SET1_GQD(val) {(val).x = 1.0; (val).y = 0.0;(val).z = 0.0; (val).w = 0.0;}

#define set0_gdd(val) SET0_GDD(val)
#define set1_gdd(val) SET1_GDD(val)
#define set0_gqd(val) SET0_GQD(val)
#define set1_gqd(val) SET1_GQD(val)

//inline gdd_real const_gdd_zero(&x = 0.0, y = 0.0)

// gddval_dev := 0
inline void set0_gdd_dev(gdd_real *gddval_dev)
{
	SET0_GDD(_const_gdd_zero);

	cudaMemcpy((void *)&(gddval_dev->x), (void *)&(_const_gdd_zero.x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gddval_dev->y), (void *)&(_const_gdd_zero.y), sizeof(double), cudaMemcpyHostToDevice);
}

// gddval_dev := 1
inline void set1_gdd_dev(gdd_real *gddval_dev)
{
	SET1_GDD(_const_gdd_one);

	cudaMemcpy((void *)&(gddval_dev->x), (void *)&(_const_gdd_one.x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gddval_dev->y), (void *)&(_const_gdd_one.y), sizeof(double), cudaMemcpyHostToDevice);
}

// gqdval_dev := 0
inline void set0_gqd_dev(gqd_real *gqdval_dev)
{
	SET0_GQD(_const_gqd_zero);

	cudaMemcpy((void *)&(gqdval_dev->x), (void *)&(_const_gqd_zero.x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->y), (void *)&(_const_gqd_zero.y), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->z), (void *)&(_const_gqd_zero.z), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->w), (void *)&(_const_gqd_zero.w), sizeof(double), cudaMemcpyHostToDevice);
}

// gqdval_dev := 1
inline void set1_gqd_dev(gqd_real *gqdval_dev)
{
	SET1_GQD(_const_gqd_one);

	cudaMemcpy((void *)&(gqdval_dev->x), (void *)&(_const_gqd_one.x), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->y), (void *)&(_const_gqd_one.y), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->z), (void *)&(_const_gqd_one.z), sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)&(gqdval_dev->w), (void *)&(_const_gqd_one.w), sizeof(double), cudaMemcpyHostToDevice);
}

// defined as macros
#define SET0_GDD_DEV(val) set0_gdd_dev(val)
#define SET1_GDD_DEV(val) set1_gdd_dev(val)
#define SET0_GQD_DEV(val) set0_gqd_dev(val)
#define SET1_GQD_DEV(val) set1_gqd_dev(val)

// reverse sign
#define NEG_GDD(ret, val) {(ret).x = -((val).x); (ret).y = -((val).y);}
#define NEG_GQD(ret, val) {(ret).x = -((val).x); (ret).y = -((val).y); (ret).z = -((val).z); (ret).w = -((val).w);}

// init_gdd on GPU
gdd_real * _bncuda_init_gdd(void);

// free_gdd_array on GPU
void _bncuda_free_gdd(gdd_real *dev_val);

// init_gdd_array on GPU
gdd_real * _bncuda_init_gdd_array(int array_num);

// free_gdd_array on GPU
void _bncuda_free_gdd_array(gdd_real *dev_array);

// init_gqd on GPU
__host__ gqd_real * _bncuda_init_gqd(void);

// free_gqd_array on GPU
void _bncuda_free_gqd(gqd_real *dev_val);

// init_gdd_array on GPU
__host__ gqd_real * _bncuda_init_gqd_array(int array_num);

// free_gdd_array on GPU
void _bncuda_free_gqd_array(gqd_real *dev_array);

// definition
#define MAX_NUM_THREADS_PER_BLOCK 128
#define MAX_NUM_BLOCKS_PER_GRID 128

// C++ Macros
#define SET_GDDVECTOR_I(vec, index, value) {\
	vec->element[index] = value;\
}
#define SET_GDDVECTOR_I_UI(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET_GDDVECTOR_I_D(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET0_GDDVECTOR_I(vec, index) {\
	vec->element[index] = (gdd_real)0.0;\
}

#define GET_GDDVECTOR_I(vec, index) (vec->element[index])

#define set_gddvector_i(vec, index, value) SET_GDDVECTOR_I(vec, index, value)
#define set_gddvector_i_d(vec, index, value) SET_GDDVECTOR_I_D(vec, index, value)
#define set_gddvector_i_ui(vec, index, value) SET_GDDVECTOR_I_UI(vec, index, value)
#define set0_gddvector_i(vec, index) SET_GDDVECTOR_I(vec, index)
#define get_gddvector_i(vec, index) GET_GDDVECTOR_I(vec, index)

// GDD vector
typedef struct
{
	long int dim;
	gdd_real *element;
} gddvector; // on CPU or GPU

typedef gddvector *GDDVector;

// initialize gddvector on CPU(host)
 GDDVector init_gddvector(long int dim);

// free gddvector on CPU(HOST)
void free_gddvector(GDDVector vec);

// initialize gddvector
GDDVector init_gddvector_dev(long int dim);

// free ddvector
void free_gddvector_dev(GDDVector vec);

// copy DDVector to GDDVector on GPU
// gddvec on GPU := ddvec
void subst_gddvector_dev_ddvec(GDDVector gddvec_dev, DDVector ddvec);

// copy GDDVector on GPU to DDVector
// ddvec := gddvec_dev on GPU
void subst_ddvector_gddvec_dev(DDVector ddvec, GDDVector gddvec_dev);

// print ddvector
void print_gddvector_dev(GDDVector dev_vec);

/* c = a + b */
void add_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a - b */
void sub_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = val * a */
void cmul_gddvector_dev(GDDVector c_dev, gdd_real val, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := vec */
void subst_gddvector_dev(GDDVector ret_dev, GDDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gddvector_dev(GDDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);

/* (a, b) */
void ip_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm2
void norm2_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm1
void norm1_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm_inf
void normi_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// GDD matrix

// C++ Macros
#define SET_GDDMATRIX_IJ(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = value;\
}
#define SET_GDDMATRIX_IJ_D(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET_GDDMATRIX_IJ_UI(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET0_GDDMATRIX_IJ(mat, row_index, col_index) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (gdd_real)0.0;\
}

#define GET_GDDMATRIX_IJ(mat, row_index, col_index) (mat->element[(row_index) * (mat->col_dim) + (col_index)])

#define set_gddmatrix_ij(mat, row_index, col_index, value) SET_GDDMATRIX_IJ(mat, row_index, col_index, value)
#define set_gddmatrix_ij_d(mat, row_index, col_index, value) SET_GDDMATRIX_IJ_D(mat, row_index, col_index, value)
#define set_gddmatrix_ij_ui(mat, row_index, col_index, value) SET_GDDMATRIX_IJ_UI(mat, row_index, col_index, value)
#define set0_gddmatrix_ij(mat, row_index, col_index) SET0_GDDMATRIX_IJ(mat, row_index, col_index)
#define get_gddmatrix_ij(mat, row_index, col_index) GET_GDDMATRIX_IJ(mat, row_index, col_index)

typedef struct
{
	long int row_dim, col_dim;
	gdd_real *element;
} gddmatrix;

typedef gddmatrix *GDDMatrix;

// set a zero matrix
void set0_gddmatrix(GDDMatrix mat, int num_blocks_per_grid, int num_threads_per_block);

// initialize ddvector
GDDMatrix init_gddmatrix(long int row_dim, long int col_dim);

// free ddvector
void free_gddmatrix(GDDMatrix mat);

// initialize ddvector
GDDMatrix init_gddmatrix_dev(long int row_dim, long int col_dim);

// free ddvector
void free_gddmatrix_dev(GDDMatrix mat);

// print ddvector
void print_gddmatrix_dev(GDDMatrix mat);

// matrix multiplication
// ret := A * B
void mul_gddmatrix_dev(GDDMatrix ret_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// copy DDMatrix to GDDMatrix on GPU
// gddvec on GPU := ddvec
void subst_gddmatrix_dev_ddmat(GDDMatrix gddmat_dev, DDMatrix ddmat);

// copy GDDMatrix on GPU to DDMatrix
// ddvec := gddmat_dev on GPU
void subst_ddmatrix_gddmat_dev(DDMatrix ddmat, GDDMatrix gddmat_dev);

// Frobenius norm
void normf_gddmatrix_dev(gdd_real *ret_dev, GDDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a - b */
void sub_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := sc * a */
void cmul_gddmatrix_dev(GDDMatrix c_dev, gdd_real sc, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a^T */
void transpose_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a */
void subst_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gddmatrix_dev(GDDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* set (i, j)-element on GPU  */
void set_gddmatrix_ij_dev(GDDMatrix mat_dev, long int row_index, long int col_index, gdd_real val);

/* set (i, j)-element on GPU from dd_real */
void set_gddmatrix_ij_dd_dev(GDDMatrix mat_dev, long int row_index, long int col_index, dd_real val);

/* c := I */
void setI_gddmatrix_dev(GDDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* v := a * vb */
void mul_gddmatrix_gddvec(GDDVector v, GDDMatrix a, GDDVector vb, int num_blocks_per_grid, int num_threads_per_block);

/* v := a^T * vb */
void mul_gddmatrixt_gddvec(GDDVector v, GDDMatrix a, GDDVector vb, int num_blocks_per_grid, int num_threads_per_block);

// C++ Macros for GQD type

#define SET_GQDVECTOR_I(vec, index, value) {\
	vec->element[index] = value;\
}
#define SET_GQDVECTOR_I_UI(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET_GQDVECTOR_I_D(vec, index, value) {\
	vec->element[index] = (double)value;\
}
#define SET0_GQDVECTOR_I(vec, index) {\
	vec->element[index] = (gqd_real)0.0;\
}

#define GET_GQDVECTOR_I(vec, index) (vec->element[index])

#define set_gqdvector_i(vec, index, value) SET_GQDVECTOR_I(vec, index, value)
#define set_gqdvector_i_d(vec, index, value) SET_GQDVECTOR_I_D(vec, index, value)
#define set_gqdvector_i_ui(vec, index, value) SET_GQDVECTOR_I_UI(vec, index, value)
#define set0_gqdvector_i(vec, index) SET_GQDVECTOR_I(vec, index)
#define get_gqdvector_i(vec, index) GET_GQDVECTOR_I(vec, index)

// GQD vector
typedef struct
{
	long int dim;
	gqd_real *element;
} gqdvector; // on CPU or GPU

typedef gqdvector *GQDVector;

// initialize gqdvector on CPU(host)
GQDVector init_gqdvector(long int dim);

// free gqdvector on CPU(HOST)
void free_gqdvector(GQDVector vec);

// initialize gqdvector
GQDVector init_gqdvector_dev(long int dim);

// free qdvector
void free_gqdvector_dev(GQDVector vec);

// copy QDVector to GQDVector on GPU
// gqdvec on GPU := qdvec
void subst_gqdvector_dev_qdvec(GQDVector gqdvec_dev, QDVector qdvec);

// copy GQDVector on GPU to QDVector
// qdvec := gqdvec_dev on GPU
void subst_qdvector_gqdvec_dev(QDVector qdvec, GQDVector gqdvec_dev);

// print qdvector
void print_gqdvector_dev(GQDVector dev_vec);

/* c = a + b */
void add_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a - b */
void sub_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = val * a */
void cmul_gqdvector_dev(GQDVector c_dev, gqd_real val, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := vec */
void subst_gqdvector_dev(GQDVector ret_dev, GQDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gqdvector_dev(GQDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);

/* (a, b) */
void ip_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm2
void norm2_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm1
void norm1_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// norm_i
void normi_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

// GQD matrix

// C++ Macros
#define SET_GQDMATRIX_IJ(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = value;\
}
#define SET_GQDMATRIX_IJ_D(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET_GQDMATRIX_IJ_UI(mat, row_index, col_index, value) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (double)value;\
}
#define SET0_GQDMATRIX_IJ(mat, row_index, col_index) {\
	mat->element[(row_index) * (mat->col_dim) + (col_index)] = (gqd_real)0.0;\
}

#define GET_GQDMATRIX_IJ(mat, row_index, col_index) (mat->element[(row_index) * (mat->col_dim) + (col_index)])

#define set_gqdmatrix_ij(mat, row_index, col_index, value) SET_GQDMATRIX_IJ(mat, row_index, col_index, value)
#define set_gqdmatrix_ij_d(mat, row_index, col_index, value) SET_GQDMATRIX_IJ_D(mat, row_index, col_index, value)
#define set_gqdmatrix_ij_ui(mat, row_index, col_index, value) SET_GQDMATRIX_IJ_UI(mat, row_index, col_index, value)
#define set0_gqdmatrix_ij(mat, row_index, col_index) SET0_GQDMATRIX_IJ(mat, row_index, col_index)
#define get_gqdmatrix_ij(mat, row_index, col_index) GET_GQDMATRIX_IJ(mat, row_index, col_index)

typedef struct
{
	long int row_dim, col_dim;
	gqd_real *element;
} gqdmatrix;

typedef gqdmatrix *GQDMatrix;

// set a zero matrix
void set0_gqdmatrix(GQDMatrix mat, int num_blocks_per_grid, int num_threads_per_block);

// initialize qdvector
GQDMatrix init_gqdmatrix(long int row_dim, long int col_dim);

// free qdvector
void free_gqdmatrix(GQDMatrix mat);

// initialize qdvector
GQDMatrix init_gqdmatrix_dev(long int row_dim, long int col_dim);

// free qdvector
void free_gqdmatrix_dev(GQDMatrix mat);

// print qdvector
void print_gqdmatrix_dev(GQDMatrix mat);

// matrix multiplication
// ret := A * B
void mul_gqdmatrix_dev(GQDMatrix ret_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// copy QDMatrix to GQDMatrix on GPU
// gqdvec on GPU := qdvec
void subst_gqdmatrix_dev_qdmat(GQDMatrix gqdmat_dev, QDMatrix qdmat);

// copy GQDMatrix on GPU to QDMatrix
// qdvec := gqdmat_dev on GPU
void subst_qdmatrix_gqdmat_dev(QDMatrix qdmat, GQDMatrix gqdmat_dev);

// Frobenius norm
void normf_gqdmatrix_dev(gqd_real *ret_dev, GQDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a - b */
void sub_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := sc * a */
void cmul_gqdmatrix_dev(GQDMatrix c_dev, gqd_real sc, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c = a^T */
void transpose_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a */
void subst_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := 0 */
void set0_gqdmatrix_dev(GQDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* set (i, j)-element on GPU from qd_real */
void set_gqdmatrix_ij_qd_dev(GQDMatrix mat_dev, long int row_index, long int col_index, qd_real val);

/* c := I */
void setI_gqdmatrix_dev(GQDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);

/* v := a * vb */
void mul_gqdmatrix_gqdvec(GQDVector v, GQDMatrix a, GQDVector vb, int num_blocks_per_grid, int num_threads_per_block);

/* v := a^T * vb */
void mul_gqdmatrixt_gqdvec(GQDVector v, GQDMatrix a, GQDVector vb, int num_blocks_per_grid, int num_threads_per_block);

#endif // __BNC_GDDLINEAR_H__
