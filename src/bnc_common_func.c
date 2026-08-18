/********************************************************************************/
/* bnc_common_func.c: BNCmatmul & BNCpack commom primary functions              */
/* Copyright (C) 2022 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1  : bnc_print_env_all has been implemented                        */
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

#include "bnc_common.h"

#include "rcdd.h" // for c[dtq]dfloat

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void bnc_print_env_all(void)
{
	printf("----------\n");

	//---------------------
	// Compiler name
	//---------------------
	printf("Compiler : ");
	// Intel Compiler or not
	// ref) https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/macros/additional-predefined-macros.html
	#if defined(__ICC) || defined(__ICL) // __ICC: Linux or macOS, __ICL: Windows
		//#define BNC_USE_ICC
		printf("Intel Compiler ");
	// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html#Common-Predefined-Macros
	#elif defined(__GNUC__) // GCC
		#define BNC_USE_GCC
		printf("GNU Compiler Collection ");
	#elif defined(_MSC_VER) // Microsoft Visual C++
		printf("Microsoft Visual C++ %ld", _MSC_VER);
	#else // unknown compiler
		//#undef BNC_USE_ICC
		//#undef BNC_USE_GCC
		printf("Unknown compiler");
	#endif // BNC_USE_ICC or _GCC
	// C++?
	#ifdef __cplusplus // C++
		printf("C++ Ver. %ld", __cplusplus);
	#else // __cplusplus
		printf("C");
		#ifdef __STDC_VERSION__ 
			printf("%ld", __STDC_VERSION__);
		#endif // __STDC_VERSION__
	#endif // __cplusplus
	printf("\n");

	//---------------------
	// OS
	// https://sourceforge.net/p/predef/wiki/OperatingSystems/
	// https://sourceforge.net/p/predef/wiki/Standards/
	//---------------------
	#if defined(_WIN32)
		printf("Windows 32bits");
	#elif defined(_WIN64)
		printf("Windows 64bits");
	#elif defined(__linux__)
		printf("Linux ");
		#ifdef __LSB_VERSION__ 
			//printf("%d", __LSB_VERSION__);
		#endif // __LSB_VERSION__
	#elif defined(__APPLE__)
		printf("macOS");
	#else // unknown os
		printf("Unknown OS");
	#endif // defined(_WIN32)
	printf("\n");

	//---------------------
	// Parallelization
	//---------------------	
	#if defined(__AVX2__)
		printf("Using AVX2 ");
	#elif defined(__AVX512F__)
		printf("AVX-512F ");
	#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
		printf("Arm SVE2 ");
	#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
		printf("Arm NEON ");
	#elif defined(_OPENMP)
		printf("OpenMP "); // Ver. %d ", _OPENMP);
		//#ifdef __OMP_H__
			//printf("max %d threads ", omp_get_max_threads());
		//#endif // __OMP_H__
	#endif // defined(__AVX2__)
	printf("\n");

	//---------------------
	// Available precisions
	//---------------------
	printf("Available kinds of precision: ");
		printf("Float ");
		printf("Double ");
		printf("DD ");
		printf("TD ");
		printf("QD ");
	#ifdef USE_GMP
		printf("GMP ");
	#endif // USE_GMP
	#ifdef USE_MPFR
		printf("MPFR ");
	#endif //USE_MPFR
	printf("\n");

	//---------------------
	// Date & Time
	// https://docs.microsoft.com/ja-jp/cpp/preprocessor/predefined-macros?view=msvc-170
	//---------------------	
	printf("Compilation date and time: %s %s\n", __DATE__, __TIME__);

	printf("----------\n");
	return;
}
//#endif // ifndef _BNC_PRINT_ENV_ALL_FUNC

// log2(x)
//#define mylog2(x) (log((double)(x)) / log((double)2.0)) // <-- fix! 2015-06-16(Tue)
double mylog2(double x)
{
	return log10(x) / log10(2.0);
}

// GFlops
double matmul_gflops(double comp_sec, int dim)
{
	return 2.0 * (double)dim * (double)dim * (double)dim / comp_sec / 1024.0 / 1024.0 / 1024.0;
}

// GB of Double prec. Square matrix
int byte_double_sqmat(int dim)
{
	return sizeof(double) * dim * dim;
}

// From BNCpack and mpflinear.c/h
#ifdef USE_GMP
/* set bnc_default_prec */
void set_bnc_default_prec(unsigned long precision)
{
	mpf_set_default_prec(precision);
	bnc_default_prec = precision;
	printf("-------------------------------------------------------------------------------\n");
	printf("BNC Default Precision    : %ld bits(%.1f decimal digits)\n", precision, (double)precision * log10(2.0));
#ifdef USE_MPFR
	bnc_default_rounding_mode = GMP_RNDN; /* round to nearest */
	mpfr_set_default_rounding_mode(bnc_default_rounding_mode);
	printf("BNC Default Rounding Mode: ");
	switch((int)bnc_default_rounding_mode)
	{
		case GMP_RNDN: printf("Round to Nearest\n"); break;
		case GMP_RNDU: printf("Round to Infinity\n"); break;
		case GMP_RNDD: printf("Round to -Infinity\n"); break;
		case GMP_RNDZ: printf("Round to Zero\n"); break;
		default: printf("unknown mode\n");
	}
#endif // USE_MPFR
	printf("-------------------------------------------------------------------------------\n");
}
void _set_bnc_default_prec(unsigned long precision)
{
	mpf_set_default_prec(precision);
	bnc_default_prec = precision;
#ifdef USE_MPFR
	bnc_default_rounding_mode = GMP_RNDN; /* round to nearest */
	mpfr_set_default_rounding_mode(bnc_default_rounding_mode);
#endif // USE_MPFR
}
void set_bnc_default_prec_decimal(unsigned long precision)
{
	unsigned long binary_prec;

	binary_prec = ceil(precision / log10(2.0));
	set_bnc_default_prec(binary_prec);
}

void _set_bnc_default_prec_decimal(unsigned long precision)
{
	unsigned long binary_prec;

	binary_prec = ceil(precision / log10(2.0));
	_set_bnc_default_prec(binary_prec);
}


#ifdef USE_MPFR
mp_rnd_t get_bnc_default_rounding_mode(void)
{
	return bnc_default_rounding_mode;
}

mpc_rnd_t get_bnc_default_rounding_mode_c(void)
{
	return bnc_default_rounding_mode_c;
}


void set_bnc_rounding_mode(mp_rnd_t rounding_mode)
{
	mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode = rounding_mode;
	printf("-------------------------------------------------------------------------------\n");
	printf("BNC Default Rounding Mode: ");
	switch((int)bnc_default_rounding_mode)
	{
		case GMP_RNDN: printf("Round to Nearest\n"); break;
		case GMP_RNDU: printf("Round to Infinity\n"); break;
		case GMP_RNDD: printf("Round to -Infinity\n"); break;
		case GMP_RNDZ: printf("Round to Zero\n"); break;
		default: printf("unknown mode\n");
	}
	printf("-------------------------------------------------------------------------------\n");
}
void _set_bnc_rounding_mode(mp_rnd_t rounding_mode)
{
	mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode = rounding_mode;
}

// for MPC
void set_bnc_rounding_mode_c(mpc_rnd_t rounding_mode)
{
	//mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode_c = rounding_mode;
	printf("-------------------------------------------------------------------------------\n");
	printf("BNC Default Rounding Mode for MPC: ");
	switch((int)bnc_default_rounding_mode_c)
	{
		case MPC_RNDNN: printf("Re: Round to Nearest, Im: Nearest\n"); break;
		case MPC_RNDNZ: printf("Re: Round to Nearest, Im: Zero\n"); break;
		case MPC_RNDNU: printf("Re: Round to Nearest, Im: +Infinity\n"); break;
		case MPC_RNDND: printf("Re: Round to Nearest, Im: -Infinity\n"); break;
		//case MPC_RNDNA: printf("Re: Round to Nearest, Im: Away from zero\n"); break;

		case MPC_RNDZN: printf("Re: Round to Zero, Im: Nearest\n"); break;
		case MPC_RNDZZ: printf("Re: Round to Zero, Im: Zero\n"); break;
		case MPC_RNDZU: printf("Re: Round to Zero, Im: +Infinity\n"); break;
		case MPC_RNDZD: printf("Re: Round to Zero, Im: -Infinity\n"); break;
		//case MPC_RNDZA: printf("Re: Round to Zero, Im: Away from zero\n"); break;

		case MPC_RNDUN: printf("Re: Round to +Infinity, Im: Nearest\n"); break;
		case MPC_RNDUZ: printf("Re: Round to +Infinity, Im: Zero\n"); break;
		case MPC_RNDUU: printf("Re: Round to +Infinity, Im: +Infinity\n"); break;
		case MPC_RNDUD: printf("Re: Round to +Infinity, Im: -Infinity\n"); break;
		//case MPC_RNDUA: printf("Re: Round to +Infinity, Im: Away from zero\n"); break;

		case MPC_RNDDN: printf("Re: Round to -Infinity, Im: Nearest\n"); break;
		case MPC_RNDDZ: printf("Re: Round to -Infinity, Im: Zero\n"); break;
		case MPC_RNDDU: printf("Re: Round to -Infinity, Im: +Infinity\n"); break;
		case MPC_RNDDD: printf("Re: Round to -Infinity, Im: -Infinity\n"); break;
		//case MPC_RNDDA: printf("Re: Round to -Infinity, Im: Away from zero\n"); break;

		//case MPC_RNDAN: printf("Re: Round away from Zero, Im: Nearest\n"); break;
		//case MPC_RNDAZ: printf("Re: Round away from Zero, Im: Zero\n"); break;
		//case MPC_RNDAU: printf("Re: Round away from Zero, Im: +Infinity\n"); break;
		//case MPC_RNDAD: printf("Re: Round away from Zero, Im: -Infinity\n"); break;
		//case MPC_RNDAA: printf("Re: Round away from Zero, Im: Away from zero\n"); break;

		default: printf("unknown mode\n");
	}
	printf("-------------------------------------------------------------------------------\n");
}
void _set_bnc_rounding_mode_c(mpc_rnd_t rounding_mode)
{
	//mpfr_set_default_rounding_mode(rounding_mode);
	bnc_default_rounding_mode_c = rounding_mode;
}
#endif // USE_MPFR

/* get bnc_default_prec */
unsigned long int get_bnc_default_prec(void)
{
	return bnc_default_prec;
}

/* fix!: 2012-07-04 */
/* Jan.26, 2006 */
/* rop := op1 * op2 + op3 */
void mpf_fma(mpf_t rop, mpf_t op1, mpf_t op2, mpf_t op3)
{
#ifdef USE_MPFR
  //#if ( _BNC_MPFR_VER > 201 )
	mpfr_fma(rop, op1, op2, op3, bnc_default_rounding_mode);
  //#else
	//mpfr_t tmp; mpfr_init2(tmp, mpfr_get_prec(rop));
	//mpfr_mul(tmp, op1, op2, bnc_default_rounding_mode);
	//mpfr_add(rop, tmp, op3, bnc_default_rounding_mode);
	//mpfr_clear(tmp);
  //#endif
#else // USE_MPFR
	mpf_t tmp; mpf_init2(tmp, mpf_get_prec(rop));
	mpf_mul(tmp, op1, op2);
	mpf_add(rop, tmp, op3);
	mpf_clear(tmp);
#endif // USE_MPFR
}

// if |a| >  |b|, return +1
//    |a| == |b|, return 0
//    |a| <  |b|, return -1
int mpc_abscmp_ui(mpc_t a, unsigned long b)
{
    int ret;
    mpf_t abs_a;

    mpf_init2(abs_a, mpc_get_prec(a));
    mpc_abs(abs_a, a, get_bnc_default_rounding_mode());
    ret = mpf_cmp_ui(abs_a, b);

    mpf_clear(abs_a);

    return ret;
}
// if |a| >  |b|, return +1
//    |a| == |b|, return 0
//    |a| <  |b|, return -1
int mpc_abscmp(mpc_t a, mpc_t b)
{
    int ret;
    mpf_t abs_a, abs_b;

	// |a|
    mpf_init2(abs_a, mpc_get_prec(a));
    mpc_abs(abs_a, a, get_bnc_default_rounding_mode());
	// |b|
	mpf_init2(abs_b, mpc_get_prec(b));
    mpc_abs(abs_b, b, get_bnc_default_rounding_mode());

	ret = mpf_cmp(abs_a, abs_b);

    mpf_clear(abs_a);
    mpf_clear(abs_b);

	return ret;
}

#endif // USE_GMP

// 2023-04-06(Thu) Random functions
#ifdef USE_GMP

// set random seed & initialize randstate
void mpf_srand(unsigned long seed)
{
	// Initialize
	//gmp_randclear(bnc_default_randstate);
	gmp_randinit_default(bnc_default_randstate);

	// set random seed
	gmp_randseed_ui(bnc_default_randstate, seed);
}

// random number on [0, 1]
void mpf_urand(mpf_t ret)
{
	mpfr_urandom(ret, bnc_default_randstate, get_bnc_default_rounding_mode());
}

// random number following standard normal distribution [-1, 1]
void mpf_nrand(mpf_t ret)
{
	mpfr_nrandom(ret, bnc_default_randstate, get_bnc_default_rounding_mode());
}

// mpc_urand on [0, 1]
void mpc_urand(mpc_t ret)
{
    mpf_urand(mpc_realref(ret));
    mpf_urand(mpc_imagref(ret));
}

// mpc_nrand on [-1, 1]
void mpc_nrand(mpc_t ret)
{
    mpf_nrand(mpc_realref(ret));
    mpf_nrand(mpc_imagref(ret));
}

// 2023-05-31 (Thu) RDD random function based on MPFR
// rdd_srand, rdd_urand, rdd_nrand
void rdd_srand(unsigned long seed)
{
	// Initialize
	//gmp_randclear(bnc_default_randstate);
	gmp_randinit_default(bnc_default_randstate_dd);

	// set random seed
	gmp_randseed_ui(bnc_default_randstate_dd, seed);
}

void rdd_urand(double ret[DDSIZE])
{
	mpfr_t tmp;

	mpfr_init2(tmp, 106);

	//mpf_urand(tmp);
	mpfr_urandom(tmp, bnc_default_randstate_dd, get_bnc_default_rounding_mode());
	mpfr_get_dd(ret, tmp, get_bnc_default_rounding_mode());

	mpfr_clear(tmp);
}
void rdd_nrand(double ret[DDSIZE])
{
	mpf_t tmp;

	mpf_init2(tmp, 106);

	//mpf_nrand(tmp);
	mpfr_nrandom(tmp, bnc_default_randstate_dd, get_bnc_default_rounding_mode());
	mpf_get_dd(ret, tmp);

	mpf_clear(tmp);
}

// 2023-06-01 (Fri) RTD random function based on MPFR
// rtd_srand, rtd_urand, rtd_nrand
void rtd_srand(unsigned long seed)
{
	// Initialize
	//gmp_randclear(bnc_default_randstate);
	gmp_randinit_default(bnc_default_randstate_td);

	// set random seed
	gmp_randseed_ui(bnc_default_randstate_td, seed);
}

void rtd_urand(double ret[TDSIZE])
{
	mpfr_t tmp;

	mpfr_init2(tmp, 159);

	//mpf_urand(tmp);
	mpfr_urandom(tmp, bnc_default_randstate_td, get_bnc_default_rounding_mode());
	mpfr_get_td(ret, tmp, get_bnc_default_rounding_mode());

	mpfr_clear(tmp);
}
void rtd_nrand(double ret[TDSIZE])
{
	mpf_t tmp;

	mpf_init2(tmp, 159);

	//mpf_nrand(tmp);
	mpfr_nrandom(tmp, bnc_default_randstate_td, get_bnc_default_rounding_mode());
	mpf_get_td(ret, tmp);

	mpf_clear(tmp);
}

// 2023-06-01 (Fri) RQD random function based on MPFR
// rqd_srand, rqd_urand, rqd_nrand
void rqd_srand(unsigned long seed)
{
	// Initialize
	//gmp_randclear(bnc_default_randstate);
	gmp_randinit_default(bnc_default_randstate_qd);

	// set random seed
	gmp_randseed_ui(bnc_default_randstate_qd, seed);
}

void rqd_urand(double ret[QDSIZE])
{
	mpfr_t tmp;

	mpfr_init2(tmp, 212);

	//mpf_urand(tmp);
	mpfr_urandom(tmp, bnc_default_randstate_qd, get_bnc_default_rounding_mode());
	mpfr_get_qd(ret, tmp, get_bnc_default_rounding_mode());

	mpfr_clear(tmp);
}
void rqd_nrand(double ret[QDSIZE])
{
	mpf_t tmp;

	mpf_init2(tmp, 212);

	//mpf_nrand(tmp);
	mpfr_nrandom(tmp, bnc_default_randstate_qd, get_bnc_default_rounding_mode());
	mpf_get_qd(ret, tmp);

	mpf_clear(tmp);
}

#if 0
// ret := a * b
void bnc_mpc_mul_cd(mpc_t ret, mpc_t a, double _Complex b)
{
	// 4M method
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp;

	mpf_init2(tmp, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// real part
	mpf_mul_d(ret_re, a_re, creal(b));
	mpf_mul_d(tmp, a_im, cimag(b));
	mpf_sub(ret_re, ret_re, tmp);

	// imag part
	mpf_mul_d(ret_im, a_re, cimag(b));
	mpf_mul_d(tmp, a_im, creal(b));
	mpf_add(ret_im, ret_im, tmp);

	mpf_clear(tmp);
}
#endif //0

// mixed precision arithmetic
// that are not priviced by GNU and MPFR
// ---
// DD
// ---
// Real
void mpf_add_dd(mpf_t ret, mpf_t a, double b[DDSIZE])
{
#ifdef USE_MPFR
	// ret := a + (b[0] + b[1])
	mpfr_add_d(ret, a, b[0], get_bnc_default_rounding_mode());
	mpfr_add_d(ret, ret, b[1], get_bnc_default_rounding_mode());
#else // USE_MPFR
	mpf_t in_b;

	mpf_init2(in_b, mpf_get_prec(ret));

	mpf_get_dd(in_b, b);
	mpf_add(ret, a, in_b);
	mpf_clear(in_b);
#endif // USE_MPFR
}
void mpf_sub_dd(mpf_t ret, mpf_t a, double b[DDSIZE])
{
#ifdef USE_MPFR
	// ret := a - (b[0] + b[1])
	mpfr_sub_d(ret, a, b[0], get_bnc_default_rounding_mode());
	mpfr_sub_d(ret, ret, b[1], get_bnc_default_rounding_mode());
#else // USE_MPFR
	mpf_t in_b;

	mpf_init2(in_b, mpf_get_prec(ret));

	mpf_get_dd(in_b, b);
	mpf_sub(ret, a, in_b);
	mpf_clear(in_b);
#endif // USE_MPFR
}
void mpf_mul_dd(mpf_t ret, mpf_t a, double b[DDSIZE])
{
#ifdef USE_MPFR
	mpfr_t tmp;

	mpfr_init2(tmp, mpfr_get_prec(ret));

	// ret := a * (b[0] + b[1])
	mpfr_mul_d(ret , a, b[0], get_bnc_default_rounding_mode());
	mpfr_mul_d(tmp, a, b[1], get_bnc_default_rounding_mode());
	mpfr_add(ret, ret, tmp, get_bnc_default_rounding_mode());

	mpfr_clear(tmp);
#else // USE_MPFR
	mpf_t in_b;

	mpf_init2(in_b, DDSIZE * 53);

	mpf_get_dd(in_b, b);
	mpf_mul(ret, a, in_b);
	mpf_clear(in_b);
#endif // USE_MPFR
}
void mpf_div_dd(mpf_t ret, mpf_t a, double b[DDSIZE])
{
#ifdef USE_MPFR
	mpfr_t tmp;

	mpfr_init2(tmp, DDSIZE * 53); // mpfr_get_prec(ret));

	// ret := a / (b[0] + b[1])
	mpfr_set_d(tmp, b[0], get_bnc_default_rounding_mode());
	mpfr_add_d(tmp, tmp, b[1], get_bnc_default_rounding_mode());
	mpfr_div(ret, a, tmp, get_bnc_default_rounding_mode());

	mpfr_clear(tmp);

#else // USE_MPFR
	mpf_t in_b;

	mpf_init2(in_b, DDSIZE * 53);

	mpf_get_dd(in_b, b);
	mpf_div(ret, a, in_b);
	mpf_clear(in_b);
#endif // USE_MPFR
}

// Complex
void mpc_add_cdd(mpc_t ret, mpc_t a, cddfloat *b)
{
	// ret := a + b
	mpf_add_dd(mpc_realref(ret), mpc_realref(a), b->val_re);
	mpf_add_dd(mpc_imagref(ret), mpc_imagref(a), b->val_im);
}
void mpc_sub_cdd(mpc_t ret, mpc_t a, cddfloat *b)
{
	// ret := a - b
	mpf_add_dd(mpc_realref(ret), mpc_realref(a), b->val_re);
	mpf_add_dd(mpc_imagref(ret), mpc_imagref(a), b->val_im);

}
void mpc_mul_cdd(mpc_t ret, mpc_t a, cddfloat *b)
{
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp;

	mpf_init2(tmp, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// real part
	mpf_mul_dd(ret_re, a_re, b->val_re);
	mpf_mul_dd(   tmp, a_im, b->val_im);
	mpf_sub(ret_re, ret_re, tmp);

	// imag part
	mpf_mul_dd(ret_im, a_re, b->val_im);
	mpf_mul_dd(   tmp, a_im, b->val_re);
	mpf_add(ret_im, ret_im, tmp);

	mpf_clear(tmp);
}
void mpc_div_cdd(mpc_t ret, mpc_t a, cddfloat *b)
{
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp, abs_b2;

	mpf_init2(tmp, mpc_get_prec(ret));
	mpf_init2(abs_b2, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// real part
	mpf_mul_dd(ret_re, a_re, b->val_re);
	mpf_mul_dd(   tmp, a_im, b->val_im);
	mpf_add(ret_re, ret_re, tmp);

	// imag part
	mpf_mul_dd(ret_im, a_re, b->val_im);
	mpf_mul_dd(   tmp, a_im, b->val_re);
	mpf_sub(ret_im, ret_im, tmp);

	// |b|^2
	mpf_set_dd(abs_b2, b->val_re);
	mpf_mul_dd(abs_b2, abs_b2, b->val_re);
	mpf_set_dd(tmp, b->val_im);
	mpf_mul_dd(tmp, tmp, b->val_im);
	mpf_add(abs_b2, abs_b2, tmp);

	// a * conj(b) / |b|^2
	mpf_div(ret_re, ret_re, abs_b2);
	mpf_div(ret_im, ret_im, abs_b2);

	mpf_clear(tmp);
	mpf_clear(abs_b2);
}


#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
