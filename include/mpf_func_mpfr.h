/********************************************************************************/
/* mpf_func_mpfr.h: Definision of mpf_* functions from mpfr functions           */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
#ifndef __BNC_MPF_FUNC_MPFR_H
#define __BNC_MPF_FUNC_MPFR_H

// defined in mpf2mpfr.h
/* types */
//#define mpf_t mpfr_t
//#define mpf_srcptr mpfr_srcptr
//#define mpf_ptr mpfr_ptr

/* Get current Rounding Mode */
#ifndef MPFR_DEFAULT_RND
    #define MPFR_DEFAULT_RND mpfr_get_default_rounding_mode()
#endif // MPFR_DEFAULT_RND

// mpf_[add,sub,mul,div]_d
// x := y [+-*/] d
#undef mpf_add_d
#define mpf_add_d(x, y, d) mpfr_add_d(x, y, d, MPFR_DEFAULT_RND)
#undef mpf_sub_d
#define mpf_sub_d(x, y, d) mpfr_sub_d(x, y, d, MPFR_DEFAULT_RND)
#undef mpf_mul_d
#define mpf_mul_d(x, y, d) mpfr_mul_d(x, y, d, MPFR_DEFAULT_RND)
#undef mpf_div_d
#define mpf_div_d(x, y, d) mpfr_div_d(x, y, d, MPFR_DEFAULT_RND)

// mul_d_[sub,div]
// x := d [-/] y
#undef mpf_d_sub
#define mpf_d_sub(x, d, y) mpfr_d_sub(x, d, y, MPFR_DEFAULT_RND)
#undef mpf_d_div
#define mpf_d_div(x, d, y) mpfr_d_div(x, d, y, MPFR_DEFAULT_RND)

#endif  // __BNC_MPF_FUNC_MPFR_H
