/********************************************************************************/
/* mpfr_dd.h:                                                                   */
/* Copyright (C) 2016 Tomonori Kouya                                            */
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
#ifndef __MPFR_DD_H_
#define __MPFR_DD_H_

// LONG_MIN, LONG_MAX
#include <limits.h>

// _GMP_IEEE_FLOATS
#ifndef _GMP_IEEE_FLOATS
//#define _GMP_IEEE_FLOATS
#include "ieee_floats.h"
#endif // _GMP_IEEE_FLOATS

// from mpfr-gmp.h
/* Use (4.0 * ...) instead of (2.0 * ...) to work around buggy compilers
   that don't convert ulong->double correctly (eg. SunOS 4 native cc).  */
#undef MP_BASE_AS_DOUBLE
#define MP_BASE_AS_DOUBLE (4.0 * ((mp_limb_t) 1 << (GMP_NUMB_BITS - 2)))

// from mpfr-impl.h
/******************************************************
 ************* GMP Basic Pointer Types ****************
 ******************************************************/

typedef mp_limb_t *mpfr_limb_ptr;
typedef __gmp_const mp_limb_t *mpfr_limb_srcptr;

// from mpfr-gmp.h
unsigned int __gmpfr_flags;
mpfr_exp_t   __gmpfr_emin;
mpfr_exp_t   __gmpfr_emax;
mpfr_prec_t  __gmpfr_default_fp_bit_precision;
mpfr_rnd_t   __gmpfr_default_rounding_mode;

/* Flags of __gmpfr_flags */
#define MPFR_FLAGS_UNDERFLOW 1
#define MPFR_FLAGS_OVERFLOW 2
#define MPFR_FLAGS_NAN 4
#define MPFR_FLAGS_INEXACT 8
#define MPFR_FLAGS_ERANGE 16
#define MPFR_FLAGS_DIVBY0 32
#define MPFR_FLAGS_ALL 63

/* Replace some common functions for direct access to the global vars */
#define mpfr_get_emin() (__gmpfr_emin + 0)
#define mpfr_get_emax() (__gmpfr_emax + 0)
#define mpfr_get_default_rounding_mode() (__gmpfr_default_rounding_mode + 0)
#define mpfr_get_default_prec() (__gmpfr_default_fp_bit_precision + 0)

#define mpfr_clear_flags() \
  ((void) (__gmpfr_flags = 0))
#define mpfr_clear_underflow() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_UNDERFLOW))
#define mpfr_clear_overflow() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_OVERFLOW))
#define mpfr_clear_nanflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_NAN))
#define mpfr_clear_inexflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_INEXACT))
#define mpfr_clear_erangeflag() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_ERANGE))
#define mpfr_clear_divby0() \
  ((void) (__gmpfr_flags &= MPFR_FLAGS_ALL ^ MPFR_FLAGS_DIVBY0))
#define mpfr_underflow_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_UNDERFLOW))
#define mpfr_overflow_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_OVERFLOW))
#define mpfr_nanflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_NAN))
#define mpfr_inexflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_INEXACT))
#define mpfr_erangeflag_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_ERANGE))
#define mpfr_divby0_p() \
  ((int) (__gmpfr_flags & MPFR_FLAGS_DIVBY0))

/* Testing an exception flag correctly is tricky. There are mainly two
   pitfalls: First, one needs to remember to clear the corresponding
   flag, in case it was set before the function call or during some
   intermediate computations (in practice, one can clear all the flags).
   Secondly, one needs to test the flag early enough, i.e. before it
   can be modified by another function. Moreover, it is quite difficult
   (if not impossible) to reliably check problems with "make check". To
   avoid these pitfalls, it is recommended to use the following macros.
   Other use of the exception-flag predicate functions/macros will be
   detected by mpfrlint.
   Note: _op can be either a statement or an expression.
   MPFR_BLOCK_EXCEP should be used only inside a block; it is useful to
   detect some exception in order to exit the block as soon as possible. */
#define MPFR_BLOCK_DECL(_flags) unsigned int _flags
/* The (void) (_flags) makes sure that _flags is read at least once (it
   makes sense to use MPFR_BLOCK while _flags will never be read in the
   source, so that we wish to avoid the corresponding warning). */
#define MPFR_BLOCK(_flags,_op)          \
  do                                    \
    {                                   \
      mpfr_clear_flags ();              \
      _op;                              \
      (_flags) = __gmpfr_flags;         \
      (void) (_flags);                  \
    }                                   \
  while (0)
#define MPFR_BLOCK_TEST(_flags,_f) MPFR_UNLIKELY ((_flags) & (_f))
#define MPFR_BLOCK_EXCEP (__gmpfr_flags & (MPFR_FLAGS_UNDERFLOW | \
                                           MPFR_FLAGS_OVERFLOW | \
                                           MPFR_FLAGS_DIVBY0 | \
                                           MPFR_FLAGS_NAN))
/* Let's use a MPFR_ prefix, because e.g. OVERFLOW is defined by glibc's
   math.h, though this is not a reserved identifier! */
#define MPFR_UNDERFLOW(_flags)  MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_UNDERFLOW)
#define MPFR_OVERFLOW(_flags)   MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_OVERFLOW)
#define MPFR_NANFLAG(_flags)    MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_NAN)
#define MPFR_INEXFLAG(_flags)   MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_INEXACT)
#define MPFR_ERANGEFLAG(_flags) MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_ERANGE)
#define MPFR_DIVBY0(_flags)     MPFR_BLOCK_TEST (_flags, MPFR_FLAGS_DIVBY0)


/******************************************************
 ******************** Assertions **********************
 ******************************************************/

/* Compile with -DWANT_ASSERT to check all assert statements */

/* Note: do not use GMP macros ASSERT_ALWAYS and ASSERT as they are not
   expressions, and as a consequence, they cannot be used in a for(),
   with a comma operator and so on. */

/* MPFR_ASSERTN(expr): assertions that should always be checked */
#define MPFR_ASSERTN(expr)  \
   ((void) ((MPFR_UNLIKELY(expr)) || MPFR_UNLIKELY( (ASSERT_FAIL(expr),0) )))

/* MPFR_ASSERTD(expr): assertions that should be checked when testing */
#ifdef WANT_ASSERT
# define MPFR_EXP_CHECK 1
# define MPFR_ASSERTD(expr)  MPFR_ASSERTN (expr)
#else
# define MPFR_ASSERTD(expr)  ((void) 0)
#endif

/* Code to deal with impossible
   WARNING: It doesn't use do { } while (0) for Insure++*/
#define MPFR_RET_NEVER_GO_HERE()  {MPFR_ASSERTN(0); return 0;}


/******************************************************
 ******************** Warnings ************************
 ******************************************************/

/* MPFR_WARNING is no longer useful, but let's keep the macro in case
   it needs to be used again in the future. */

#ifdef MPFR_USE_WARNINGS
# include <stdlib.h>
# define MPFR_WARNING(W)                    \
  do                                        \
    {                                       \
      char *q = getenv ("MPFR_QUIET");      \
      if (q == NULL || *q == 0)             \
        fprintf (stderr, "MPFR: %s\n", W);  \
    }                                       \
  while (0)
#else
# define MPFR_WARNING(W)  ((void) 0)
#endif


/******************************************************
 ****************** double macros *********************
 ******************************************************/

/* Precision used for lower precision computations */
#define MPFR_SMALL_PRECISION 32

/* Definition of constants */
#define LOG2 0.69314718055994528622 /* log(2) rounded to zero on 53 bits */
#define ALPHA 4.3191365662914471407 /* a+2 = a*log(a), rounded to +infinity */
#define EXPM1 0.36787944117144227851 /* exp(-1), rounded to zero */

/* MPFR_DOUBLE_SPEC = 1 if the C type 'double' corresponds to IEEE-754
   double precision, 0 if it doesn't, and undefined if one doesn't know.
   On all the tested machines, MPFR_DOUBLE_SPEC = 1. To have this macro
   defined here, #include <float.h> is needed. If need be, other values
   could be defined for other specs (once they are known). */
#if !defined(MPFR_DOUBLE_SPEC) && defined(FLT_RADIX) && \
    defined(DBL_MANT_DIG) && defined(DBL_MIN_EXP) && defined(DBL_MAX_EXP)
# if FLT_RADIX == 2 && DBL_MANT_DIG == 53 && \
     DBL_MIN_EXP == -1021 && DBL_MAX_EXP == 1024
#  define MPFR_DOUBLE_SPEC 1
# else
#  define MPFR_DOUBLE_SPEC 0
# endif
#endif

/* Debug non IEEE floats */
#ifdef XDEBUG
# undef _GMP_IEEE_FLOATS
#endif
#ifndef _GMP_IEEE_FLOATS
# define _GMP_IEEE_FLOATS 0
#endif

#ifndef IEEE_DBL_MANT_DIG
#define IEEE_DBL_MANT_DIG 53
#endif
#define MPFR_LIMBS_PER_DOUBLE ((IEEE_DBL_MANT_DIG-1)/GMP_NUMB_BITS+1)

#ifndef IEEE_FLT_MANT_DIG
#define IEEE_FLT_MANT_DIG 24
#endif
#define MPFR_LIMBS_PER_FLT ((IEEE_FLT_MANT_DIG-1)/GMP_NUMB_BITS+1)

/* Visual C++ doesn't support +1.0/0.0, -1.0/0.0 and 0.0/0.0
   at compile time. */
#if defined(_MSC_VER) && defined(_WIN32) && (_MSC_VER >= 1200)
static double double_zero = 0.0;
# define DBL_NAN (double_zero/double_zero)
# define DBL_POS_INF ((double) 1.0/double_zero)
# define DBL_NEG_INF ((double)-1.0/double_zero)
# define DBL_NEG_ZERO (-double_zero)
#else
# define DBL_POS_INF ((double) 1.0/0.0)
# define DBL_NEG_INF ((double)-1.0/0.0)
# define DBL_NAN     ((double) 0.0/0.0)
# define DBL_NEG_ZERO (-0.0)
#endif

/* Note: In the past, there was specific code for _GMP_IEEE_FLOATS, which
   was based on NaN and Inf memory representations. This code was breaking
   the aliasing rules (see ISO C99, 6.5#6 and 6.5#7 on the effective type)
   and for this reason it did not behave correctly with GCC 4.5.0 20091119.
   The code needed a memory transfer and was probably not better than the
   macros below with a good compiler (a fix based on the NaN / Inf memory
   representation would be even worse due to C limitations), and this code
   could be selected only when MPFR was built with --with-gmp-build, thus
   introducing a difference (bad for maintaining/testing MPFR); therefore
   it has been removed. The old code required that the argument x be an
   lvalue of type double. We still require that, in case one would need
   to change the macros below, e.g. for some broken compiler. But the
   LVALUE(x) condition could be removed if really necessary. */
/* Below, the &(x) == &(x) || &(x) != &(x) allows to make sure that x
   is a lvalue without (probably) any warning from the compiler.  The
   &(x) != &(x) is needed to avoid a failure under Mac OS X 10.4.11
   (with Xcode 2.4.1, i.e. the latest one). */
#define LVALUE(x) (&(x) == &(x) || &(x) != &(x))
#define DOUBLE_ISINF(x) (LVALUE(x) && ((x) > DBL_MAX || (x) < -DBL_MAX))
#ifdef MPFR_NANISNAN
/* Avoid MIPSpro / IRIX64 / gcc -ffast-math (incorrect) optimizations.
   The + must not be replaced by a ||. With gcc -ffast-math, NaN is
   regarded as a positive number or something like that; the second
   test catches this case. */
# define DOUBLE_ISNAN(x) \
    (LVALUE(x) && !((((x) >= 0.0) + ((x) <= 0.0)) && -(x)*(x) <= 0.0))
#else
# define DOUBLE_ISNAN(x) (LVALUE(x) && (x) != (x))
#endif

/******************************************************
 **************** mpfr_t properties *******************
 ******************************************************/

/* In the following macro, p is usually a mpfr_prec_t, but this macro
   works with other integer types (without integer overflow). Checking
   that p >= 1 in debug mode is useful here because this macro can be
   used on a computed precision (in particular, this formula does not
   work for a degenerate case p = 0, and could give different results
   on different platforms). But let us not use an assertion checking
   in the MPFR_LAST_LIMB() and MPFR_LIMB_SIZE() macros below to avoid
   too much expansion for assertions (in practice, this should be a
   problem just when testing MPFR with the --enable-assert configure
   option and the -ansi -pedantic-errors gcc compiler flags). */
#define MPFR_PREC2LIMBS(p) \
  (MPFR_ASSERTD ((p) >= 1), ((p) - 1) / GMP_NUMB_BITS + 1)

#define MPFR_PREC(x)      ((x)->_mpfr_prec)
#define MPFR_EXP(x)       ((x)->_mpfr_exp)
#define MPFR_MANT(x)      ((x)->_mpfr_d)
#define MPFR_LAST_LIMB(x) ((MPFR_PREC (x) - 1) / GMP_NUMB_BITS)
#define MPFR_LIMB_SIZE(x) (MPFR_LAST_LIMB (x) + 1)


/******************************************************
 **************** exponent properties *****************
 ******************************************************/

/* Limits of the mpfr_exp_t type (NOT those of valid exponent values).
   These macros can be used in preprocessor directives. */
#if   _MPFR_EXP_FORMAT == 1
# define MPFR_EXP_MAX (SHRT_MAX)
# define MPFR_EXP_MIN (SHRT_MIN)
#elif _MPFR_EXP_FORMAT == 2
# define MPFR_EXP_MAX (INT_MAX)
# define MPFR_EXP_MIN (INT_MIN)
#elif _MPFR_EXP_FORMAT == 3
# define MPFR_EXP_MAX (LONG_MAX)
# define MPFR_EXP_MIN (LONG_MIN)
#elif _MPFR_EXP_FORMAT == 4
# define MPFR_EXP_MAX (MPFR_INTMAX_MAX)
# define MPFR_EXP_MIN (MPFR_INTMAX_MIN)
#else
# error "Invalid MPFR Exp format"
#endif

/* Before doing a cast to mpfr_uexp_t, make sure that the value is
   nonnegative. */
#define MPFR_UEXP(X) (MPFR_ASSERTD ((X) >= 0), (mpfr_uexp_t) (X))

#if MPFR_EXP_MIN >= LONG_MIN && MPFR_EXP_MAX <= LONG_MAX
typedef long int mpfr_eexp_t;
# define mpfr_get_exp_t(x,r) mpfr_get_si((x),(r))
# define mpfr_set_exp_t(x,e,r) mpfr_set_si((x),(e),(r))
# define MPFR_EXP_FSPEC "l"
#elif defined (_MPFR_H_HAVE_INTMAX_T)
typedef intmax_t mpfr_eexp_t;
# define mpfr_get_exp_t(x,r) mpfr_get_sj((x),(r))
# define mpfr_set_exp_t(x,e,r) mpfr_set_sj((x),(e),(r))
# define MPFR_EXP_FSPEC "j"
#else
# error "Cannot define mpfr_get_exp_t and mpfr_set_exp_t"
#endif

/* Invalid exponent value (to track bugs...) */
#define MPFR_EXP_INVALID \
 ((mpfr_exp_t) 1 << (GMP_NUMB_BITS*sizeof(mpfr_exp_t)/sizeof(mp_limb_t)-2))

/* Definition of the exponent limits for MPFR numbers.
 * These limits are chosen so that if e is such an exponent, then 2e-1 and
 * 2e+1 are representable. This is useful for intermediate computations,
 * in particular the multiplication.
 */
#undef MPFR_EMIN_MIN
#undef MPFR_EMIN_MAX
#undef MPFR_EMAX_MIN
#undef MPFR_EMAX_MAX
#define MPFR_EMIN_MIN (1-MPFR_EXP_INVALID)
#define MPFR_EMIN_MAX (MPFR_EXP_INVALID-1)
#define MPFR_EMAX_MIN (1-MPFR_EXP_INVALID)
#define MPFR_EMAX_MAX (MPFR_EXP_INVALID-1)

/* Use MPFR_GET_EXP and MPFR_SET_EXP instead of MPFR_EXP directly,
   unless when the exponent may be out-of-range, for instance when
   setting the exponent before calling mpfr_check_range.
   MPFR_EXP_CHECK is defined when WANT_ASSERT is defined, but if you
   don't use WANT_ASSERT (for speed reasons), you can still define
   MPFR_EXP_CHECK by setting -DMPFR_EXP_CHECK in $CFLAGS. */

#ifdef MPFR_EXP_CHECK
# define MPFR_GET_EXP(x)          (mpfr_get_exp) (x)
# define MPFR_SET_EXP(x, exp)     MPFR_ASSERTN (!mpfr_set_exp ((x), (exp)))
# define MPFR_SET_INVALID_EXP(x)  ((void) (MPFR_EXP (x) = MPFR_EXP_INVALID))
#else
# define MPFR_GET_EXP(x)          MPFR_EXP (x)
# define MPFR_SET_EXP(x, exp)     ((void) (MPFR_EXP (x) = (exp)))
# define MPFR_SET_INVALID_EXP(x)  ((void) 0)
#endif



/******************************************************
 ********** Singular Values (NAN, INF, ZERO) **********
 ******************************************************/

/* Enum special value of exponent. */
# define MPFR_EXP_ZERO (MPFR_EXP_MIN+1)
# define MPFR_EXP_NAN  (MPFR_EXP_MIN+2)
# define MPFR_EXP_INF  (MPFR_EXP_MIN+3)

#define MPFR_IS_NAN(x)   (MPFR_EXP(x) == MPFR_EXP_NAN)
#define MPFR_SET_NAN(x)  (MPFR_EXP(x) =  MPFR_EXP_NAN)
#define MPFR_IS_INF(x)   (MPFR_EXP(x) == MPFR_EXP_INF)
#define MPFR_SET_INF(x)  (MPFR_EXP(x) =  MPFR_EXP_INF)
#define MPFR_IS_ZERO(x)  (MPFR_EXP(x) == MPFR_EXP_ZERO)
#define MPFR_SET_ZERO(x) (MPFR_EXP(x) =  MPFR_EXP_ZERO)
#define MPFR_NOTZERO(x)  (MPFR_EXP(x) != MPFR_EXP_ZERO)

#define MPFR_IS_FP(x)       (!MPFR_IS_NAN(x) && !MPFR_IS_INF(x))
#define MPFR_IS_SINGULAR(x) (MPFR_EXP(x) <= MPFR_EXP_INF)
#define MPFR_IS_PURE_FP(x)  (!MPFR_IS_SINGULAR(x) && \
  (MPFR_ASSERTD ((MPFR_MANT(x)[MPFR_LAST_LIMB(x)]  \
                  & MPFR_LIMB_HIGHBIT) != 0), 1))

#define MPFR_ARE_SINGULAR(x,y) \
  (MPFR_UNLIKELY(MPFR_IS_SINGULAR(x)) || MPFR_UNLIKELY(MPFR_IS_SINGULAR(y)))

#define MPFR_IS_POWER_OF_2(x) \
  (mpfr_cmp_ui_2exp ((x), 1, MPFR_GET_EXP (x) - 1) == 0)


/******************************************************
 ********************* Sign Macros ********************
 ******************************************************/

#define MPFR_SIGN_POS (1)
#define MPFR_SIGN_NEG (-1)

#define MPFR_IS_STRICTPOS(x)  (MPFR_NOTZERO((x)) && MPFR_SIGN(x) > 0)
#define MPFR_IS_STRICTNEG(x)  (MPFR_NOTZERO((x)) && MPFR_SIGN(x) < 0)

#define MPFR_IS_NEG(x) (MPFR_SIGN(x) < 0)
#define MPFR_IS_POS(x) (MPFR_SIGN(x) > 0)

#define MPFR_SET_POS(x) (MPFR_SIGN(x) = MPFR_SIGN_POS)
#define MPFR_SET_NEG(x) (MPFR_SIGN(x) = MPFR_SIGN_NEG)

#define MPFR_CHANGE_SIGN(x) (MPFR_SIGN(x) = -MPFR_SIGN(x))
#define MPFR_SET_SAME_SIGN(x, y) (MPFR_SIGN(x) = MPFR_SIGN(y))
#define MPFR_SET_OPPOSITE_SIGN(x, y) (MPFR_SIGN(x) = -MPFR_SIGN(y))
#define MPFR_ASSERT_SIGN(s) \
 (MPFR_ASSERTD((s) == MPFR_SIGN_POS || (s) == MPFR_SIGN_NEG))
#define MPFR_SET_SIGN(x, s) \
  (MPFR_ASSERT_SIGN(s), MPFR_SIGN(x) = s)
#define MPFR_IS_POS_SIGN(s1) (s1 > 0)
#define MPFR_IS_NEG_SIGN(s1) (s1 < 0)
#define MPFR_MULT_SIGN(s1, s2) ((s1) * (s2))
/* Transform a sign to 1 or -1 */
#define MPFR_FROM_SIGN_TO_INT(s) (s)
#define MPFR_INT_SIGN(x) MPFR_FROM_SIGN_TO_INT(MPFR_SIGN(x))



/******************************************************
 ***************** Ternary Value Macros ***************
 ******************************************************/

/* Special inexact value */
#define MPFR_EVEN_INEX 2

/* Macros for functions returning two inexact values in an 'int' */
#define INEXPOS(y) ((y) == 0 ? 0 : (((y) > 0) ? 1 : 2))
#define INEX(y,z) (INEXPOS(y) | (INEXPOS(z) << 2))

/* When returning the ternary inexact value, ALWAYS use one of the
   following two macros, unless the flag comes from another function
   returning the ternary inexact value */
#define MPFR_RET(I) return \
  (I) ? ((__gmpfr_flags |= MPFR_FLAGS_INEXACT), (I)) : 0
#define MPFR_RET_NAN return (__gmpfr_flags |= MPFR_FLAGS_NAN), 0

#define MPFR_SET_ERANGE() (__gmpfr_flags |= MPFR_FLAGS_ERANGE)

#define SIGN(I) ((I) < 0 ? -1 : (I) > 0)
#define SAME_SIGN(I1,I2) (SIGN (I1) == SIGN (I2))

// from mpfr-impl.h
/******************************************************
 **************  Save exponent macros  ****************
 ******************************************************/

/* See README.dev for details on how to use the macros.
   They are used to set the exponent range to the maximum
   temporarily */

typedef struct {
  unsigned int saved_flags;
  mpfr_exp_t saved_emin;
  mpfr_exp_t saved_emax;
} mpfr_save_expo_t;

/* Minimum and maximum exponents of the extended exponent range. */
#define MPFR_EXT_EMIN MPFR_EMIN_MIN
#define MPFR_EXT_EMAX MPFR_EMAX_MAX

#define MPFR_SAVE_EXPO_DECL(x) mpfr_save_expo_t x
#define MPFR_SAVE_EXPO_MARK(x)     \
 ((x).saved_flags = __gmpfr_flags, \
  (x).saved_emin = __gmpfr_emin,   \
  (x).saved_emax = __gmpfr_emax,   \
  __gmpfr_emin = MPFR_EXT_EMIN,    \
  __gmpfr_emax = MPFR_EXT_EMAX)
#define MPFR_SAVE_EXPO_FREE(x)     \
 (__gmpfr_flags = (x).saved_flags, \
  __gmpfr_emin = (x).saved_emin,   \
  __gmpfr_emax = (x).saved_emax)
#define MPFR_SAVE_EXPO_UPDATE_FLAGS(x, flags)  \
  (x).saved_flags |= (flags)

/* Speed up final checking */
#define mpfr_check_range(x,t,r) \
 (MPFR_LIKELY (MPFR_EXP (x) >= __gmpfr_emin && MPFR_EXP (x) <= __gmpfr_emax) \
  ? ((t) ? (__gmpfr_flags |= MPFR_FLAGS_INEXACT, (t)) : 0)                   \
  : mpfr_check_range(x,t,r))

/* extracts the bits of d in rp[0..n-1] where n=ceil(53/GMP_NUMB_BITS).
   Assumes d is neither 0 nor NaN nor Inf. */
long
__gmpfr_extract_dd (mpfr_limb_ptr rp, double d)
     /* e=0 iff GMP_NUMB_BITS=32 and rp has only one limb */
{
  long exp;
  mp_limb_t manl;
#if GMP_NUMB_BITS == 32
  mp_limb_t manh;
#endif

  /* BUGS
     1. Should handle Inf and NaN in IEEE specific code.
     2. Handle Inf and NaN also in default code, to avoid hangs.
     3. Generalize to handle all GMP_NUMB_BITS.
     4. This lits is incomplete and misspelled.
   */

  MPFR_ASSERTD(!DOUBLE_ISNAN(d));
  MPFR_ASSERTD(!DOUBLE_ISINF(d));
  MPFR_ASSERTD(d != 0.0);

#if _GMP_IEEE_FLOATS

  {
    union ieee_double_extract x;
    x.d = d;

    exp = x.s.exp;
    if (exp)
      {
#if GMP_NUMB_BITS >= 64
        manl = ((MPFR_LIMB_ONE << 63)
                | ((mp_limb_t) x.s.manh << 43) | ((mp_limb_t) x.s.manl << 11));
#else
        manh = (MPFR_LIMB_ONE << 31) | (x.s.manh << 11) | (x.s.manl >> 21);
        manl = x.s.manl << 11;
#endif
      }
    else /* denormalized number */
      {
#if GMP_NUMB_BITS >= 64
        manl = ((mp_limb_t) x.s.manh << 43) | ((mp_limb_t) x.s.manl << 11);
#else
        manh = (x.s.manh << 11) /* high 21 bits */
          | (x.s.manl >> 21); /* middle 11 bits */
        manl = x.s.manl << 11; /* low 21 bits */
#endif
      }

    if (exp)
      exp -= 1022;
    else
      exp = -1021;
  }

#else /* _GMP_IEEE_FLOATS */

  {
    /* Unknown (or known to be non-IEEE) double format.  */
    exp = 0;
    if (d >= 1.0)
      {
        MPFR_ASSERTN (d * 0.5 != d);
        while (d >= 32768.0)
          {
            d *= (1.0 / 65536.0);
            exp += 16;
          }
        while (d >= 1.0)
          {
            d *= 0.5;
            exp += 1;
          }
      }
    else if (d < 0.5)
      {
        while (d < (1.0 / 65536.0))
          {
            d *=  65536.0;
            exp -= 16;
          }
        while (d < 0.5)
          {
            d *= 2.0;
            exp -= 1;
          }
      }

    d *= MP_BASE_AS_DOUBLE;
#if GMP_NUMB_BITS >= 64
    manl = d;
#else
    manh = (mp_limb_t) d;
    manl = (mp_limb_t) ((d - manh) * MP_BASE_AS_DOUBLE);
#endif
  }

#endif /* _GMP_IEEE_FLOATS */

#if GMP_NUMB_BITS >= 64
  rp[0] = manl;
#else
  rp[1] = manh;
  rp[0] = manl;
#endif

  return exp;
}

/* End of part included from gmp-2.0.2 */


#endif // __MPFR_DD_H_
