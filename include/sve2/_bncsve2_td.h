// -------------------------------------------------------
// _bncsve2_td.h  --  SVE2 Triple-Double (TD) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026.  A TD value is THREE separate svfloat64_t variables.
// Mirrors include/neon/_bncneon_td.h's structure: the renorm-based
// _bncsve2_rtd_addq / _bncsve2_rtd_mulq are defined in _bncsve2_qd.h
// (they need renorm, which lives there).  This file declares only the
// set/neg helpers; the user must #include _bncsve2_qd.h to get add/mul.
// -------------------------------------------------------
#ifndef __BNCSVE2_TD_H
#define __BNCSVE2_TD_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef TDSIZE
    #define TDSIZE 3
#endif

/* set0 / set_d / neg are defined alongside addq/mulq in _bncsve2_qd.h to
 * keep the QD-dependent helpers (renorm, three_sum) and TD helpers in one
 * compilation unit.  Including this header just brings in the size macro
 * and forward declarations; the user MUST also #include _bncsve2_qd.h to
 * call _bncsve2_rtd_{addq,mulq,set0,set_d,neg}. */

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_TD_H
