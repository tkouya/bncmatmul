// -------------------------------------------------------
// _bncsve2_ts.h  --  SVE2 Triple-Single (TS) per-limb-variable helpers
// -------------------------------------------------------
// Rewritten 2026.  TS = three svfloat32_t.  All TS helpers (set0, set_f,
// neg, addq, mulq) are defined in _bncsve2_qs.h because they need
// frenorm/fthree_sum, which live there.  This file only carries the size
// macro; the user MUST also #include _bncsve2_qs.h.
// -------------------------------------------------------
#ifndef __BNCSVE2_TS_H
#define __BNCSVE2_TS_H

#if defined(__ARM_SVE2)
#include <arm_sve.h>

#ifndef TSSIZE
    #define TSSIZE 3
#endif

#endif // defined(__ARM_SVE2)
#endif // __BNCSVE2_TS_H
