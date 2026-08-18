/********************************************************************************/
/* bncfma.h : Branch-free fused multiply-add (FMA) for DW/TW/QW arithmetic       */
/*            z := x * y + c                                                    */
/*                                                                              */
/* Implementation of Algorithms 1-3 of                                          */
/*   T. Kouya, "Performance evaluation of branch-free fused multiply-add        */
/*   algorithms for multi-component-type multiple-precision floating-point      */
/*   arithmetic", arXiv:2607.11391v1 (2026).                                    */
/* Operation-by-operation port of the reference implementation fma_ref.c        */
/* (Appendix of the above), so every routine matches the FPANVerifier-certified */
/* netlists dwfma_f2s.fpan / twfma_fix2.fpan / qwfma_fix3.fpan.                 */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
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
/*
 * IMPORTANT: compile with -ffp-contract=off.  If the compiler is allowed to
 * contract a*b+c into an fma on its own, two_sum / two_prod stop being
 * error-free transformations and the verified netlist no longer holds.
 * (BNCmatmul builds already pass -ffp-contract=off everywhere.)
 *
 * Certified error bounds (u = 2^-p, FPANVerifier, all precisions p at once):
 *   DW: |z - (xy+c)| <=  34 u^2 (|xy| + |c|)   17 flops
 *   TW: |z - (xy+c)| <= 184 u^3 (|xy| + |c|)   66 flops
 *   QW: |z - (xy+c)| <= 812 u^4 (|xy| + |c|)  146 flops
 * (a fused hardware FMA counts as 1 flop; two_sum = 6, fast_two_sum = 3,
 *  two_prod = 2)
 *
 * Naming: the paper's fast_two_sum() is BNCmatmul's quick_two_sum().
 *   double-based DW/TW/QW == BNCmatmul DD/TD/QD
 *   single-based DW/TW/QW == BNCmatmul DS/TS/QS  (bnc_*fmaf)
 */
#ifndef __BNC_FMA_H
#define __BNC_FMA_H

/* Umbrella header: double-based (DD/TD/QD) and single-based (DS/TS/QS)
   branch-free FMA. */
#include "bncfma_d.h"
#include "bncfma_f.h"

#endif /* __BNC_FMA_H */
