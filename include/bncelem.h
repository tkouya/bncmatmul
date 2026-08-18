/********************************************************************************/
/* bncelem.h : umbrella header for the DD/TD/QD elementary functions in plain C */
/*                                                                              */
/* Port of the fused-multiply-add elementary functions of dtq-0.0.3:            */
/*   exp / expm1 / log / log10 / sin / cos / sincos                             */
/* for double-double (bncelem_dd.h), triple-double (bncelem_td.h) and           */
/* quad-double (bncelem_qd.h), with the coefficient tables extracted into       */
/* bncelem_tables.h.  Each Taylor term is accumulated with one certified        */
/* branch-free FMA (bncfma_d.h) instead of a separate multiply and add.         */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#ifndef __BNC_ELEM_H
#define __BNC_ELEM_H

#include "bncelem_tables.h"
#include "bncelem_dd.h"
#include "bncelem_td.h"
#include "bncelem_qd.h"

#endif /* __BNC_ELEM_H */
