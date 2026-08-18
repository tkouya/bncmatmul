/********************************************************************************/
/* gqslinear.h: Quadruple-single precision Linear Computation Library (GPU+CUDA)*/
/* Copyright (C) 2015-2026 Tomonori Kouya                                       */
/*                                                                              */
/* GQS (quad-single, 4 IEEE floats) GPU types and prototypes.                   */
/*                                                                              */
/* Mirrors the GQD convention: there is no separate gqdlinear.h in the          */
/* double-based tree (src/gqdlinear.cu just includes gddlinear.h, which already */
/* declares the GQD types/prototypes).  This header therefore only forwards to  */
/* gdslinear.h, which declares GDSVector/GDSMatrix and GQSVector/GQSMatrix and   */
/* all the gqs_* helpers.  src/gqslinear.cu includes gdslinear.h directly.      */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/********************************************************************************/
#ifndef __BNC_GQSLINEAR_H__
  #define __BNC_GQSLINEAR_H__

#include "gdslinear.h"   /* GDSVector/GDSMatrix, GQSVector/GQSMatrix, gqs_* helpers */

#endif /* __BNC_GQSLINEAR_H__ */
