/********************************************************************************/
/* bnc_common_def.h: Common macros and funtions used in BNCpack                 */
/* Copyright (C) 2021 Tomonori Kouya                                            */
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
// common definition 
#ifndef __BNC_COMMON_DEF_H
#define __BNC_COMMON_DEF_H

// SUCCESS or ERROR
#define BNC_SUCCESS (0)
#define BNC_ERROR (-1)

// Intel Compiler or not
// ref) https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/macros/additional-predefined-macros.html
#if defined(__ICC) || defined(__ICL) // __ICC: Linux or macOS, __ICL: Windows
    #define BNC_USE_ICC
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html#Common-Predefined-Macros
#elif defined(__GNUC__) // GCC
    #define BNC_USE_GCC
#else // unknown compiler
    #undef BNC_USE_ICC
    #undef BNC_USE_GCC
#endif // BNC_USE_ICC or _GCC

#endif // __BNC_COMMON_DEF_H
