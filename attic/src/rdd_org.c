/********************************************************************************/
/* rdd.c: Reverse definition for double-double and quadruple-double arithmetic  */
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
#include "rdd.h"

#ifndef __cplusplus

// DD print(no appending CR)
void rdd_out_str_base(FILE *fp, int base, int length, double val[DDSIZE])
{
	static char str[64];
	c_dd_swrite(val, (length > 40) ? 40 : length, str, 46);
	fprintf(fp, "%s", str);
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
int rdd_cmp(double a[DDSIZE], double b[DDSIZE])
{
	int ret;

	c_dd_comp(a, b, &ret);

	return ret;
}

// DD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
int rdd_cmp_d(double a[DDSIZE], double b)
{
	int ret;

	c_dd_comp_dd_d(a, b, &ret);

	return ret;
}

// DD sqrt_d
void rdd_sqrt_d(double ret[DDSIZE], double a)
{
	static double tmp[DDSIZE];

	c_dd_copy_d(a, tmp);
	c_dd_sqrt(tmp, ret);

	return;
}

// DD fma
// ret = a * b + c
void rdd_fma(double ret[DDSIZE], double a[DDSIZE], double b[DDSIZE], double c[DDSIZE])
{
	static double tmp[DDSIZE];

	c_dd_mul(a, b, tmp);
	c_dd_add(tmp, c, ret);

	return;
}

// DD pow
// ret = base^power = exp(power * log(base))
void rdd_pow(double ret[DDSIZE], double base[DDSIZE], double power[DDSIZE])
{
	static double tmp[DDSIZE], tmp1[DDSIZE];

	c_dd_log(base, tmp);
	c_dd_mul(power, tmp, tmp1);
	c_dd_exp(tmp1, ret);

	return;
}

// QD print(no appending CR)
void rqd_out_str_base(FILE *fp, int base, int length, double val[QDSIZE])
{
	static char str[128];
	// void c_qd_swrite(const double *a, int precision, char *s, int len);
	c_qd_swrite(val, (length > 70) ? 70 : length, str, 80);
	fprintf(fp, "%s", str);
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
int rqd_cmp(double a[QDSIZE], double b[QDSIZE])
{
	int ret;

	c_qd_comp(a, b, &ret);

	return ret;
}

// QD comparing functions
// ret = 1 in case of a  > b
// ret = 0            a == b
// ret = -1           a <  b
int rqd_cmp_d(double a[QDSIZE], double b)
{
	int ret;

	c_qd_comp_qd_d(a, b, &ret);

	return ret;
}

// QD sqrt_d
void rqd_sqrt_d(double ret[QDSIZE], double a)
{
	static double tmp[QDSIZE];

	c_qd_copy_d(a, tmp);
	c_qd_sqrt(tmp, ret);

	return;
}

// QD fma
// ret = a * b + c
void rqd_fma(double ret[QDSIZE], double a[QDSIZE], double b[QDSIZE], double c[QDSIZE])
{
	static double tmp[QDSIZE];

	c_qd_mul(a, b, tmp);
	c_qd_add(tmp, c, ret);

	return;
}

// QD pow
// ret = base^power = exp(power * log(base))
void rqd_pow(double ret[QDSIZE], double base[QDSIZE], double power[QDSIZE])
{
	static double tmp[QDSIZE], tmp1[QDSIZE];

	c_qd_log(base, tmp);
	c_qd_mul(power, tmp, tmp1);
	c_qd_exp(tmp1, ret);

	return;
}

#endif // __cplusplus
