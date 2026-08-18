/********************************************************************************/
/* bncomp.h: Parallelized Mutiple Precision Linear Computation Library          */
/*                                                             with OpenMP      */
/* Copyright (C) 2013-2015 Tomonori Kouya                                       */
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
// BNCpack with OpenMP
#include "bncomp.h"

// set_bncomp_num_threads(int num_threads)
int set_bncomp_num_threads(int num_threads)
{	
#ifdef _OPENMP
	if((num_threads > 0) || (num_threads <= BNCOMP_MAX_NUM_THREADS))
	{
		_bncomp_num_threads = num_threads;
		omp_set_num_threads(_bncomp_num_threads);
	}
	printf("Max.Num.Threads           : %d\n", omp_get_max_threads());
	printf("OMP %d threads will be used.\n", omp_get_num_threads());
#else // _OPENMP
	_bncomp_num_threads = 1;
	printf("Max.Num.Threads           : %d\n", 1);
	printf("OMP %d threads will be used.\n", 1);
#endif // _OPENMP

	return _bncomp_num_threads;
}

// get number of threads in BNCOMP
int get_bncomp_num_threads(void)
{
	return _bncomp_num_threads;
}

//---------------------------------------
// DD
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_ddvector(DDVector c, DDVector a)
{
	long int i;

	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
#ifdef __cplusplus
		c->element[i] = a->element[i];
#else // __cplusplus
		set_ddvector_i(c, i, get_ddvector_i(a, i));
#endif // __cplusplus
}

/* c = a + b */
void _bncomp_add_ddvector(DDVector c, DDVector a, DDVector b)
{
	int thread_index;
	long int i;
#ifdef __cplusplus
	dd_real tmp;
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif //  __cplusplus

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = a->element[i] + b->element[i];
#else // __cplusplus
		rdd_add(tmp[thread_index], get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
#endif //  __cplusplus
	}
}

/* c = a - b */
void _bncomp_sub_ddvector(DDVector c, DDVector a, DDVector b)
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_ddvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = a->element[i] - b->element[i];
#else // __cplusplus
		rdd_sub(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
#endif // __cplusplus
	}

}

/* c = val * a */
#ifdef __cplusplus
void _bncomp_cmul_ddvector(DDVector c, dd_real val, DDVector a)
#else // __cplusplus
void _bncomp_cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
#endif // __cplusplus
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_ddvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = val * a->element[i];
#else // __cplusplus
		rdd_mul(tmp[thread_index], val, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp[thread_index]);
#endif // __cplusplus
	}

}

/* (a, b) */
#ifdef __cplusplus
void _bncomp_ip_ddvector(dd_real *ret, DDVector a, DDVector b)
#else // __cplusplus
void _bncomp_ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b)
#endif // __cplusplus
{
	int thread_index;
#ifdef __cplusplus
	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cplusplus
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_ddvector\n");
		return;
	}

#ifdef __cplusplus
	*ret = 0.0;
#else // __cplusplus
	set0_dd(ret);
#endif // __cplusplus

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		//*ret += a->element[i] * b->element[i];
		tmp[thread_index] = a->element[i] * b->element[i];

#pragma omp critical
		*ret = *ret + tmp[thread_index];

#else // __cplusplus
		rdd_mul(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));

#pragma omp critical
		rdd_add(ret, ret, tmp[thread_index]);

#endif // __cplusplus
	}

	return;
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	int thread_index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = a->element[index[thread_index]] + b->element[index[thread_index]];
#else // __cplusplus
			rdd_add(tmp[thread_index], get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c := a - b */
void _bncomp_sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	int thread_index;
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = a->element[index[thread_index]] - b->element[index[thread_index]];
#else // __cplusplus
			rdd_sub(tmp[thread_index], get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c := sc * a */
#ifdef __cplusplus
void _bncomp_cmul_ddmatrix(DDMatrix c, dd_real sc, DDMatrix a)
#else // __cplusplus
void _bncomp_cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a)
#endif // __cplusplus
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = sc * a->element[index[thread_index]];
#else // __cplusplus
			rdd_mul(tmp[thread_index], sc, get_ddmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c = a * b */
//void _bncomp_mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
//{
//	mul_ddmatrix(ret, a, b);
//}
void _bncomp_mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;
#ifdef __cplusplus
	dd_real tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cplusplus

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
#ifdef __cplusplus
		set0_dd(tmp1[thread_index]);
#endif // __cplusplus
	}

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
#ifdef __cplusplus
			tmp1[thread_index] = (dd_real)0.0;
			tmp[thread_index] = (dd_real)0.0;
			for(k = 0; k < mid_dim; k++)
			{
				tmp1[thread_index] = a->element[i * a->col_dim + k] * b->element[k * b->col_dim + j];
				tmp[thread_index] = dd_real::ieee_add(tmp[thread_index], tmp1[thread_index]);
			}

			ret->element[i * col_dim + j] = tmp[thread_index];

#else // __cplusplus
			c_dd_copy_d((double)0.0, GET_DDMATRIX_IJ(ret, i, j));
			for(k = 0; k < mid_dim; k++)
			{
				c_dd_mul(GET_DDMATRIX_IJ(a, i, k), GET_DDMATRIX_IJ(b, k, j), tmp[thread_index]);
				c_dd_add(tmp[thread_index], GET_DDMATRIX_IJ(ret, i, j), GET_DDMATRIX_IJ(ret, i, j));
			}
#endif // __cplusplus
		}
	}
}


/* c := a */
void _bncomp_subst_ddmatrix(DDMatrix c, DDMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix\n");
		return;
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
#ifdef __cplusplus
			index = i * c->col_dim + j;
			c->element[index] = a->element[index];
#else // __cplusplus
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, i, j));
#endif // __cplusplus
		}
	}
}

/* c := I */
void _bncomp_setI_ddmatrix(DDMatrix c)
{
	long int i, j;
#ifdef __cplusplus
#else // __cplusplus
	double tmp0[DDSIZE], tmp1[DDSIZE];
#endif // _cplusplus

#ifdef __cplusplus
#else // __cplusplus
	rdd_set_ui(tmp0, 0UL);
	rdd_set_ui(tmp1, 1UL);
#endif // __cplusplus

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
#ifdef __cplusplus
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = 0.0;
		if(i < c->col_dim)
			c->element[i * c->col_dim + i] = 1.0;
#else // __cplusplus
		for(j = 0; j < c->col_dim; j++)
			set_ddmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_ddmatrix_ij(c, i, i, tmp1);
#endif // __cplusplus
	}
}

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_ddmatrix(DDMatrix mat)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
#ifdef __cplusplus
			mat->element[i * mat->col_dim + j] = 0.0;
#else // __cplusplus
			SET0_DDMATRIX_IJ(mat, i, j);
#endif // __cplusplus
		}
	}
}

/* v := a * vb */
void _bncomp_mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	int thread_index;
#ifdef __cplusplus
	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrix_ddvec\n");
		return;
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->row_dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		tmp[thread_index] = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp[thread_index] += a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp[thread_index];
#else  // __cplusplus
		rdd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_mul(tmp1[thread_index], get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			rdd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_ddvector_i(v, i, tmp[thread_index]);
#endif // __cplusplus
	}
}

/* v := a^T * vb */
void _bncomp_mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	int thread_index;
#ifdef __cplusplus
	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrixt_ddvec\n");
		return;
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->col_dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		tmp[thread_index] = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp[thread_index] += a->element[j * a->col_dim + i] * vb->element[j];

		v->element[i] = tmp[thread_index];
#else // __cplusplus
		set0_dd(tmp[thread_index]);
		for(j = 0; j < a->row_dim; j++)
		{
			rdd_mul(tmp1[thread_index], get_ddmatrix_ij(a, j, i), get_ddvector_i(vb, j));
			rdd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_ddvector_i(v, i, tmp[thread_index]);
#endif // __cplusplus
	}
}

//---------------------------------------
// TD
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_tdvector(TDVector c, TDVector a)
{
	long int i;

	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
//#ifdef __cplusplus
//		c->element[i] = a->element[i];
//#else // __cplusplus
		set_tdvector_i(c, i, get_tdvector_i(a, i));
//#endif // __cplusplus
}

/* c = a + b */
void _bncomp_add_tdvector(TDVector c, TDVector a, TDVector b)
{
	int thread_index;
	long int i;
//#ifdef __cplusplus
//	dd_real tmp;
//#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];
//#endif //  __cplusplus

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		c->element[i] = a->element[i] + b->element[i];
//#else // __cplusplus
		rtd_add(tmp[thread_index], get_tdvector_i(a, i),  get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp[thread_index]);
//#endif //  __cplusplus
	}
}

/* c = a - b */
void _bncomp_sub_tdvector(TDVector c, TDVector a, TDVector b)
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_tdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		c->element[i] = a->element[i] - b->element[i];
//#else // __cplusplus
		rtd_sub(tmp[thread_index], get_tdvector_i(a, i), get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp[thread_index]);
//#endif // __cplusplus
	}

}

/* c = val * a */
//#ifdef __cplusplus
//void _bncomp_cmul_tdvector(TDVector c, dd_real val, TDVector a)
//#else // __cplusplus
void _bncomp_cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a)
//#endif // __cplusplus
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_tdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		c->element[i] = val * a->element[i];
//#else // __cplusplus
		rtd_mul(tmp[thread_index], val, get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp[thread_index]);
//#endif // __cplusplus
	}

}

/* (a, b) */
//#ifdef __cplusplus
//void _bncomp_ip_tdvector(dd_real *ret, TDVector a, TDVector b)
//#else // __cplusplus
void _bncomp_ip_tdvector(double ret[TDSIZE], TDVector a, TDVector b)
//#endif // __cplusplus
{
	int thread_index;
//#ifdef __cplusplus
//	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
//#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];
//#endif // __cplusplus
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_tdvector\n");
		return;
	}

//#ifdef __cplusplus
//	*ret = 0.0;
//#else // __cplusplus
	set0_td(ret);
//#endif // __cplusplus

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		//*ret += a->element[i] * b->element[i];
//		tmp[thread_index] = a->element[i] * b->element[i];
//
//#pragma omp critical
//		*ret = *ret + tmp[thread_index];
//
//#else // __cplusplus
		rtd_mul(tmp[thread_index], get_tdvector_i(a, i), get_tdvector_i(b, i));

#pragma omp critical
		rtd_add(ret, ret, tmp[thread_index]);

//#endif // __cplusplus
	}

	return;
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	int thread_index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		
		for(j = 0; j < col_dim; j++)
		{
//#ifdef __cplusplus
//			index[thread_index] = i * col_dim + j;
//			c->element[index[thread_index]] = a->element[index[thread_index]] + b->element[index[thread_index]];
//#else // __cplusplus
			rtd_add(tmp[thread_index], get_tdmatrix_ij(a, i, j), get_tdmatrix_ij(b, i, j));
			set_tdmatrix_ij(c, i, j, tmp[thread_index]);
//#endif // __cplusplus
		}
	}
}

/* c := a - b */
void _bncomp_sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	int thread_index;
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
//#ifdef __cplusplus
//			index[thread_index] = i * col_dim + j;
//			c->element[index[thread_index]] = a->element[index[thread_index]] - b->element[index[thread_index]];
//#else // __cplusplus
			rtd_sub(tmp[thread_index], get_tdmatrix_ij(a, i, j), get_tdmatrix_ij(b, i, j));
			set_tdmatrix_ij(c, i, j, tmp[thread_index]);
//#endif // __cplusplus
		}
	}
}

/* c := sc * a */
//#ifdef __cplusplus
//void _bncomp_cmul_tdmatrix(TDMatrix c, dd_real sc, TDMatrix a)
//#else // __cplusplus
void _bncomp_cmul_tdmatrix(TDMatrix c, double sc[TDSIZE], TDMatrix a)
//#endif // __cplusplus
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
//#ifdef __cplusplus
//			index[thread_index] = i * col_dim + j;
//			c->element[index[thread_index]] = sc * a->element[index[thread_index]];
//#else // __cplusplus
			rtd_mul(tmp[thread_index], sc, get_tdmatrix_ij(a, i, j));
			set_tdmatrix_ij(c, i, j, tmp[thread_index]);
//#endif // __cplusplus
		}
	}
}

/* c = a * b */
//void _bncomp_mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b)
//{
//	mul_tdmatrix(ret, a, b);
//}
void _bncomp_mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;
//#ifdef __cplusplus
//	dd_real tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
//#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][TDSIZE];
//#endif // __cplusplus

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_td(tmp[thread_index]);
//#ifdef __cplusplus
		set0_td(tmp1[thread_index]);
//#endif // __cplusplus
	}

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
//#ifdef __cplusplus
//			tmp1[thread_index] = (dd_real)0.0;
//			tmp[thread_index] = (dd_real)0.0;
//			for(k = 0; k < mid_dim; k++)
//			{
//				tmp1[thread_index] = a->element[i * a->col_dim + k] * b->element[k * b->col_dim + j];
//				tmp[thread_index] = dd_real::ieee_add(tmp[thread_index], tmp1[thread_index]);
//			}
//
//			ret->element[i * col_dim + j] = tmp[thread_index];
//
//#else // __cplusplus
			c_td_copy_d((double)0.0, GET_TDMATRIX_IJ(ret, i, j));
			for(k = 0; k < mid_dim; k++)
			{
				c_td_mul(GET_TDMATRIX_IJ(a, i, k), GET_TDMATRIX_IJ(b, k, j), tmp[thread_index]);
				c_td_add(tmp[thread_index], GET_TDMATRIX_IJ(ret, i, j), GET_TDMATRIX_IJ(ret, i, j));
			}
//#endif // __cplusplus
		}
	}
}


/* c := a */
void _bncomp_subst_tdmatrix(TDMatrix c, TDMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix\n");
		return;
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
//#ifdef __cplusplus
//			index = i * c->col_dim + j;
//			c->element[index] = a->element[index];
//#else // __cplusplus
			set_tdmatrix_ij(c, i, j, get_tdmatrix_ij(a, i, j));
//#endif // __cplusplus
		}
	}
}

/* c := I */
void _bncomp_setI_tdmatrix(TDMatrix c)
{
	long int i, j;
//#ifdef __cplusplus
//#else // __cplusplus
	double tmp0[TDSIZE], tmp1[TDSIZE];
//#endif // _cplusplus

//#ifdef __cplusplus
//#else // __cplusplus
	rtd_set_ui(tmp0, 0UL);
	rtd_set_ui(tmp1, 1UL);
//#endif // __cplusplus

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
//#ifdef __cplusplus
//		for(j = 0; j < c->col_dim; j++)
//			c->element[i * c->col_dim + j] = 0.0;
//		if(i < c->col_dim)
//			c->element[i * c->col_dim + i] = 1.0;
//#else // __cplusplus
		for(j = 0; j < c->col_dim; j++)
			set_tdmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_tdmatrix_ij(c, i, i, tmp1);
//#endif // __cplusplus
	}
}

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void _bncomp_set0_tdmatrix(TDMatrix mat)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
//#ifdef __cplusplus
//			mat->element[i * mat->col_dim + j] = 0.0;
//#else // __cplusplus
			SET0_TDMATRIX_IJ(mat, i, j);
//#endif // __cplusplus
		}
	}
}

/* v := a * vb */
void _bncomp_mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j;
	int thread_index;
//#ifdef __cplusplus
//	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
//#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][TDSIZE];
//#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrix_tdvec\n");
		return;
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->row_dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		tmp[thread_index] = 0.0;
//		for(j = 0; j < a->col_dim; j++)
//			tmp[thread_index] += a->element[i * a->col_dim + j] * vb->element[j];
//
//		v->element[i] = tmp[thread_index];
//#else  // __cplusplus
		rtd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_mul(tmp1[thread_index], get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			rtd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_tdvector_i(v, i, tmp[thread_index]);
//#endif // __cplusplus
	}
}

/* v := a^T * vb */
void _bncomp_mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j;
	int thread_index;
//#ifdef __cplusplus
//	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
//#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][TDSIZE];
//#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrixt_tdvec\n");
		return;
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->col_dim; i++)
	{
		thread_index = omp_get_thread_num();

//#ifdef __cplusplus
//		tmp[thread_index] = 0.0;
//		for(j = 0; j < a->row_dim; j++)
//			tmp[thread_index] += a->element[j * a->col_dim + i] * vb->element[j];
//
//		v->element[i] = tmp[thread_index];
//#else // __cplusplus
		set0_td(tmp[thread_index]);
		for(j = 0; j < a->row_dim; j++)
		{
			rtd_mul(tmp1[thread_index], get_tdmatrix_ij(a, j, i), get_tdvector_i(vb, j));
			rtd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_tdvector_i(v, i, tmp[thread_index]);
//#endif // __cplusplus
	}
}

//---------------------------------------
// QD
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_qdvector(QDVector c, QDVector a)
{
	long int i;

	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
#ifdef __cplusplus
		c->element[i] = a->element[i];
#else // __cplusplus
		set_qdvector_i(c, i, get_qdvector_i(a, i));
#endif // __cplusplus
}

/* c = a + b */
void _bncomp_add_qdvector(QDVector c, QDVector a, QDVector b)
{
	int thread_index;
	long int i;
#ifdef __cplusplus
	dd_real tmp;
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#endif //  __cplusplus

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = a->element[i] + b->element[i];
#else // __cplusplus
		rqd_add(tmp[thread_index], get_qdvector_i(a, i),  get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp[thread_index]);
#endif //  __cplusplus
	}
}

/* c = a - b */
void _bncomp_sub_qdvector(QDVector c, QDVector a, QDVector b)
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = a->element[i] - b->element[i];
#else // __cplusplus
		rqd_sub(tmp[thread_index], get_qdvector_i(a, i), get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp[thread_index]);
#endif // __cplusplus
	}

}

/* c = val * a */
#ifdef __cplusplus
void _bncomp_cmul_qdvector(QDVector c, qd_real val, QDVector a)
#else // __cplusplus
void _bncomp_cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
#endif // __cplusplus
{
	int thread_index;
	long int i;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_qdvector\n");
		return;
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		c->element[i] = val * a->element[i];
#else // __cplusplus
		rqd_mul(tmp[thread_index], val, get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp[thread_index]);
#endif // __cplusplus
	}

}

/* (a, b) */
#ifdef __cplusplus
void _bncomp_ip_qdvector(qd_real *ret, QDVector a, QDVector b)
#else // __cplusplus
void _bncomp_ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b)
#endif // __cplusplus
{
	int thread_index;
#ifdef __cplusplus
	qd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#endif // __cplusplus
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_qdvector\n");
		return;
	}

#ifdef __cplusplus
	*ret = 0.0;
#else // __cplusplus
	set0_qd(ret);
#endif // __cplusplus

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

#ifdef __cplusplus
		//*ret += a->element[i] * b->element[i];
		tmp[thread_index] = a->element[i] * b->element[i];

#pragma omp critical
		*ret = *ret + tmp[thread_index];

#else // __cplusplus
		rqd_mul(tmp[thread_index], get_qdvector_i(a, i), get_qdvector_i(b, i));

#pragma omp critical
		rqd_add(ret, ret, tmp[thread_index]);

#endif // __cplusplus
	}

	return;
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	int thread_index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = a->element[index[thread_index]] + b->element[index[thread_index]];
#else // __cplusplus
			rqd_add(tmp[thread_index], get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c := a - b */
void _bncomp_sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	int thread_index;
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = a->element[index[thread_index]] - b->element[index[thread_index]];
#else // __cplusplus
			rqd_sub(tmp[thread_index], get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c := sc * a */
#ifdef __cplusplus
void _bncomp_cmul_qdmatrix(QDMatrix c, dd_real sc, QDMatrix a)
#else // __cplusplus
void _bncomp_cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a)
#endif // __cplusplus
{
	long int i, j, row_dim, col_dim, index[BNCOMP_MAX_NUM_THREADS];
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index[thread_index] = i * col_dim + j;
			c->element[index[thread_index]] = sc * a->element[index[thread_index]];
#else // __cplusplus
			rqd_mul(tmp[thread_index], sc, get_qdmatrix_ij(a, i, j));
			set_qdmatrix_ij(c, i, j, tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c = a * b */
void _bncomp_mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;
#ifdef __cplusplus
	qd_real tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#endif // __cplusplus

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

	thread_num = omp_get_num_threads();

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel private(thread_index, i, j, k)
	{
		thread_index = omp_get_thread_num();

		set0_qd(tmp[thread_index]);
#ifdef __cplusplus
		set0_qd(tmp1[thread_index]);
#endif // __cplusplus

		for(i = thread_index; i < row_dim; i += thread_num)
		{
			for(j = 0; j < col_dim; j++)
			{
	#ifdef __cplusplus
				tmp1[thread_index] = (qd_real)0.0;
				tmp[thread_index] = (qd_real)0.0;
				for(k = 0; k < mid_dim; k++)
				{
					#if 0
						tmp1[thread_index] = a->element[i * a->col_dim + k] * b->element[k * b->col_dim + j];
						//tmp[thread_index] = qd_real::ieee_add(tmp[thread_index], tmp1[thread_index]);
						tmp[thread_index] = tmp[thread_index] + tmp1[thread_index];
					#endif // 0
					tmp1[thread_index] = qd_real::accurate_mul((qd_real &)(a->element[i * a->col_dim + k]), (qd_real &)(b->element[k * b->col_dim + j]));
					tmp[thread_index] = qd_real::ieee_add(tmp[thread_index], tmp1[thread_index]);
				}

				ret->element[i * col_dim + j] = tmp[thread_index];

	#else // __cplusplus
				c_qd_copy_d((double)0.0, GET_DDMATRIX_IJ(ret, i, j));
				for(k = 0; k < mid_dim; k++)
				{
					c_qd_mul(GET_DDMATRIX_IJ(a, i, k), GET_DDMATRIX_IJ(b, k, j), tmp[thread_index]);
					c_qd_add(tmp[thread_index], GET_DDMATRIX_IJ(ret, i, j), GET_DDMATRIX_IJ(ret, i, j));
				}
	#endif // __cplusplus
			}
		}
	}
}

/* c := a */
void _bncomp_subst_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix\n");
		return;
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
#ifdef __cplusplus
			index = i * c->col_dim + j;
			c->element[index] = a->element[index];
#else // __cplusplus
			set_qdmatrix_ij(c, i, j, get_qdmatrix_ij(a, i, j));
#endif // __cplusplus
		}
	}
}

/* c := I */
void _bncomp_setI_qdmatrix(QDMatrix c)
{
	long int i, j;
	double tmp0[QDSIZE], tmp1[QDSIZE];

#ifdef __cplusplus
#else // __cplusplus
	rqd_set_ui(tmp0, 0UL);
	rqd_set_ui(tmp1, 1UL);
#endif // __cplusplus

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
#ifdef __cplusplus
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = 0.0;
		if(i < c->col_dim)
			c->element[i * c->col_dim + i] = 1.0;
#else // __cplusplus
		for(j = 0; j < c->col_dim; j++)
			set_qdmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_qdmatrix_ij(c, i, i, tmp1);
#endif // __cplusplus
	}
}

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void _bncomp_set0_qdmatrix(QDMatrix mat)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
#ifdef __cplusplus
			mat->element[i * mat->col_dim + j] = 0.0;
#else // __cplusplus
			SET0_DDMATRIX_IJ(mat, i, j);
#endif // __cplusplus
		}
	}
}

/* v := a * vb */
void _bncomp_mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
#ifdef __cplusplus
	qd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrix_qdvec\n");
		return;
	}

	row_dim = a->row_dim;
	col_dim = vb->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index, i, j)
	{
		thread_index = omp_get_thread_num();

		for(i = thread_index; i < row_dim; i += thread_num)
		{
	#ifdef __cplusplus
			tmp[thread_index] = 0.0;
			for(j = 0; j < col_dim; j++)
				tmp[thread_index] += a->element[i * a->col_dim + j] * vb->element[j];

			v->element[i] = tmp[thread_index];
	#else  // __cplusplus
			rqd_set_ui(tmp[thread_index], 0UL);
			for(j = 0; j < col_dim; j++)
			{
				rqd_mul(tmp1[thread_index], get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
				rqd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
			}
			set_qdvector_i(v, i, tmp[thread_index]);
	#endif // __cplusplus
		}
	}
}

/* v := a^T * vb */
void _bncomp_mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
#ifdef __cplusplus
	qd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrixt_qdvec\n");
		return;
	}

	row_dim = vb->dim;
	col_dim = a->col_dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index, i, j)
	{
		thread_index = omp_get_thread_num();

		for(i = thread_index; i < col_dim; i += thread_num)
		{
	#ifdef __cplusplus
			tmp[thread_index] = 0.0;
			for(j = 0; j < row_dim; j++)
				tmp[thread_index] += a->element[j * a->col_dim + i] * vb->element[j];

			v->element[i] = tmp[thread_index];
	#else // __cplusplus
			set0_qd(tmp[thread_index]);
			for(j = 0; j < row_dim; j++)
			{
				rqd_mul(tmp1[thread_index], get_qdmatrix_ij(a, j, i), get_qdvector_i(vb, j));
				rqd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
			}
			set_qdvector_i(v, i, tmp[thread_index]);
	#endif // __cplusplus
		}
	}
}

//---------------------------------------
// GMP & MPFR
//---------------------------------------
#ifdef USE_GMP

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_mpfvector(MPFVector c, MPFVector a, long int dim)
{
	long int i;//, dim;

//	dim = c->dim;

	#pragma omp parallel for
	for(i = 0; i < dim; i++)
		set_mpfvector_i(c, i, get_mpfvector_i(a, i));
}

/* c = a - b */
void _bncomp_sub_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim)
{
	long int i;//, dim;
	int thread_num, thread_index;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR:_bncomp_sub_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_sub(tmp[thread_index], get_mpfvector_i(a, i), get_mpfvector_i(b, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c = a + b */
void _bncomp_add_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim)
{
	long int i;//, dim;
	int thread_num, thread_index;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR:_bncomp_add_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_add(tmp[thread_index], get_mpfvector_i(a, i), get_mpfvector_i(b, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c = val * a */
void _bncomp_cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a, long int dim)
{
	long int i;//, dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);
//	dim = c->dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

//		#pragma omp critical
		{
			mpf_mul(tmp[thread_index], val, get_mpfvector_i(a, i));
			set_mpfvector_i(c, i, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* inner product of vector blocks */
void _bncomp_ip_mpfvector(mpf_t ret, MPFVector va, MPFVector vb)
{
	long int i, dim;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if(va->dim != vb->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_mpfvector: va->dim = %ld, vb->dim = %ld !\n", va->dim, vb->dim);
		return;
	}

	dim = va->dim;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL); // = 0.0;
#pragma omp parallel for private(thread_index)
	for(i = 0; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_mul(tmp[thread_index], get_mpfvector_i(va, i), get_mpfvector_i(vb, i));

#pragma omp critical
		mpf_add(ret, ret, tmp[thread_index]);
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp[thread_index]);
}

	return;
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_set_ui(tmp[thread_index], 0UL);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_add(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c := a - b */
void _bncomp_sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_sub(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c := sc * a */
void _bncomp_cmul_mpfmatrix(MPFMatrix c, mpf_t sc, MPFMatrix a)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], in_sc[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	prec = prec_mpfmatrix(c);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(in_sc[thread_index], prec);
		mpf_set(in_sc[thread_index], sc);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_mul(tmp[thread_index], in_sc[thread_index], get_mpfmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(in_sc[thread_index]);
	}
}


/* c = a * b */
void _bncomp_mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, k, row_dim, col_dim, mid_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrix\n");
		return;
	}

	prec = prec_mpfmatrix(c);

	row_dim = c->row_dim;
	col_dim = c->col_dim;
	mid_dim = a->col_dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpf_set_ui(tmp[thread_index], 0UL);
			for(k = 0; k < mid_dim; k++)
			{
#ifndef USE_MPFR
				mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j));
				mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
#else
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j), tmp[thread_index], bnc_default_rounding_mode);
				mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j), tmp[thread_index], MPFR_RNDN);
#endif
			}
			set_mpfmatrix_ij(c, i, j, tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}

/* c := a */
void _bncomp_subst_mpfmatrix(MPFMatrix c, MPFMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_subst_mpfmatrix\n");
		return;
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, get_mpfmatrix_ij(a, i, j));
	}
}

/* c := 0 */
void _bncomp_set0_mpfmatrix(MPFMatrix c)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij_ui(c, i, j, 0UL);
	}
}

/* c := I */
void _bncomp_setI_mpfmatrix(MPFMatrix c)
{
	long int i, j;

	#pragma omp parallel for private(j)
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij_ui(c, i, j, 0UL);

		if(i < c->col_dim)
			set_mpfmatrix_ij_ui(c, i, i, 1UL);
	}
}

/* v := a * vb */
void _bncomp_mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrix_dvec\n");
		return;
	}

	prec = prec_mpfvector(v);
	row_dim = a->row_dim;
	col_dim = vb->dim;
	
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index, i, j)
	{

		thread_index = omp_get_thread_num();

		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);

		for(i = thread_index; i < row_dim; i += thread_num)
		{

			//printf("i, thread_index = %ld, %d\n", i, thread_index);

			mpf_set_ui(tmp[thread_index], 0UL);
			for(j = 0; j < col_dim; j++)
			{
	#ifndef USE_MPFR
				mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j));
				mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
	#else
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j), tmp[thread_index], MPFR_RNDN);
	#endif
			}

			set_mpfvector_i(v, i, tmp[thread_index]);
			//printf("end of i, thread_index = %ld, %d\n", i, thread_index);
		}

		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}

/* v := a^T * vb */
void _bncomp_mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j, row_dim, col_dim;
	int thread_index, thread_num;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfmatrixt_mpfvec\n");
		return;
	}

	prec = prec_mpfvector(v);

	row_dim = vb->dim;
	col_dim = a->col_dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index, i, j)
	{

		thread_index = omp_get_thread_num();

		mpf_init2(tmp[thread_index], prec);
		mpf_init2(tmp1[thread_index], prec);

		for(i = thread_index; i < col_dim; i += thread_num)
		{

			mpf_set_ui(tmp[thread_index], 0UL);
			for(j = 0; j < row_dim; j++)
			{
	#ifndef USE_MPFR
				mpf_mul(tmp1[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j));
				mpf_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
	#else
				//mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], bnc_default_rounding_mode);
				mpfr_fma(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp[thread_index], MPFR_RNDN);
	#endif
			}
			set_mpfvector_i(v, i, tmp[thread_index]);
		}

		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp1[thread_index]);
	}
}
#endif // USE_GMP
