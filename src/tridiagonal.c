/****************************************************************************/
/* tridiagonal.c: for eigenproblems of real symmetric dense matrices       */
/*                                                                          */
/* Copyright (C) 2008- Tomonori Kouya <http://na-inet.jp/>                  */
/*                                                                          */
/* This program is free software: you can redistribute it and/or modify it  */
/* under the terms of the GNU Lesser General Public License as published    */
/* by the Free Software Foundation, either version 3 of the License,        */
/* or any later version.                                                    */
/*                                                                          */
/* This program is distributed in the hope that it will be useful,          */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of           */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            */
/* GNU Lesser General Public License for more details.                      */
/*                                                                          */
/* You should have received a copy of the GNU Lesser General Public License */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.    */
/****************************************************************************/
#include <stdio.h>
#include <math.h>
//#include "bnc.h"
#include "bncmatmul.h"

/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
int init_dmatrix_tri(DVector trimat[3], long int dim)
{
	if((trimat[0] = init_dvector(dim - 1)) == NULL)
		return -1;
	if((trimat[1] = init_dvector(dim)) == NULL)
	{
		free_dvector(trimat[0]);
		return -2;
	}
	if((trimat[2] = init_dvector(dim - 1)) == NULL)
	{
		free_dvector(trimat[0]);
		free_dvector(trimat[1]);
		return -3;
	}

	return 0;
}

void free_dmatrix_tri(DVector trimat[3])
{
	free_dvector(trimat[0]);
	free_dvector(trimat[1]);
	free_dvector(trimat[2]);
}

void print_dmatrix_tri(DVector trimat[3])
{
	long int i;

	printf("%3d:                          %25.17e %25.17e\n", 0, get_dvector_i(trimat[1], 0), get_dvector_i(trimat[0], 0));
	for(i = 1; i < trimat[1]->dim - 1; i++)
		printf("%3ld:%25.17e %25.17e %25.17e\n", i, get_dvector_i(trimat[2], i - 1), get_dvector_i(trimat[1], i), get_dvector_i(trimat[0], i));
	printf("%3ld:%25.17e %25.17e\n", i, get_dvector_i(trimat[2], trimat[1]->dim - 2), get_dvector_i(trimat[1], trimat[1]->dim - 1));
}

/* Transform Real Symmetric Square Matric to TridiagonalForm */
int dstrimat(DVector trimat[3], DMatrix mat, DMatrix proj_mat, int flag_get_proj_mat)
{
	long int n, i, j, k, dim;
	double c, s, s2, mat_n1n, tmp[3];
	DVector w, caw, q;

	dim = mat->row_dim;
	if(dim <= 0 || dim != mat->col_dim)
		return 0;

	if(proj_mat == NULL)
		flag_get_proj_mat = 0;

	/* Initialize */
	w   = init_dvector(dim);
	q   = init_dvector(dim);
	caw = init_dvector(dim);

	if(flag_get_proj_mat == 1)
		setI_dmatrix(proj_mat);

	/* Main loop */
	for(n = 0; n < dim - 1; n++)
	{
		/* get sigma, beta, v */
		s2 = 0.0;
		for(i = n + 1; i < dim; i++)
			s2 += gdmij(mat, i, n) * gdmij(mat, i, n);

		mat_n1n = gdmij(mat, n + 1, n);

		if(mat_n1n > 0.0)
			s = sqrt(s2);
		else
			s = -sqrt(s2);

		c = 1.0 / (s2 + mat_n1n * s);

		/* w := [0 ... 0 a_{n+1,n}+s a_{n+2,n} ... a_{dim,n} */
		tmp[0] = mat_n1n + s;
		sdvi(w, n + 1, tmp[0]);
		for(i = n + 2; i < dim; i++)
			sdvi(w, i, gdmij(mat, i, n));
	
		/* Projection matrix */
		if(flag_get_proj_mat == 1)
		{
			for(i = 0; i < dim; i++)
			{
				for(j = n + 1; j < dim; j++)
				{
					tmp[0] = 0.0;
					for(k = n + 1; k < dim; k++)
					{
						if(j == k)
							tmp[1] = 1.0;
						else
							tmp[1] = 0.0;

						tmp[1] -= c * gdvi(w, k) * gdvi(w, j);
						tmp[0] += gdmij(proj_mat, i, k) * tmp[1];
					}
					sdvi(caw, j, tmp[0]);
				}
				for(j = n + 1; j < dim; j++)
					sdmij(proj_mat, i, j, gdvi(caw, j));
			}
		}

		/* caw := c * A * w */
		for(i = 0; i < dim; i++)
		{
			tmp[0] = 0.0;
			for(j = n + 1; j < dim; j++)
				tmp[0] += gdmij(mat, i, j) * gdvi(w, j);
			tmp[0] *= c;
			sdvi(caw, i, tmp[0]);
		}

		/* q := caw - c / 2 * (w * caw^T) * w */
		for(i = 0; i <= n; i++)
			sdvi(q, i, gdvi(caw, i));

		for(i = n + 1; i < dim; i++)
		{
			tmp[1] = 0.0;
			for(j = n + 1; j < dim; j++)
				tmp[1] += gdvi(w, i) * gdvi(caw, j) * gdvi(w, j);
			tmp[1] *= c / 2.0;
			tmp[0] = gdvi(caw, i) - tmp[1];
			sdvi(q, i, tmp[0]);
		}

		/* P_i * A P_i = A - q * w^T - w^T * q */

		/* (1) A' := A - q * w^T */
		for(i = 0; i < dim; i++)
		{
			for(j = n + 1; j < dim; j++)
			{
				tmp[0] = gdvi(q, i) * gdvi(w, j);
				tmp[2] = gdmij(mat, i, j) - tmp[0];
				sdmij(mat, i, j, tmp[2]);
			}
		}

		/* (2) A'' := A' - w * q^T */
		for(i = n + 1; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				tmp[1] = gdvi(w, i) * gdvi(q, j);
				tmp[2] = gdmij(mat, i, j) - tmp[1];
				sdmij(mat, i, j, tmp[2]);
			}
		}
	}

	/* copy tridiagonal element to trimat[3] */
	for(i = 0; i < dim - 1; i++)
	{
		set_dvector_i(trimat[0], i, get_dmatrix_ij(mat, i    , i + 1));
		set_dvector_i(trimat[1], i, get_dmatrix_ij(mat, i    , i    ));
		set_dvector_i(trimat[2], i, get_dmatrix_ij(mat, i + 1, i    ));
	}
	set_dvector_i(trimat[1], dim - 1, get_dmatrix_ij(mat, dim - 1, dim - 1));

	/* clear */
	free_dvector(w);
	free_dvector(q);
	free_dvector(caw);

	return dim;
}

#ifdef SINGLE_USE
/* [cosine   sine] [vec0] = [ nu * vec0 / |vec0| ] */
/* [-sine  cosine] [vec1]   [ 0                  ] */
/* where nu = sqrt(vec0^2+vec1^2) */
/* Input : vec0, vec1 */
/* Output: cosine, sine, new vec0(, vec1 = 0) */
void dplane_rotation(double *cosine, double *sine, double *vec0, double *vec1)
{
	double phi, mu, tau, nu, abs_vec0, abs_vec1, tmp1, tmp2;

	abs_vec0 = fabs(*vec0);
	abs_vec1 = fabs(*vec1);

	if(abs_vec1 == 0.0)
	{
		*cosine = 1.0;
		*sine = 0.0;
		return;
	}
	if(abs_vec0 == 0.0)
	{
		*cosine = 0.0;
		*sine = 1.0;
		*vec0 = *vec1;
		*vec1 = 0.0;
		return;
	}

	mu = *vec0 / abs_vec0;
	tau = abs_vec0 + abs_vec1;
	tmp1 = abs_vec0 / tau;
	tmp2 = abs_vec1 / tau;
	nu = tau * sqrt(tmp1 * tmp1 + tmp2 * tmp2);
	*cosine = abs_vec0 / nu;
	*sine   = mu * (*vec1) / nu;
	*vec0 = nu * mu;
	*vec1 = 0.0;

	return;
}
#endif // SINGLE_USE

//int dtriqr(DVector trimat[3], double shift, long int i1, long int i2)
void dtriqr(DVector trimat[3], double shift, long int i1, long int i2)
{
	long int i;
	double g, sine, cosine, p, f, b, u, v;

	g = get_dvector_i(trimat[1], i1) - shift;
	sine = 1.0;
	cosine = 1.0;
	p = 0.0;

	for(i = i1; i < i2; i++)
	{
		f = sine   * get_dvector_i(trimat[0], i);
		b = cosine * get_dvector_i(trimat[0], i);
		dplane_rotation(&cosine, &sine, &g, &f);
		if(i != i1)
			set_dvector_i(trimat[0], i - 1, g);
		u = get_dvector_i(trimat[1], i) - p;
		v = (get_dvector_i(trimat[1], i + 1) - u) * sine + 2.0 * cosine * b;
		p = sine * v;
		set_dvector_i(trimat[1], i, u + p);
		g = cosine * v - b;
//		printf("g, p, c, s, u, v:  %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e\n", g, p, cosine, sine, u, v);
	}

	set_dvector_i(trimat[1], i2, get_dvector_i(trimat[1], i2) - p);
	set_dvector_i(trimat[0], i2 - 1, g);
}

/* [mat11 mat12] */
/* [mat21 mat22] (mat21 = mat12) */
double dwilkinson_shift_tri(double mat11, double mat12, double mat22)
{
	double coef[3], shift[2], shift_im[2];

	coef[2] = 1.0;
	coef[1] = -(mat11 + mat22);
	coef[0] = mat11 * mat22 - mat12 * mat12;

	dquadratic_eq(shift, shift_im, coef);

	if(fabs(shift[0] - mat22) <= fabs(shift[1] - mat22))
		return shift[0];
	else
		return shift[1];
}

int dtriqr_iteration(DVector trimat[3], double rtol, double atol, long int maxtimes)
{
	long int i, times, dim, index;
	double shift;

	dim = trimat[1]->dim;

	for(i = 0; i < dim - 1; i++)
	{
		index = dim - i - 1;
		for(times = 0; times < maxtimes; times++)
		{
			shift = dwilkinson_shift_tri(
				get_dvector_i(trimat[1], index - 1),
				get_dvector_i(trimat[0], index - 1),
				get_dvector_i(trimat[1], index)
			);
			printf("shift: %10.5e\n", shift);
			dtriqr(trimat, shift, 0, index);
			if(fabs(get_dvector_i(trimat[0], index - 1)) <= rtol * sqrt(fabs(get_dvector_i(trimat[1], index - 1) * get_dvector_i(trimat[1], index))) + atol)
				break;
		}
		if(times >= maxtimes)
			fprintf(stderr, "WARNING: Not Convergent! (dtriqr_iteration: %ld th, %ld deg\n", index, dim);
	}
	return times;
}

/* Get eigenvector of tridiagonal matrices */
void dget_eigenvector_dtri(DVector eigen_vec, DVector trimat[3], double eigenvalue, long int drop_rank)
{
	long int i;
	double l, d, u, ev, evm1, evm2;

	if(drop_rank <= 0)
		return;

	/* set 1 */
	for(i = 0; i < drop_rank; i++)
		set_dvector_i(eigen_vec, i, 1.0);

	evm1 = get_dvector_i(eigen_vec, drop_rank - 1);
	if(drop_rank >= 2)
	{
		l = get_dvector_i(trimat[2], drop_rank - 2);
		evm2 = get_dvector_i(eigen_vec, drop_rank - 2);
	}
	else
	{
		l = 0.0;
		evm2 = 0.0;
	}
	for(i = drop_rank; i < trimat[1]->dim; i++)
	{
		d = get_dvector_i(trimat[1], i - 1) - eigenvalue;
		u = get_dvector_i(trimat[0], i - 1);

		ev = -(l * evm2 + d * evm1) / u;

		set_dvector_i(eigen_vec, i, ev);

		evm2 = evm1;
		evm1 = ev;
		l = get_dvector_i(trimat[2], i - 1);
//		printf("---%d, %25.17e\n", i, ev);
	}

	/* ||eiven_vec||_2 = 1 */
	ev = norm2_dvector(eigen_vec);
	for(i = 0; i < trimat[1]->dim; i++)
		set_dvector_i(eigen_vec, i, get_dvector_i(eigen_vec, i) / ev);
}

/* Multiply diagonal matrix */
void mul_ddiagmat_dvec(DVector ret, DVector diagmat, DVector vec)
{
	long int i, dim;
	double d, v;

	dim = (diagmat->dim < vec->dim) ? diagmat->dim : vec->dim;
	for(i = 0; i < dim; i++)
	{
		d = get_dvector_i(diagmat, i); // d = a_ii
		v = get_dvector_i(vec, i); 
		set_dvector_i(ret, i, d * v);
	}
}

/* GMP & MPFR */
#ifdef USE_GMP
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
int init_mpfmatrix_tri(MPFVector trimat[3], long int dim)
{
	if((trimat[0] = init_mpfvector(dim - 1)) == NULL)
		return -1;
	if((trimat[1] = init_mpfvector(dim)) == NULL)
	{
		free_mpfvector(trimat[0]);
		return -2;
	}
	if((trimat[2] = init_mpfvector(dim - 1)) == NULL)
	{
		free_mpfvector(trimat[0]);
		free_mpfvector(trimat[1]);
		return -3;
	}

	return 0;
}
int init2_mpfmatrix_tri(MPFVector trimat[3], long int dim, unsigned long prec)
{
	if((trimat[0] = init2_mpfvector(dim - 1, prec)) == NULL)
		return -1;
	if((trimat[1] = init2_mpfvector(dim, prec)) == NULL)
	{
		free_mpfvector(trimat[0]);
		return -2;
	}
	if((trimat[2] = init2_mpfvector(dim - 1, prec)) == NULL)
	{
		free_mpfvector(trimat[0]);
		free_mpfvector(trimat[1]);
		return -3;
	}

	return 0;
}


void free_mpfmatrix_tri(MPFVector trimat[3])
{
	free_mpfvector(trimat[0]);
	free_mpfvector(trimat[1]);
	free_mpfvector(trimat[2]);
}

void print2_mpfmatrix_tri(MPFVector trimat[3], int dprec)
{
	long int i;
	unsigned long in_prec, prec_bin;

	in_prec = trimat[1]->prec;

	prec_bin = (unsigned long)ceil(dprec / log10(2.0));

	if((dprec > 0) && (prec_bin < in_prec))
		in_prec = dprec;

	printf("%3ld:                          ", 0UL);
	mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[1], 0));
	printf(", ");
	mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[0], 0));
	printf("\n");
	for(i = 1; i < trimat[1]->dim - 1; i++)
	{
		printf("%3ld:", i);
		mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[2], i - 1));
		printf(", ");
		mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[1], i));
		printf(", ");
		mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[0], i));
		printf("\n");
	}
	printf("%3ld:", i);
	mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[2], trimat[1]->dim - 2));
	printf(", ");
	mpf_out_str(stdout, 10, in_prec, get_mpfvector_i(trimat[1], trimat[1]->dim - 1));
	printf("\n");
}

void print_mpfmatrix_tri(MPFVector trimat[3])
{
	long int i;

	printf("%3ld:                          ", 0UL);
	mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[1], 0));
	printf(", ");
	mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[0], 0));
	printf("\n");
	for(i = 1; i < trimat[1]->dim - 1; i++)
	{
		printf("%3ld:", i);
		mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[2], i - 1));
		printf(", ");
		mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[1], i));
		printf(", ");
		mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[0], i));
		printf("\n");
	}
	printf("%3ld:", i);
	mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[2], trimat[1]->dim - 2));
	printf(", ");
	mpf_out_str(stdout, 10, 17, get_mpfvector_i(trimat[1], trimat[1]->dim - 1));
	printf("\n");
}

/* Transform Real Square Matrix to Hessemberg Form */
int mpfstrimat(MPFVector trimat[3], MPFMatrix mat, MPFMatrix proj_mat, int flag_get_proj_mat)
{
	unsigned long prec;
	long int n, i, j, k, dim;
	mpf_t c, s, s2, mat_n1n, tmp[3];
	MPFVector w, caw, q;

	dim = mat->row_dim;
	if(dim <= 0 || dim != mat->col_dim)
		return 0;

	if(proj_mat == NULL)
		flag_get_proj_mat = 0;

	/* Initialize */

	prec = mat->prec;

	mpf_init2(c, prec);
	mpf_init2(s, prec);
	mpf_init2(s2, prec);
	mpf_init2(mat_n1n, prec);
	for(i = 0; i < 3; i++) mpf_init2(tmp[i], prec);

	w    = init2_mpfvector(dim, prec);
	q    = init2_mpfvector(dim, prec);
	caw  = init2_mpfvector(dim, prec);

	if(flag_get_proj_mat == 1)
		setI_mpfmatrix(proj_mat);

	/* Main loop */
	for(n = 0; n < dim - 1; n++)
	{
		/* get sigma, beta, v */
		mpf_set_ui(s2, 0UL);
		for(i = n + 1; i < dim; i++)
		{
			/* s2 += gmpfmij(mat, i, n) * gmpfmij(mat, i, n) */
			mpf_mul(tmp[0], gmpfmij(mat, i, n), gmpfmij(mat, i, n));
			mpf_add(s2, s2, tmp[0]);
		}

		mpf_set(mat_n1n, gmpfmij(mat, n + 1, n));

		mpf_sqrt(s, s2);
		if(mpf_cmp_ui(mat_n1n, 0UL) < 0)
			mpf_neg(s, s);

		/* c = 1.0 / (s2 + mat_n1n * s) */
		mpf_mul(tmp[0], mat_n1n, s);
		mpf_add(tmp[0], s2, tmp[0]);
		mpf_ui_div(c, 1UL, tmp[0]);

		/* w := [0 ... 0 a_{n+1,n}+s a_{n+2,n} ... a_{dim,n} */
		set0_mpfvector(w);
		mpf_add(tmp[0],  mat_n1n, s);
		smpfvi(w, n + 1, tmp[0]);
		for(i = n + 2; i < dim; i++)
			smpfvi(w, i, gmpfmij(mat, i, n));
	
		/* Projection matrix */
		if(flag_get_proj_mat == 1)
		{
			for(i = 0; i < dim; i++)
			{
				for(j = n + 1; j < dim; j++)
				{
					mpf_set_ui(tmp[0], 0UL);
					for(k = n + 1; k < dim; k++)
					{
						if(j == k)
							mpf_set_ui(tmp[1], 1UL);
						else
							mpf_set_ui(tmp[1], 0UL);

						/* tmp[1] -= c * gmpfvi(w, k) * gmpfvi(w, j)  */
						mpf_mul(tmp[2], gmpfvi(w, k), gmpfvi(w, j));
						mpf_mul(tmp[2], tmp[2], c);
						mpf_sub(tmp[1], tmp[1], tmp[2]);

						/* tmp[0] += gmpfmij(proj_mat, i, k) * tmp[1] */
						mpf_mul(tmp[2], gmpfmij(proj_mat, i, k), tmp[1]);
						mpf_add(tmp[0], tmp[0], tmp[2]);
					}
					smpfvi(caw, j, tmp[0]);
				}
				for(j = n + 1; j < dim; j++)
					smpfmij(proj_mat, i, j, gmpfvi(caw, j));
			}
		}

		/* caw  := c * A * w */
		for(i = 0; i < dim; i++)
		{
			mpf_set_ui(tmp[0], 0UL);
			for(j = n + 1; j < dim; j++)
			{
				/* tmp[0] += gmpfmij(mat, i, j) * gmpfvi(w, j) */
				mpf_mul(tmp[2], gmpfmij(mat, i, j), gmpfvi(w, j));
				mpf_add(tmp[0], tmp[0], tmp[2]);
			}
			mpf_mul(tmp[0], tmp[0], c);
			smpfvi(caw , i, tmp[0]);
		}

		/* q := caw  := caw  - c / 2 * (w * caw^T) * w */
		for(i = 0; i <= n; i++)
			smpfvi(q, i, gmpfvi(caw, i));

		for(i = n + 1; i < dim; i++)
		{
			mpf_set_ui(tmp[1], 0UL);
			for(j = n + 1; j < dim; j++)
			{
				/* tmp[1] += gmpfvi(w, i) * gmpfvi(caw, j) * gmpfvi(w, j) */
				mpf_mul(tmp[2], gmpfvi(w, i), gmpfvi(caw, j));
				mpf_mul(tmp[2], tmp[2], gmpfvi(w, j));
				mpf_add(tmp[1], tmp[1], tmp[2]);
			}
			/* tmp[1] *= c / 2.0 */
			mpf_div_ui(tmp[2], c, 2UL);
			mpf_mul(tmp[1], tmp[1], tmp[2]);
			
			mpf_sub(tmp[0], gmpfvi(caw, i), tmp[1]);
			smpfvi(q, i, tmp[0]);
		}

		/* P_i * A P_i = A - q * w^T - w^T * q' */

		/* (1) A' := A - q * w^T */
		for(i = 0; i < dim; i++)
		{
			for(j = n + 1; j < dim; j++)
			{
				mpf_mul(tmp[0], gmpfvi(q, i), gmpfvi(w, j));
				mpf_sub(tmp[2], gmpfmij(mat, i, j), tmp[0]);
				smpfmij(mat, i, j, tmp[2]);
			}
		}
		/* (2) A'' := A' - w * q^T */
		for(i = n + 1; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				mpf_mul(tmp[1], gmpfvi(w, i), gmpfvi(q, j));
				mpf_sub(tmp[2], gmpfmij(mat, i, j), tmp[1]);
				smpfmij(mat, i, j, tmp[2]);
			}
		}

		/* Nonsense process (due to my lack of knowledge) */
		mpf_set_ui(tmp[0], 0UL);
		for(i = n + 2; i < dim; i++)
			smpfmij(mat, i, n, tmp[0]);
	}

	/* copy tridiagonal element to trimat[3] */
	for(i = 0; i < dim - 1; i++)
	{
		set_mpfvector_i(trimat[0], i, get_mpfmatrix_ij(mat, i    , i + 1));
		set_mpfvector_i(trimat[1], i, get_mpfmatrix_ij(mat, i    , i    ));
		set_mpfvector_i(trimat[2], i, get_mpfmatrix_ij(mat, i + 1, i    ));
	}
	set_mpfvector_i(trimat[1], dim - 1, get_mpfmatrix_ij(mat, dim - 1, dim - 1));

	/* clear */
	mpf_clear(c);
	mpf_clear(s);
	mpf_clear(s2);
	mpf_clear(mat_n1n);
	for(i = 0; i < 3; i++) mpf_clear(tmp[i]);

	free_mpfvector(w);
	free_mpfvector(q);
	free_mpfvector(caw);

	return dim;
}

#ifdef SINGLE_USE
/* [cosine   sine] [vec0] = [ nu * vec0 / |vec0| ] */
/* [-sine  cosine] [vec1]   [ 0                  ] */
/* where nu = sqrt(vec0^2+vec1^2) */
/* Input : vec0, vec1 */
/* Output: cosine, sine, new vec0(, vec1 = 0) */
void mpfplane_rotation(mpf_t cosine, mpf_t sine, mpf_t vec0, mpf_t vec1)
{
	unsigned prec;
	mpf_t phi, mu, tau, nu, abs_vec0, abs_vec1, tmp, tmp1, tmp2;

	prec = (mpf_get_prec(cosine) > mpf_get_prec(sine)) ? mpf_get_prec(cosine) : mpf_get_prec(sine);

	/* Initialize */
	mpf_init2(phi, prec);
	mpf_init2(mu, prec);
	mpf_init2(tau, prec);
	mpf_init2(nu, prec);
	mpf_init2(abs_vec1, prec);
	mpf_init2(abs_vec0, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp2, prec);
	mpf_init2(tmp, prec);

	mpf_abs(abs_vec0, vec0);
	mpf_abs(abs_vec1, vec1);

	if(mpf_cmp_ui(abs_vec1, 0UL) == 0)
	{
		mpf_set_ui(cosine, 1UL);
		mpf_set_ui(sine, 0UL);
		return;
	}
	if(mpf_cmp_ui(abs_vec0, 0UL) == 0)
	{
		mpf_set_ui(cosine, 0UL);
		mpf_set_ui(sine, 1UL);
		mpf_set(vec0, vec1);
		mpf_set_ui(vec1, 0UL);
		return;
	}

	mpf_div(mu, vec0, abs_vec0);
	mpf_add(tau, abs_vec0, abs_vec1);
	mpf_div(tmp1, abs_vec0, tau);
	mpf_div(tmp2, abs_vec1, tau);
	mpf_mul(tmp, tmp1, tmp1);
	mpf_mul(nu, tmp2, tmp2);
	mpf_add(nu, nu, tmp);
	mpf_sqrt(nu, nu);
	mpf_mul(nu, nu, tau);

	/*
	*cosine = abs_vec0 / nu;
	*sine   = mu * (*vec1) / nu;
	*vec0 = nu * mu;
	*vec1 = 0.0;
	*/
	mpf_div(cosine, abs_vec0, nu);
	mpf_div(sine, vec1, nu);
	mpf_mul(sine, sine, mu);
	mpf_mul(vec0, nu, mu);
	mpf_set_ui(vec1, 0UL);

	/* free */
	mpf_clear(phi);
	mpf_clear(mu);
	mpf_clear(tau);
	mpf_clear(nu);
	mpf_clear(abs_vec1);
	mpf_clear(abs_vec0);
	mpf_clear(tmp1);
	mpf_clear(tmp2);
	mpf_clear(tmp);

	return;
}
#endif // SINGLE_USE

//int mpftriqr(MPFVector trimat[3], mpf_t shift, long int i1, long int i2)
void mpftriqr(MPFVector trimat[3], mpf_t shift, long int i1, long int i2)
{
	unsigned long prec;
	long int i;
	mpf_t g, sine, cosine, p, f, b, u, v, tmp;

	prec = trimat[1]->prec;

	/* Initialize */
	mpf_init2(g, prec);
	mpf_init2(sine, prec);
	mpf_init2(cosine, prec);
	mpf_init2(p, prec);
	mpf_init2(f, prec);
	mpf_init2(b, prec);
	mpf_init2(u, prec);
	mpf_init2(v, prec);
	mpf_init2(tmp, prec);

	/* g = get_mpfvector_i(trimat[1], i1) - shift; */
	mpf_sub(g, get_mpfvector_i(trimat[1], i1), shift);

	mpf_set_ui(sine, 1UL);
	mpf_set_ui(cosine, 1UL);
	mpf_set_ui(p, 0UL);

	for(i = i1; i < i2; i++)
	{
		mpf_mul(f, sine, get_mpfvector_i(trimat[0], i));
		mpf_mul(b, cosine, get_mpfvector_i(trimat[0], i));
		mpfplane_rotation(cosine, sine, g, f);
		if(i != i1)
			set_mpfvector_i(trimat[0], i - 1, g);
		/* 
		u = get_mpfvector_i(trimat[1], i) - p;
		v = (get_mpfvector_i(trimat[1], i + 1) - u) * sine + 2.0 * cosine * b;
		p = sine * v;
		set_mpfvector_i(trimat[1], i, u + p);
		g = cosine * v - b;
		*/
		mpf_set(u, get_mpfvector_i(trimat[1], i));
		mpf_sub(u, u, p);
		mpf_set(v, get_mpfvector_i(trimat[1], i + 1));
		mpf_sub(v, v, u);
		mpf_mul(v, v, sine);
		mpf_mul_ui(tmp, cosine, 2UL);
		mpf_mul(tmp, tmp, b);
		mpf_add(v, v, tmp);
		mpf_mul(p, sine, v);
		mpf_add(tmp, u, p);
		set_mpfvector_i(trimat[1], i, tmp);
		mpf_mul(g, cosine, v);
		mpf_sub(g, g, b);
//		printf("g, p, c, s, u, v:  %10.3e, %10.3e, %10.3e, %10.3e, %10.3e, %10.3e\n", g, p, cosine, sine, u, v);
	}

	mpf_sub(tmp, get_mpfvector_i(trimat[1], i2), p);
	set_mpfvector_i(trimat[1], i2, tmp);
	set_mpfvector_i(trimat[0], i2 - 1, g);

	/* free */
	mpf_clear(g);
	mpf_clear(sine);
	mpf_clear(cosine);
	mpf_clear(p);
	mpf_clear(f);
	mpf_clear(b);
	mpf_clear(u);
	mpf_clear(v);
	mpf_clear(tmp);
}

/* [mat11 mat12] */
/* [mat21 mat22] (mat21 = mat12) */
void mpfwilkinson_shift_tri(mpf_t ret, mpf_t mat11, mpf_t mat12, mpf_t mat22)
{
	unsigned long prec;
	mpf_t coef[3], shift[2], shift_im[2], tmp, tmp1;

	//prec = (mpf_get_prec(mat11) > mpf_get_prec(mat22)) ? mpf_get_prec(mat11) : mpf_get_prec(mat22);
	prec = mpf_get_prec(ret);

	/* Initialize */
	mpf_init2(coef[0], prec);
	mpf_init2(coef[1], prec);
	mpf_init2(coef[2], prec);
	mpf_init2(shift[0], prec);
	mpf_init2(shift[1], prec);
	mpf_init2(shift_im[0], prec);
	mpf_init2(shift_im[1], prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);

	mpf_set_ui(coef[2], 1UL);
	mpf_add(coef[1], mat11, mat22);
	mpf_neg(coef[1], coef[1]);
	mpf_mul(coef[0], mat11, mat22);
	mpf_mul(tmp, mat12, mat12);
	mpf_sub(coef[0], coef[0], tmp);

	mpfquadratic_eq(shift, shift_im, coef);

	mpf_sub(tmp, shift[0], mat22);
	mpf_abs(tmp, tmp);
	mpf_sub(tmp1, shift[1], mat22);
	mpf_abs(tmp1, tmp1);
	if(mpf_cmp(tmp, tmp1) <= 0)
		mpf_set(ret, shift[0]);
	else
		mpf_set(ret, shift[1]);

	/* Free */
	mpf_clear(coef[0]);
	mpf_clear(coef[1]);
	mpf_clear(coef[2]);
	mpf_clear(shift[0]);
	mpf_clear(shift[1]);
	mpf_clear(shift_im[0]);
	mpf_clear(shift_im[1]);
	mpf_clear(tmp);
	mpf_clear(tmp1);

	return;
}

int mpftriqr_iteration(MPFVector trimat[3], mpf_t rel_tol, mpf_t abs_tol, long int maxtimes)
{
	unsigned long prec;
	long int i, times, dim, index, flag, next_stop;
	mpf_t shift, tmp, tmp1;
	double terr, rerr;

	prec = trimat[1]->prec;
	dim = trimat[1]->dim;

#ifdef GET_MPFTRIQR_ERR
	char fname[128];
	FILE *fp;

	sprintf(fname, "mpftriqrerr_%04ddim_%04dbit.dat", dim, prec);
	fp = fopen(fname, "w");
#endif // GET_MPFTRIQR_ERR

	/* prec */
	mpf_init2(shift, prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);

	flag = 0;
	for(i = 0; i < dim - 1; i++)
	{
		index = dim - i - 1;
		next_stop = 0;
		for(times = 0; times < maxtimes; times++)
		{
			mpfwilkinson_shift_tri(shift, 
				get_mpfvector_i(trimat[1], index - 1),
				get_mpfvector_i(trimat[0], index - 1),
				get_mpfvector_i(trimat[1], index)
			);
			printf("shift(%5ld): %10.5e ", index, mpf_get_d(shift));
			//mpf_out_str(stdout, 10, 50, get_mpfvector_i(trimat[1], index));
			mpf_out_str(stdout, 10, 20, get_mpfvector_i(trimat[1], index));
			printf("\n");

			mpftriqr(trimat, shift, 0, index);

			/*
			if(fabs(get_mpfvector_i(trimat[0], index - 1)) <= rtol * sqrt(fabs(get_mpfvector_i(trimat[1], index - 1) * get_mpfvector_i(trimat[1], index))) + atol)
			*/
			mpf_abs(tmp, get_mpfvector_i(trimat[0], index - 1));
			mpf_mul(tmp1, get_mpfvector_i(trimat[1], index - 1), get_mpfvector_i(trimat[1], index));
			mpf_abs(tmp1, tmp1);
			mpf_sqrt(tmp1, tmp1);
			mpf_mul(tmp1, tmp1, rel_tol);
			mpf_add(tmp1, tmp1, abs_tol);
			if((mpf_cmp(tmp, tmp1) <= 0) && (times >= 2))
			{
#ifdef GET_MPFTRIQR_ERR
				fprintf(fp, "%10.3e %10.3e ", terr, rerr); 
#endif // GET_MPFTRIQR_ERR
				break;
			}
		}
		if(times >= maxtimes)
			fprintf(stderr, "WARNING: Not Convergent! (mpftriqr_iteration: %ld th, %ld deg\n", index, dim);

	}
#ifdef GET_MPFTRIQR_ERR
	fclose(fp);
#endif // GET_MPFTRIQR_ERR

	mpf_clear(shift);
	mpf_clear(tmp);
	mpf_clear(tmp1);

	return times;

}
/* Get eigenvector of tridiagonal matrices */
void mpfget_eigenvector_mpftri_cee(MPFVector eigen_vec, MPFVector trimat[3], mpf_t eigenvalue, long int drop_rank)
{
	long int i;
	unsigned long prec;
	mpf_t l, d, u, ev, evm1, evm2, tmp_ev1, tmp_ev2;

	if(drop_rank <= 0)
		return;

	prec = eigen_vec->prec;

	mpf_init2(l, prec);
	mpf_init2(d, prec);
	mpf_init2(u, prec);
	mpf_init2(ev, prec);
	mpf_init2(evm1, prec);
	mpf_init2(evm2, prec);
	mpf_init2(tmp_ev1, prec);
	mpf_init2(tmp_ev2, prec);

	/* set 1 */
	for(i = 0; i < drop_rank; i++)
		set_mpfvector_i_ui(eigen_vec, i, 1UL);

	mpf_set(evm1, get_mpfvector_i(eigen_vec, drop_rank - 1));
	if(drop_rank >= 2)
	{
		mpf_set(l, get_mpfvector_i(trimat[2], drop_rank - 2));
		mpf_set(evm2, get_mpfvector_i(eigen_vec, drop_rank - 2));
	}
	else
	{
		mpf_set_ui(l, 0UL);
		mpf_set_ui(evm2, 0UL);
	}
	for(i = drop_rank; i < trimat[1]->dim; i++)
	{
		mpf_set(d, get_mpfvector_i(trimat[1], i - 1));
		mpf_sub(d, d, eigenvalue);
		mpf_set(u, get_mpfvector_i(trimat[0], i - 1));

		//ev = -(l * evm2 + d * evm1) / u;
		mpf_mul(tmp_ev2, l, evm2);
		mpf_mul(tmp_ev1, d, evm1);
		mpf_add(ev, tmp_ev2, tmp_ev1);
		mpf_neg(ev, ev);
		mpf_div(ev, ev, u);

		set_mpfvector_i(eigen_vec, i, ev);

		mpf_set(evm2, evm1);
		mpf_set(evm1, ev);
		mpf_set(l, get_mpfvector_i(trimat[2], i - 1));
//		printf("---%d, %25.17e\n", i, ev);
	}

	/* ||eiven_vec||_2 = 1 */
	norm2_mpfvector(ev, eigen_vec);
	for(i = 0; i < trimat[1]->dim; i++)
	{
		mpf_div(tmp_ev1, get_mpfvector_i(eigen_vec, i), ev);
		set_mpfvector_i(eigen_vec, i, tmp_ev1);
	}

	mpf_clear(l);
	mpf_clear(d);
	mpf_clear(u);
	mpf_clear(evm1);
	mpf_clear(evm2);
	mpf_clear(ev);
	mpf_clear(tmp_ev1);
	mpf_clear(tmp_ev2);
}

/* Multiply diagonal matrix */
void mul_mpfdiagmat_mpfvec(MPFVector ret, MPFVector diagmat, MPFVector vec)
{
	long int i, dim;
	unsigned long prec;
	mpf_t d, v, tmp;

	prec = ret->prec;

	mpf_init2(d, prec);
	mpf_init2(v, prec);
	mpf_init2(tmp, prec);

	dim = (diagmat->dim < vec->dim) ? diagmat->dim : vec->dim;
	for(i = 0; i < dim; i++)
	{
		mpf_set(d, get_mpfvector_i(diagmat, i)); // d = a_ii
		mpf_set(v, get_mpfvector_i(vec, i)); 
		mpf_mul(tmp, d, v);
		set_mpfvector_i(ret, i, tmp);
	}

	mpf_clear(d);
	mpf_clear(v);
	mpf_clear(tmp);
}
#endif // USE_GMP
