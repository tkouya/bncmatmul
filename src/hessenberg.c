/****************************************************************************/
/* hessenberg.c: for eigenproblems of real unsymmetric dense matrices       */
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

//#include "linear_opt.c"

/***************************************************/
/* IEEE754 double precision                        */
/***************************************************/

/* Transform Real Square Matric to Hessemberg Form */
int dhessenberg(DMatrix mat, DMatrix proj_mat, int flag_get_proj_mat)
{
	long int n, i, j, k, dim;
	double c, s, s2, mat_n1n, tmp[3];
	DVector w, caw, catw, q, qd;

	dim = mat->row_dim;
	if(dim <= 0 || dim != mat->col_dim)
		return 0;

	if(proj_mat == NULL)
		flag_get_proj_mat = 0;

	/* Initialize */
	w    = init_dvector(dim);
	q    = init_dvector(dim);
	qd   = init_dvector(dim);
	catw = init_dvector(dim);
	caw  = init_dvector(dim);

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
		set0_dvector(w);
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

		/* caw  := c * A * w */
		/* catw := c * A^T * w */
		for(i = 0; i < dim; i++)
		{
			tmp[0] = 0.0;
			tmp[1] = 0.0;
			for(j = n + 1; j < dim; j++)
			{
				tmp[0] += gdmij(mat, i, j) * gdvi(w, j);
				tmp[1] += gdmij(mat, j, i) * gdvi(w, j);
			}
			tmp[0] *= c;
			tmp[1] *= c;
			sdvi(caw , i, tmp[0]);
			sdvi(catw, i, tmp[1]);
		}

		/* q := caw  := caw  - c / 2 * (w * caw^T) * w */
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

		/* q' := catw := catw - c / 2 * (w * catw^T) * w */
		for(i = 0; i <= n; i++)
			sdvi(qd, i, gdvi(catw, i));
		for(i = n + 1; i < dim; i++)
		{
			tmp[1] = 0.0;
			for(j = n + 1; j < dim; j++)
				tmp[1] += gdvi(w, i) * gdvi(catw, j) * gdvi(w, j);
			tmp[1] *= c / 2.0;
			tmp[0] = gdvi(catw, i) - tmp[1];
			sdvi(qd, i, tmp[0]);
		}

		/* P_i * A P_i = A - q * w^T - w^T * q' */

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
		/* (2) A'' := A' - w * q'^T */
		for(i = n + 1; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				tmp[1] = gdvi(w, i) * gdvi(qd, j);
				tmp[2] = gdmij(mat, i, j) - tmp[1];
				sdmij(mat, i, j, tmp[2]);
			}
		}
		/* Nonsense process (due to my lack of knowledge) */
		for(i = n + 2; i < dim; i++)
			sdmij(mat, i, n, 0.0);
	}

	/* clear */
	free_dvector(w);
	free_dvector(q);
	free_dvector(qd);
	free_dvector(catw);
	free_dvector(caw);

	return dim;
}

/* Generate vector for Householder transformation        */
/* H = I - ret_u * ret_u^T, H * a = mu * [1 0 ... 0]^T } */
double dhousevec(DVector ret_u, DVector a)
{
	double rho, u0, mu;

	subst_dvector(ret_u, a);
	mu = norm2_dvector(a);

	if(mu == 0.0)
	{
		sdvi(ret_u, 0, sqrt(2.0));
		return mu;
	}

	u0 = gdvi(ret_u, 0);
	if(u0 != 0.0)
		rho = u0 / fabs(u0);
	else
		rho = 1.0;

	cmul_dvector(ret_u, rho / mu, ret_u);

	u0 = gdvi(ret_u, 0);
	sdvi(ret_u, 0, 1.0 + u0);

	u0 = gdvi(ret_u, 0);
	cmul_dvector(ret_u, 1.0 / sqrt(u0), ret_u);

	mu *= -rho;

	return mu;
}

/* backsearch for Hessenberg matrix */
void backsearch2(long int *ret_i1, long int *ret_i2, long int start_index, DMatrix hmat, double eps)
{
	long int i1, i2;
	double eig_re[2], eig_im[2], coef[3];

	i1 = start_index;
	i2 = start_index;

	while(i1 > 0)
	{
		//if((fabs(gdmij(hmat, i1 - 1, i1)) < eps) || (i2 == 1))
		if((fabs(gdmij(hmat, i1, i1 - 1)) < eps) || (i2 == 1))
		{
			if(i2 != 1)
				//sdmij(hmat, i1 - 1, i1, 0.0);
				sdmij(hmat, i1, i1 - 1, 0.0);

			if((i1 == (i2 - 1)) || (i2 == 1))
			{
				/* Process 2 * 2 block */
				/* h(i2-1,i2-1) h(i2-1,i2) */
				/* h(i2  ,i2-1) h(i2  ,i2) */
				coef[2] = 1.0;
				coef[1] = -(gdmij(hmat, i2-1, i2-1) + gdmij(hmat, i2, i2));
				coef[0] = gdmij(hmat, i2-1, i2-1) * gdmij(hmat, i2, i2) - gdmij(hmat, i2-1, i2) * gdmij(hmat, i2, i2 - 1);
				dquadratic_eq(eig_re, eig_im, coef);
				printf("eig[%ld] = (%25.17e, %25.17e)\n", i2-1, eig_re[0], eig_im[0]);
				printf("eig[%ld] = (%25.17e, %25.17e)\n", i2, eig_re[1], eig_im[1]);
				
				if(i2 != 1)
				{
					i1 = i1 - 1;
					i2 = i1;
				}
				else
				{
					i1 = 0;
					i2 = 0;
				}
			}
			else if(i1 == i2)
			{
				i1 = i1 - 1;
				i2 = i1;
			}
			else
				goto end;
		}
		else
			i1 = i1 - 1;
	}
end:

	*ret_i1 = i1;
	*ret_i2 = i2;

}

/* Householder Transformation for Real Hessenberg Matrix */
double start_dqrh(DVector u, double h11, double h12, double h21, double h22, double h32, double hn1n1, double hn1n, double hnn1, double hnn)
{
	double s, ret, p, q, r, tmp;
	DVector c;

	c = init_dvector(3);

	/* s = 1 / max(|h11|, ..., |hnn| */
	s = fabs(h11);
	tmp = fabs(h12)  ; if(s < tmp) s = tmp;
	tmp = fabs(h21)  ; if(s < tmp) s = tmp;
	tmp = fabs(h22)  ; if(s < tmp) s = tmp;
	tmp = fabs(h32)  ; if(s < tmp) s = tmp;
	tmp = fabs(hn1n1); if(s < tmp) s = tmp;
	tmp = fabs(hn1n) ; if(s < tmp) s = tmp;
	tmp = fabs(hnn1) ; if(s < tmp) s = tmp;
	tmp = fabs(hnn)  ; if(s < tmp) s = tmp;
	s = 1.0 / s;

	/* h11 *= s, ..., hnn *= s */
	h11   *= s;
	h12   *= s;
	h21   *= s;
	h22   *= s;
	h32   *= s;
	hn1n1 *= s;
	hn1n  *= s;
	hnn1  *= s;
	hnn   *= s;

	p = hnn   - h11;
	q = hn1n1 - h11;
	r = h22   - h11;

	sdvi(c, 0, (p * q - hnn1 * hn1n) / h21 + h12);
	sdvi(c, 1, r - p - q);
	sdvi(c, 2, h32);

	ret = dhousevec(u, c);

	free_dvector(c);

	// fix by Golub
	return ret;
}

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

/* Double-shift QR method : Main loop */
int dqrh(DMatrix hmat, DMatrix qmat, long int i1, long int i2, DVector u, double mu, int flag_get_proj_mat)
{
	long int i, j, k, l, iu, dim;
	double tmp, in_mu, c, s, tmp_t, tmp_x, tmp_y;
	DVector v, tmp_u;

	dim = hmat->row_dim;

	v = init_dvector(dim);
	tmp_u = init_dvector(3);

	/* set in_mu */
	in_mu = mu;

	for(i = i1; i <= i2 - 2; i++)
	{
		//printf("(i1,i2) = (%d, %d) -> mu = %15.7e, ||u||=%15.7e\n", i1, i2, in_mu, norm2_dvector(u));
		//printf("u = [%g %g %g]\n", gdvi(u, 0), gdvi(u, 1), gdvi(u, 2));
		in_mu = 1.0;

		j = ((i - 1) > i1) ? (i - 1) : i1;

		/* v^T = u^T * H[i:i+2, j:n] */
		for(l = j; l < dim; l++)
		{
			tmp = 0.0;
			for(k = 0; k < 3; k++)
				tmp += gdvi(u, k) * gdmij(hmat, k + i, l);
			sdvi(v, l, tmp);
		}
		/* H[i:i+2, j:n] = H[i:i+2, j:n] - mu * u * v^T */
		for(l = j; l < dim; l++)
		{
			for(k = 0; k < 3; k++)
			{
				tmp = gdmij(hmat, k + i, l) - in_mu * gdvi(u, k) * gdvi(v, l);
				sdmij(hmat, k + i, l, tmp);
			}
		}

		iu = ((i + 3) < i2) ? (i + 3) : i2;

		/* v = H[1:iu, i:i+2] * u */
		for(l = 0; l <= iu; l++)
		{
			tmp = 0.0;
			for(k = 0; k < 3; k++)
				tmp += gdmij(hmat, l, k + i) * gdvi(u, k);
			sdvi(v, l, tmp);
		}
		/* H[1:iu, i:i+2] = H[1:iu, i:i+2] - mu * v * u^T */
		for(l = 0; l <= iu; l++)
		{
			for(k = 0; k < 3; k++)
			{
				tmp = gdmij(hmat, l, k + i) - in_mu * gdvi(v, l) * gdvi(u, k);
				sdmij(hmat, l, k + i, tmp);
			}
		}

		if(flag_get_proj_mat == 1)
		{
			/* v = Q[1:n, i:i+2] * u */
			for(l = 0; l < dim; l++)
			{
				tmp = 0.0;
				for(k = 0; k < 3; k++)
					tmp += gdmij(qmat, l, k + i) * gdvi(u, k);
				sdvi(v, l, tmp);
			}
			/* Q[1:n, i:i+2] = Q[1:n, i:i+2] - mu * v * u^T */
			for(l = 0; l < dim; l++)
			{
				for(k = 0; k < 3; k++)
				{
					tmp = gdmij(qmat, l, k + i) - in_mu * gdvi(v, l) * gdvi(u, k);
					sdmij(qmat, l, k + i, tmp);
				}
			}
		}

		if(i != i2 - 2)
		{
			for(k = 0; k < 3; k++)
				sdvi(tmp_u, k, gdmij(hmat, i + 1 + k, i));
			in_mu = dhousevec(u, tmp_u);
		}
		if(i != i1)
		{
			sdmij(hmat, i + 1, j, 0.0);
			sdmij(hmat, i + 2, j, 0.0);
		}
	}

	tmp_x = gdmij(hmat, i2 - 1, i2 - 2);
	tmp_y = gdmij(hmat, i2, i2 - 2);
	dplane_rotation(&c, &s, &tmp_x, &tmp_y);
	sdmij(hmat, i2 - 1, i2 - 2, tmp_x);
	sdmij(hmat, i2, i2 - 2, tmp_y);

	/* rotapp(c, s, H[i2-1, i2-1:n], H[i2, i2-1:n]) */
	for(i = i2 - 1; i < dim; i++)
	{
		tmp_x = gdmij(hmat, i2 - 1, i);
		tmp_y = gdmij(hmat, i2    , i);

		tmp_t = c * tmp_x + s * tmp_y;
		tmp_y = c * tmp_y - s * tmp_x;
		tmp_x = tmp_t;

		sdmij(hmat, i2 - 1, i, tmp_x);
		sdmij(hmat, i2    , i, tmp_y);
	}

	/* rotapp(c, s, H[1:i2, i2-1]  , H[1:i2, i2]  ) */
	for(i = 0; i <= i2; i++)
	{
		tmp_x = gdmij(hmat, i, i2 - 1);
		tmp_y = gdmij(hmat, i, i2    );

		tmp_t = c * tmp_x + s * tmp_y;
		tmp_y = c * tmp_y - s * tmp_x;
		tmp_x = tmp_t;

		sdmij(hmat, i, i2 - 1, tmp_x);
		sdmij(hmat, i, i2    , tmp_y);
	}

	if(flag_get_proj_mat == 1)
	{
		/* rotapp(c, s, Q[1:n, i2-1]   , Q[1:n, i2]   ) */
		for(i = 0; i < dim; i++)
		{
			tmp_x = gdmij(qmat, i, i2 - 1);
			tmp_y = gdmij(qmat, i, i2    );

			tmp_t = c * tmp_x + s * tmp_y;
			tmp_y = c * tmp_y - s * tmp_x;
			tmp_x = tmp_t;

			sdmij(qmat, i, i2 - 1, tmp_x);
			sdmij(qmat, i, i2    , tmp_y);
		}
	}

	free_dvector(v);
	free_dvector(tmp_u);

	return SUCCESS;
}

/* Double QR Method for Real Hessenberg Matrix */
/* hmat := [ x x x ... x x x ] */
/*         [ x x x ... x x x ] */
/*         [ 0 x x ... x x x ] */
/*         [ ............... ] */
/*         [ 0 0 0 ... x x x ] */
/*         [ 0 0 0 ... 0 x x ] */
int dqrh_iteration(DMatrix hmat, DMatrix qmat, int flag_get_proj_mat, long int maxtimes, double reps)
{
	long int i1, i2, old_i2, dim, times;
	double eps, mu;
	DVector u;

	dim = hmat->row_dim;
	if(dim <= 0 || (dim != hmat->col_dim))
		return -1;

	u = init_dvector(3);

	i1 = 0;
	i2 = dim - 1;

	if(qmat == NULL)
		flag_get_proj_mat = 0;

	if(flag_get_proj_mat == 1)
		setI_dmatrix(qmat);

	/* norm1_dmatrix(hmat) * reps */
	eps = norm1_dmatrix(hmat) * reps;

	for(times = 0; times < maxtimes; times++)
	{
		old_i2 = i2;
		backsearch2(&i1, &i2, i2, hmat, eps);

		printf("(i1, i2) = (%ld, %ld)\n", i1, i2);

		if(i2 == 0)
			break;

		mu = start_dqrh(
			u,
			gdmij(hmat, i1    , i1    ),
			gdmij(hmat, i1    , i1 + 1),
			gdmij(hmat, i1 + 1, i1    ),
			gdmij(hmat, i1 + 1, i1 + 1),
			gdmij(hmat, i1 + 2, i1 + 1),
			gdmij(hmat, i2 - 1, i2 - 1),
			gdmij(hmat, i2 - 1, i2    ),
			gdmij(hmat, i2    , i2 - 1),
			gdmij(hmat, i2    , i2    )
		);
	
		dqrh(hmat, qmat, i1, i2, u, mu, flag_get_proj_mat);
	}

	free_dvector(u);

	return times;
}

/***************************************************/
/* MPFR or GMP                                     */
/***************************************************/

#ifdef USE_GMP
/* Transform Real Square Matric to Hessemberg Form */
int mpfhessenberg(MPFMatrix mat, MPFMatrix proj_mat, int flag_get_proj_mat)
{
	unsigned long prec;
	long int n, i, j, k, dim;
	mpf_t c, s, s2, mat_n1n, tmp[3];
	MPFVector w, caw, catw, q, qd;

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
	qd   = init2_mpfvector(dim, prec);
	catw = init2_mpfvector(dim, prec);
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
		/* catw := c * A^T * w */
		for(i = 0; i < dim; i++)
		{
			mpf_set_ui(tmp[0], 0UL);
			mpf_set_ui(tmp[1], 0UL);
			for(j = n + 1; j < dim; j++)
			{
				/* tmp[0] += gmpfmij(mat, i, j) * gmpfvi(w, j) */
				mpf_mul(tmp[2], gmpfmij(mat, i, j), gmpfvi(w, j));
				mpf_add(tmp[0], tmp[0], tmp[2]);

				/* tmp[1] += gmpfmij(mat, j, i) * gmpfvi(w, j) */
				mpf_mul(tmp[2], gmpfmij(mat, j, i), gmpfvi(w, j));
				mpf_add(tmp[1], tmp[1], tmp[2]);
			}
			mpf_mul(tmp[0], tmp[0], c);
			mpf_mul(tmp[1], tmp[1], c);
			smpfvi(caw , i, tmp[0]);
			smpfvi(catw, i, tmp[1]);
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

		/* q' := catw := catw - c / 2 * (w * catw^T) * w */
		for(i = 0; i <= n; i++)
			smpfvi(qd, i, gmpfvi(catw, i));

		for(i = n + 1; i < dim; i++)
		{
			mpf_set_ui(tmp[1], 0UL);
			for(j = n + 1; j < dim; j++)
			{
				/* tmp[1] += gmpfvi(w, i) * gmpfvi(catw, j) * gmpfvi(w, j) */
				mpf_mul(tmp[2], gmpfvi(w, i), gmpfvi(catw, j));
				mpf_mul(tmp[2], tmp[2], gmpfvi(w, j));
				mpf_add(tmp[1], tmp[1], tmp[2]);
			}
			/* tmp[1] *= c / 2.0 */
			mpf_div_ui(tmp[2], c, 2UL);
			mpf_mul(tmp[1], tmp[1], tmp[2]);
			
			mpf_sub(tmp[0], gmpfvi(catw, i), tmp[1]);
			smpfvi(qd, i, tmp[0]);
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
		/* (2) A'' := A' - w * q'^T */
		for(i = n + 1; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				mpf_mul(tmp[1], gmpfvi(w, i), gmpfvi(qd, j));
				mpf_sub(tmp[2], gmpfmij(mat, i, j), tmp[1]);
				smpfmij(mat, i, j, tmp[2]);
			}
		}

		/* Nonsense process (due to my lack of knowledge) */
		mpf_set_ui(tmp[0], 0UL);
		for(i = n + 2; i < dim; i++)
			smpfmij(mat, i, n, tmp[0]);
	}

	/* clear */
	mpf_clear(c);
	mpf_clear(s);
	mpf_clear(s2);
	mpf_clear(mat_n1n);
	for(i = 0; i < 3; i++) mpf_clear(tmp[i]);

	free_mpfvector(w);
	free_mpfvector(q);
	free_mpfvector(qd);
	free_mpfvector(catw);
	free_mpfvector(caw);

	return dim;
}

/* Generate vector for Householder transformation        */
/* H = I - ret_u * ret_u^T, H * a = mu * [1 0 ... 0]^T } */
void mpfhousevec(mpf_t mu, MPFVector ret_u, MPFVector a)
{
	unsigned long prec;
	mpf_t rho, u0, tmp;

	prec = ret_u->prec;
	mpf_init2(rho, prec);
	mpf_init2(u0 , prec);
	mpf_init2(tmp, prec);

	subst_mpfvector(ret_u, a);
	norm2_mpfvector(mu, a);

	if(mpf_cmp_ui(mu, 0UL) == 0)
	{
		mpf_sqrt_ui(tmp, 2UL);
		smpfvi(ret_u, 0, tmp);
		return;
	}

	mpf_set(u0, gmpfvi(ret_u, 0));
	if(mpf_cmp_ui(u0, 0UL) != 0)
	{
		/* rho = u0 / fabs(u0) */
		mpf_abs(tmp, u0);
		mpf_div(rho, u0, tmp);
	}
	else
		mpf_set_ui(rho, 1UL);

	mpf_div(tmp, rho, mu);
	cmul_mpfvector(ret_u, tmp, ret_u);

	mpf_set(u0, gmpfvi(ret_u, 0));
	mpf_add_ui(tmp, u0, 1UL);
	smpfvi(ret_u, 0, tmp);

	mpf_set(u0, gmpfvi(ret_u, 0));
	mpf_sqrt(tmp, u0);
	mpf_ui_div(tmp, 1UL, tmp);
	cmul_mpfvector(ret_u, tmp, ret_u);

	/* *mu *= -rho */
	mpf_neg(tmp, rho);
	mpf_mul(mu, mu, tmp);

	mpf_clear(rho);
	mpf_clear(u0);
	mpf_clear(tmp);
}

/* backsearch for Hessenberg matrix */
void mpfbacksearch2(long int *ret_i1, long int *ret_i2, long int start_index, MPFMatrix hmat, mpf_t eps)
{
	long int i1, i2;
	mpf_t zero, tmp, tmp_eps;
	mpf_t coef[3], eig_re[2], eig_im[2];

	/* Initialize */
	mpf_init2(tmp, hmat->prec);
	mpf_init2(zero, hmat->prec); mpf_set_ui(zero, 0UL);
	mpf_init2(tmp_eps, hmat->prec); mpf_set_str(tmp_eps, "1.0e-100", 10);
	mpf_init2(coef[0], hmat->prec);
	mpf_init2(coef[1], hmat->prec);
	mpf_init2(coef[2], hmat->prec);
	mpf_init2(eig_re[0], hmat->prec);
	mpf_init2(eig_re[1], hmat->prec);
	mpf_init2(eig_im[0], hmat->prec);
	mpf_init2(eig_im[1], hmat->prec);

	i1 = start_index;
	i2 = start_index;

	while(i1 > 0)
	{
//		printf("(i1, i2) : (%d, %d)", *i1, *i2);

		/* if((fabs(gmpfmij(hmat, *i1 - 1, *i1)) < eps) || (*i2 == 1)) */
		//mpf_abs(tmp, gmpfmij(hmat, *i1 - 1, *i1));
		mpf_abs(tmp, gmpfmij(hmat, i1, i1 - 1));
		if((mpf_cmp(tmp, eps) < 0) || (i2 == 1))
		{
			if(i2 != 1)
			{
				//smpfmij(hmat, i1 - 1, i1, tmp);
				smpfmij(hmat, i1, i1 - 1, zero);
			}

			if((i1 == (i2 - 1)) || (i2 == 1))
			{
				/* Processing 2x2 block */
				//mpf_mul(tmp, gmpfmij(hmat, i1, i2), gmpfmij(hmat, i2, i1));
				//smpfmij(hmat, i1, i2, tmp);
				//smpfmij(hmat, i2, i1, zero);
				/* Process 2 * 2 block */
				/* h(i2-1,i2-1) h(i2-1,i2) */
				/* h(i2  ,i2-1) h(i2  ,i2) */
				mpf_set_ui(coef[2], 1UL);
				//coef[1] = -(gdmij(hmat, *i2-1, *i2-1) + gdmij(hmat, *i2, *i2));
				mpf_add(coef[1], gmpfmij(hmat, i2-1, i2-1), gmpfmij(hmat, i2, i2));
				mpf_neg(coef[1], coef[1]);
				//coef[0] = gdmij(hmat, *i2-1, *i2-1) * gdmij(hmat, *i2, *i2) - gdmij(hmat, *i2-1, *i2) * gdmij(hmat, *i2, *i2-1);
				mpf_mul(coef[0], gmpfmij(hmat, i2-1, i2-1), gmpfmij(hmat, i2, i2  ));
				mpf_mul(tmp    , gmpfmij(hmat, i2-1, i2  ), gmpfmij(hmat, i2, i2-1));
				mpf_sub(coef[0], coef[0], tmp);

				mpfquadratic_eq(eig_re, eig_im, coef);
				printf("eig[%ld] = (", i2-1);
				mpf_out_str(stdout, 10, 0, eig_re[0]);
				printf(", ");
				mpf_out_str(stdout, 10, 0, eig_im[0]);
				printf(")\n");
				printf("eig[%ld] = (", i2);
				mpf_out_str(stdout, 10, 0, eig_re[1]);
				printf(", ");
				mpf_out_str(stdout, 10, 0, eig_im[1]);
				printf(")\n");

				if(i2 != 1)
				{
					i1 = i1 - 1;
					i2 = i1;
				}
				else
				{
					i1 = 0;
					i2 = 0;
				}
			}
			else if(i1 == i2)
			{
				i1 = i1 - 1;
				i2 = i1;
			}
			else
				goto end;
		}
		else
			i1 = i1 - 1;
	
//		printf(" -> (%d, %d)\n", i1, i2);
	}
end:
	/* clear */
	mpf_clear(tmp);
	mpf_clear(tmp_eps);
	mpf_clear(zero);
	mpf_clear(coef[0]);
	mpf_clear(coef[1]);
	mpf_clear(coef[2]);
	mpf_clear(eig_re[0]);
	mpf_clear(eig_re[1]);
	mpf_clear(eig_im[0]);
	mpf_clear(eig_im[1]);

	*ret_i1 = i1;
	*ret_i2 = i2;
}

/* Householder Transformation for Real Hessenberg Matrix */
void start_mpfqrh(mpf_t mu, MPFVector u, mpf_t o_h11, mpf_t o_h12, mpf_t o_h21, mpf_t o_h22, mpf_t o_h32, mpf_t o_hn1n1, mpf_t o_hn1n, mpf_t o_hnn1, mpf_t o_hnn)
{
	unsigned long prec;
	mpf_t s, tmp, p, q, r;
	mpf_t h11, h12, h21, h22, h32, hn1n1, hn1n, hnn1, hnn;
	MPFVector c;

	/* Initialize */
	prec = u->prec;

	mpf_init2(h11  , prec); mpf_set(h11  , o_h11  );
	mpf_init2(h12  , prec); mpf_set(h12  , o_h12  );
	mpf_init2(h21  , prec); mpf_set(h21  , o_h21  );
	mpf_init2(h22  , prec); mpf_set(h22  , o_h22  );
	mpf_init2(h32  , prec); mpf_set(h32  , o_h32  );
	mpf_init2(hn1n1, prec); mpf_set(hn1n1, o_hn1n1);
	mpf_init2(hn1n , prec); mpf_set(hn1n , o_hn1n );
	mpf_init2(hnn1 , prec); mpf_set(hnn1 , o_hnn1 );
	mpf_init2(hnn  , prec); mpf_set(hnn  , o_hnn  );

	mpf_init2(s, prec);
	mpf_init2(tmp, prec);
	mpf_init2(p, prec);
	mpf_init2(q, prec);
	mpf_init2(r, prec);

	c = init2_mpfvector(3, prec);

	/* s = 1 / max(|h11|, ..., |hnn| */
	mpf_abs(s  , h11);
	mpf_abs(tmp, h12)  ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, h21)  ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, h22)  ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, h32)  ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, hn1n1); if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, hn1n) ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, hnn1) ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	mpf_abs(tmp, hnn)  ; if(mpf_cmp(s, tmp) < 0) mpf_set(s, tmp);
	/* s = 1.0 / s */
	mpf_ui_div(s, 1UL, s);

	/* h11 *= s, ..., hnn *= s */
	mpf_mul(h11  , h11  , s);
	mpf_mul(h12  , h12  , s);
	mpf_mul(h21  , h21  , s);
	mpf_mul(h22  , h22  , s);
	mpf_mul(h32  , h32  , s);
	mpf_mul(hn1n1, hn1n1, s);
	mpf_mul(hn1n , hn1n , s);
	mpf_mul(hnn1 , hnn1 , s);
	mpf_mul(hnn  , hnn  , s);

	mpf_sub(p, hnn  , h11);
	mpf_sub(q, hn1n1, h11);
	mpf_sub(r, h22  , h11);

	/* smpfvi(c, 0, (p * q - hnn1 * hn1n) / h21 + h12) */
	mpf_mul(s, p, q);
	mpf_mul(tmp, hnn1, hn1n);
	mpf_sub(tmp, s, tmp);
	mpf_div(tmp, tmp, h21);
	mpf_add(tmp, tmp, h12);
	smpfvi(c, 0, tmp);

	/* smpfvi(c, 1, r - p - q) */
	mpf_sub(tmp, r, p);
	mpf_sub(tmp, tmp, q);
	smpfvi(c, 1, tmp);

	smpfvi(c, 2, h32);

	mpfhousevec(mu, u, c);

	/* clear */
	mpf_clear(s);
	mpf_clear(tmp);
	mpf_clear(p);
	mpf_clear(q);
	mpf_clear(r);

	mpf_clear(h11);
	mpf_clear(h12);
	mpf_clear(h21);
	mpf_clear(h22);
	mpf_clear(h32);
	mpf_clear(hn1n1);
	mpf_clear(hn1n);
	mpf_clear(hnn1);
	mpf_clear(hnn);

	free_mpfvector(c);
}

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

/* mpf_t-shift QR method : Main loop */
int mpfqrh(MPFMatrix hmat, MPFMatrix qmat, long int i1, long int i2, MPFVector u, mpf_t mu, int flag_get_proj_mat)
{
	unsigned long prec;
	long int i, j, k, l, iu, dim;
	mpf_t tmp, tmp1, c, s, tmp_t, tmp_x, tmp_y;
	MPFVector v, tmp_u;

	dim = hmat->row_dim;

	/* Initialize */
	prec = hmat->prec;

	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(c, prec);
	mpf_init2(s, prec);
	mpf_init2(tmp_t, prec);
	mpf_init2(tmp_x, prec);
	mpf_init2(tmp_y, prec);

	v = init2_mpfvector(dim, prec);
	tmp_u = init2_mpfvector(3, prec);

	/* to keep Hessenberg form */
	for(i = i1; i <= i2 - 2; i++)
	{
		j = ((i - 1) > i1) ? (i - 1) : i1;

		/* v^T = u^T * H[i:i+2, j:n] */
		for(l = j; l < dim; l++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < 3; k++)
			{
				/* tmp += gmpfvi(u, k) * gmpfmij(hmat, k + i, l) */
				mpf_mul(tmp1, gmpfvi(u, k), gmpfmij(hmat, k + i, l));
				mpf_add(tmp, tmp, tmp1);
			}
			smpfvi(v, l, tmp);
		}
		/* H[i:i+2, j:n] = H[i:i+2, j:n] - u * v^T */
		for(l = j; l < dim; l++)
		{
			for(k = 0; k < 3; k++)
			{
				/* tmp = gmpfmij(hmat, k + i, l) - gmpfvi(u, k) * gmpfvi(v, l) */
				mpf_mul(tmp1, gmpfvi(u, k), gmpfvi(v, l));
				mpf_sub(tmp, gmpfmij(hmat, k + i, l), tmp1);

				smpfmij(hmat, k + i, l, tmp);
			}
		}

		iu = ((i + 3) < i2) ? (i + 3) : i2;

		/* v = H[1:iu, i:i+2] * u */
		for(l = 0; l <= iu; l++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < 3; k++)
			{
				/* tmp += gmpfmij(hmat, l, k + i) * gmpfvi(u, k) */
				mpf_mul(tmp1, gmpfmij(hmat, l, k + i), gmpfvi(u, k));
				mpf_add(tmp, tmp, tmp1);
			}
			smpfvi(v, l, tmp);
		}
		/* H[1:iu, i:i+2] = H[1:iu, i:i+2] - v * u^T */
		for(l = 0; l <= iu; l++)
		{
			for(k = 0; k < 3; k++)
			{
				/* tmp = gmpfmij(hmat, l, k + i) - gmpfvi(v, l) * gmpfvi(u, k) */
				mpf_mul(tmp1, gmpfvi(v, l), gmpfvi(u, k));
				mpf_sub(tmp, gmpfmij(hmat, l, k + i), tmp1);

				smpfmij(hmat, l, k + i, tmp);
			}
		}

		if(flag_get_proj_mat == 1)
		{
			/* v = Q[1:n, i:i+2] * u */
			for(l = 0; l < dim; l++)
			{
				mpf_set_ui(tmp, 0UL);
				for(k = 0; k < 3; k++)
				{
					/* tmp += gmpfmij(qmat, l, k + i) * gmpfvi(u, k) */
					mpf_mul(tmp1, gmpfmij(qmat, l, k + i), gmpfvi(u, k));
					mpf_add(tmp, tmp, tmp1);
				}
				smpfvi(v, l, tmp);
			}
			/* Q[1:n, i:i+2] = Q[1:n, i:i+2] - v * u^T */
			for(l = 0; l < dim; l++)
			{
				for(k = 0; k < 3; k++)
				{
					/* tmp = gdmij(qmat, l, k + i) - gdvi(v, l) * gdvi(u, k) */
					mpf_mul(tmp1, gmpfvi(v, l), gmpfvi(u, k));
					mpf_sub(tmp, gmpfmij(qmat, l, k + i), tmp1);
					smpfmij(qmat, l, k + i, tmp);
				}
			}
		}

		if(i != i2 - 2)
		{
			for(k = 0; k < 3; k++)
				smpfvi(tmp_u, k, gmpfmij(hmat, i + 1 + k, i));
			mpfhousevec(tmp, u, tmp_u);
		}
		if(i != i1)
		{
			mpf_set_ui(tmp, 0UL);
			smpfmij(hmat, i + 1, j, tmp);
			smpfmij(hmat, i + 2, j, tmp);
		}
	}

	/* QR -> RQ */

	mpf_set(tmp_x, gmpfmij(hmat, i2 - 1, i2 - 2));
	mpf_set(tmp_y, gmpfmij(hmat, i2, i2 - 2));
	mpfplane_rotation(c, s, tmp_x, tmp_y);
	smpfmij(hmat, i2 - 1, i2 - 2, tmp_x);
	smpfmij(hmat, i2, i2 - 2, tmp_y);

	/* rotapp(c, s, H[i2-1, i2-1:n], H[i2, i2-1:n]) */
	for(i = i2 - 1; i < dim; i++)
	{
		mpf_set(tmp_x, gmpfmij(hmat, i2 - 1, i));
		mpf_set(tmp_y, gmpfmij(hmat, i2    , i));

		/* tmp_t = c * tmp_x + s * tmp_y */
		mpf_mul(tmp , c, tmp_x);
		mpf_mul(tmp1, s, tmp_y);
		mpf_add(tmp_t, tmp, tmp1);

		/* tmp_y = c * tmp_y - s * tmp_x */
		mpf_mul(tmp , c, tmp_y);
		mpf_mul(tmp1, s, tmp_x);
		mpf_sub(tmp_y, tmp, tmp1);

		/* tmp_x = tmp_t                 */
		mpf_set(tmp_x, tmp_t);

		smpfmij(hmat, i2 - 1, i, tmp_x);
		smpfmij(hmat, i2    , i, tmp_y);
	}

	/* rotapp(c, s, H[1:i2, i2-1]  , H[1:i2, i2]  ) */
	for(i = 0; i <= i2; i++)
	{
		mpf_set(tmp_x, gmpfmij(hmat, i, i2 - 1));
		mpf_set(tmp_y, gmpfmij(hmat, i, i2    ));

		/* tmp_t = c * tmp_x + s * tmp_y */
		mpf_mul(tmp , c, tmp_x);
		mpf_mul(tmp1, s, tmp_y);
		mpf_add(tmp_t, tmp, tmp1);

		/* tmp_y = c * tmp_y - s * tmp_x */
		mpf_mul(tmp , c, tmp_y);
		mpf_mul(tmp1, s, tmp_x);
		mpf_sub(tmp_y, tmp, tmp1);

		/* tmp_x = tmp_t                 */
		mpf_set(tmp_x, tmp_t);

		smpfmij(hmat, i, i2 - 1, tmp_x);
		smpfmij(hmat, i, i2    , tmp_y);
	}

	if(flag_get_proj_mat == 1)
	{
		/* rotapp(c, s, Q[1:n, i2-1]   , Q[1:n, i2]   ) */
		for(i = 0; i < dim; i++)
		{
			mpf_set(tmp_x, gmpfmij(qmat, i, i2 - 1));
			mpf_set(tmp_y, gmpfmij(qmat, i, i2    ));

			/* tmp_t = c * tmp_x + s * tmp_y */
			mpf_mul(tmp , c, tmp_x);
			mpf_mul(tmp1, s, tmp_y);
			mpf_add(tmp_t, tmp, tmp1);

			/* tmp_y = c * tmp_y - s * tmp_x */
			mpf_mul(tmp , c, tmp_y);
			mpf_mul(tmp1, s, tmp_x);
			mpf_sub(tmp_y, tmp, tmp1);

			/* tmp_x = tmp_t                 */
			mpf_set(tmp_x, tmp_t);

			smpfmij(qmat, i, i2 - 1, tmp_x);
			smpfmij(qmat, i, i2    , tmp_y);
		}
	}

	/* clear */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(c);
	mpf_clear(s);
	mpf_clear(tmp_t);
	mpf_clear(tmp_x);
	mpf_clear(tmp_y);

	free_mpfvector(v);
	free_mpfvector(tmp_u);

	return SUCCESS;
}

/* mpf_t QR Method for Real Hessenberg Matrix */
/* hmat := [ x x x ... x x x ] */
/*         [ x x x ... x x x ] */
/*         [ 0 x x ... x x x ] */
/*         [ ............... ] */
/*         [ 0 0 0 ... x x x ] */
/*         [ 0 0 0 ... 0 x x ] */
int mpfqrh_iteration(MPFMatrix hmat, MPFMatrix qmat, int flag_get_proj_mat, long int maxtimes, mpf_t reps)
{
	unsigned long prec;
	long int i1, i2, tmp_i1, old_i2, dim, times;
	mpf_t eps, tmp[4];
	MPFVector u;

	dim = hmat->row_dim;
	if(dim <= 0 || (dim != hmat->col_dim))
		return -1;

	/* Initialize */
	prec = hmat->prec;
	mpf_init2(eps, prec);
	mpf_init2(tmp[0], prec);
	mpf_init2(tmp[1], prec);
	mpf_init2(tmp[2], prec);
	mpf_init2(tmp[3], prec);
	u = init2_mpfvector(3, prec);

	i1 = 0;
	i2 = dim - 1;

	if(qmat == NULL)
		flag_get_proj_mat = 0;

	if(flag_get_proj_mat == 1)
		setI_mpfmatrix(qmat);

	/* eps = norm1_mpfmatrix(hmat) * 1.0e-15 */
	norm1_mpfmatrix(eps, hmat);
	mpf_set_ui(eps, 1UL);
	mpf_mul(eps, eps, reps);
//	mpf_sqrt(eps, reps);

	/*for(i2 = dim - 1; i2 > 1; )
	{
	*/
	for(times = 0; times < maxtimes; times++)
	{
		old_i2 = i2;
		mpfbacksearch2(&i1, &i2, i2, hmat, eps);

		if(i2 == 0)
			break;

		printf("(i1, i2) = (%ld, %ld)\n", i1, i2);

		start_mpfqrh(
			tmp[0],
			u,
			gmpfmij(hmat, i1    , i1    ),
			gmpfmij(hmat, i1    , i1 + 1),
			gmpfmij(hmat, i1 + 1, i1    ),
			gmpfmij(hmat, i1 + 1, i1 + 1),
			gmpfmij(hmat, i1 + 2, i1 + 1),
			gmpfmij(hmat, i2 - 1, i2 - 1),
			gmpfmij(hmat, i2 - 1, i2    ),
			gmpfmij(hmat, i2    , i2 - 1),
			gmpfmij(hmat, i2    , i2    )
		);
	
		mpfqrh(hmat, qmat, i1, i2, u, tmp[0], flag_get_proj_mat);

		/* deflation */
/*		mpf_abs(tmp[0], gmpfmij(hmat, i2    , i2 - 1));
		mpf_abs(tmp[1], gmpfmij(hmat, i2 - 1, i2 - 2));
		mpf_abs(tmp[2], gmpfmij(hmat, i2    , i2    ));
		mpf_abs(tmp[3], gmpfmij(hmat, i2 - 1, i2 - 1));
		mpf_mul(tmp[2], tmp[2], tmp[3]);
		mpf_sqrt(tmp[2], tmp[2]);
		mpf_mul(tmp[2], tmp[2], reps);
		mpf_add(tmp[2], tmp[2], reps);
		//if(mpf_cmp(tmp[0], tmp[2]) <= 0)
		if(mpf_cmp(tmp[0], tmp[2]) <= 0)
		{
			i2--;
			break;
		}
		else if(mpf_cmp(tmp[1], tmp[2]) <= 0)
		{
			i2 -= 2;
			break;
		}
		if(times >= maxtimes)
		{
			printf("WARNING: (i1, i2) = (%d, %d) %d times\n", i1, i2, times);
			i2--;
		}
*/	}

	/* clear */
	mpf_clear(eps);
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	mpf_clear(tmp[3]);
	free_mpfvector(u);

	return times;
}

/* get complex eigenvalues from converged hessenberg matrix via double QR */
void get_ceig_mpfqrh(CMPFVector eig, MPFMatrix hmat, mpf_t reps)
{
	unsigned long prec;
	long int i, dim;
	mpf_t hi1i, hii, hi1i1, tmp[3], coef[3], tmp_real[2], tmp_imag[2];

	dim = hmat->row_dim;
	//if(dim > eig->size)
	//	dim = eig->size;
	if(dim > eig->dim)
		dim = eig->dim;

	/* Initialize */
	prec = eig->prec;
	for(i = 0; i < 3; i++)
	{
		mpf_init2(tmp[i], prec);
		mpf_init2(coef[i], prec);
	}
	mpf_init2(tmp_real[0], prec); mpf_init2(tmp_real[1], prec);
	mpf_init2(tmp_imag[0], prec); mpf_init2(tmp_imag[1], prec);
	mpf_init2(hi1i,  prec);
	mpf_init2(hii ,  prec);
	mpf_init2(hi1i1, prec);

	/* search larger low subdiagonal elements */
	for(i = 0; i < dim - 1; )
	{
		mpf_set(hii  , gmpfmij(hmat, i    , i    ));
		mpf_set(hi1i , gmpfmij(hmat, i + 1, i    ));
		mpf_set(hi1i1, gmpfmij(hmat, i + 1, i + 1));

		mpf_abs(tmp[0], hii);
		mpf_abs(tmp[1], hi1i1);
		mpf_add(tmp[1], tmp[0], tmp[1]);
		//mpf_mul(tmp[1], tmp[0], tmp[1]);
		//mpf_sqrt(tmp[1], tmp[1]);
		mpf_mul(tmp[1], tmp[1], reps);
		//mpf_add(tmp[1], tmp[1], reps);

		mpf_abs(tmp[2], hi1i);

		/* complex eigenvalues (pair) */
		/* if(fabs(h[i+1, i]) > sqrt(fabs(h[i, i] * h[i+1, i+1])) * reps + aeps) */
		if(mpf_cmp(tmp[2], tmp[1]) >= 0)
		{
			/* 1 * x^2 - (h[i,i] + h[i+1,i+1]) * x + (h[i,i]*h[i+1,i+1] - h[i,i+1] * h[i+1,i]) = 0 */
			mpf_set_ui(coef[2], 1UL);

			mpf_add(coef[1], hii, hi1i1);
			mpf_neg(coef[1], coef[1]);

			mpf_mul(tmp[0], hii, hi1i1);
			mpf_mul(tmp[1], gmpfmij(hmat, i, i+1), hi1i);
			mpf_sub(coef[0], tmp[0], tmp[1]);

			mpfquadratic_eq(tmp_real, tmp_imag, coef);

			set_cmpfvector_i_re(eig, i    , tmp_real[0]);
			set_cmpfvector_i_im(eig, i    , tmp_imag[0]);
			set_cmpfvector_i_re(eig, i + 1, tmp_real[1]);
			set_cmpfvector_i_im(eig, i + 1, tmp_imag[1]);

			printf("%ld, %ld -> complex\n", i, i+1);
			i += 2;
		}
		/* real eigenvalue */
		else
		{
			mpf_set(tmp_real[0], hii);
			mpf_set_ui(tmp_imag[0], 0UL);

			set_cmpfvector_i_re(eig, i, tmp_real[0]);
			set_cmpfvector_i_im(eig, i, tmp_imag[0]);

			printf("%ld    -> real\n", i);

			i++;
		}

		/* i == dim - 1 (final) */
		/* real eigenvalue */
		if(i == dim - 1)
		{
			mpf_set(hii  , gmpfmij(hmat, dim - 1, dim - 1));

			mpf_set(tmp_real[0], hii);
			mpf_set_ui(tmp_imag[0], 0UL);

			set_cmpfvector_i_re(eig, dim - 1, tmp_real[0]);
			set_cmpfvector_i_im(eig, dim - 1, tmp_imag[0]);

			printf("%ld    -> real\n", i);

			break;
		}

	}

	/* clear */
	for(i = 0; i < 3; i++)
	{
		mpf_clear(tmp[i]);
		mpf_clear(coef[i]);
	}
	mpf_clear(tmp_real[0]); mpf_clear(tmp_real[1]);
	mpf_clear(tmp_imag[0]); mpf_clear(tmp_imag[1]);
	mpf_clear(hi1i);
	mpf_clear(hii);
	mpf_clear(hi1i1);
}


#endif
