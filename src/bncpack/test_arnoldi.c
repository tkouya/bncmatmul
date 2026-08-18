/********************************************************************************/
/* test arnoldi.c                                                               */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc.h"

#ifdef USE_SPARSE_MATRIX
#include "bncsparse.h"
#endif

/* set_dmatrix_i_dvec: set one vector to i-th column of matrix */
void set_dmatrix_i_dvec(DMatrix mat, long int col_index, DVector vec)
{
	long int col_dim, i;

	col_dim = mat->col_dim;
	if(mat->col_dim != vec->dim)
	{
		fprintf(stderr, "Warning!: matrix column dimension(%d) is not the same as vector dimension(%d)!\n", mat->col_dim, vec->dim);
		if(col_dim > vec->dim)
			col_dim = vec->dim;
	}
	if((col_index < 0) || (col_index >= mat->row_dim))
	{
		fprintf(stderr, "Error!: col_index(%d) exceed matrix row dimension(%d)!\n", col_index, mat->row_dim);
		return;
	}

	for(i = 0; i < col_dim; i++)
		set_dmatrix_ij(mat, i, col_index, get_dvector_i(vec, i));
}

/* set random vector */
void set_random_dvector(DVector vec, int seed)
{
	long int i;

	srand(seed);

	for(i = 0; i < vec->dim; i++)
		set_dvector_i(vec, i, (double)rand());
}

/* Arnoldi Method to transform Hessenberg Matrix */
/* Q^T * A * Q = H                           */
/* Q = q = {q_1, q_2, ..., q_k}              */
/* H = hess = [ h_11 h_12 ... h_1,k-1 h_1k ] */
/*            [ h_21 h_22 ... h_2,k-1 h_2k ] */
/*            [ 0    h_32 ... h_3,k-1 h_3k ] */
/*            [ ...  ...  ... .....   ...  ] */
/*            [ 0    0    ... h_k,k-1 h_kk ] */
#ifdef USE_SPARSE_MATRIX
int arnoldi_drsmatrix(DMatrix hess, DMatrix q, long int dim_k, DRSMatrix mat, DVector init_q)
#else
int arnoldi_dmatrix(DMatrix hess, DMatrix q, long int dim_k, DMatrix mat, DVector init_q)
#endif
{
	long int i, j, k, row_dim, col_dim;
	double htmp, tmp;
	DVector r, vtmp, aq;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	if(dim_k <= 0)
		return BNC_SUCCESS;

	if(dim_k > row_dim)
	{
		fprintf(stderr, "Warning!: dim_k(=%d) is larger than row_dim(=%d)!\n", dim_k, row_dim);
		dim_k = row_dim;
		fprintf(stderr, "Reset dim_k = %d\n", dim_k);
	}

	r = init_dvector(col_dim);
	vtmp = init_dvector(col_dim);
	aq = init_dvector(col_dim);

	/* r := init_q  */
	subst_dvector(r, init_q);

	htmp = norm2_dvector(r);
	for(i = 0; i < dim_k; i++)
	{
		if(htmp == 0.0)
		{
			fprintf(stderr, "Error!: h[%d][%d] is zero!\n", i + 1, i);
			free_dvector(r);
			free_dvector(vtmp);
			free_dvector(aq);

			return BNC_ERROR;
		}

		/* q_i := r_i / htmp */
		tmp = 1.0 / htmp;
		cmul_dvector(vtmp, tmp, r);
		set_dmatrix_i_dvec(q, i, vtmp);
		
		/* r := Aq_i */
#ifdef USE_SPARSE_MATRIX
		mul_drsmatrix_dvec(aq, mat, vtmp);
#else
		mul_dmatrix_dvec(aq, mat, vtmp);
#endif
		subst_dvector(r, aq);

		/* h_1i, h_2i, ..., h_ii */
		for(j = 0; j <= i; j++)
		{
			/* h_ji := (q_j, Aq_i) */
			htmp = 0.0;
			for(k = 0; k < col_dim; k++)
				htmp += get_dmatrix_ij(q, k, j) * get_dvector_i(aq, k);
			set_dmatrix_ij(hess, j, i, htmp);

			/* r := r - h_ji * q_j */
			for(k = 0; k < col_dim; k++)
			{
				tmp = get_dvector_i(r, k) - htmp * get_dmatrix_ij(q, k, j);
				set_dvector_i(r, k, tmp);
			}
		}

		if(i == (dim_k - 1))
			break;

		/* h_i+1,i := ||r||_2 */
		htmp = norm2_dvector(r);
		set_dmatrix_ij(hess, i + 1, i, htmp);
	}

	free_dvector(r);
	free_dvector(vtmp);
	free_dvector(aq);

	return BNC_SUCCESS;
}

/* Lanczos Method for Unsymmetric Real square matrix */
/* P^T * A * Q = T                                             */
/* P = p = {p_1, p_2, ..., p_k}                                */
/* Q = q = {q_1, q_2, ..., q_k}                                */
/* T = tridiag = [ t_11 t_12 0    0 ...                    0 ] */
/*               [ t_21 t_22 t_23 0 ...                    0 ] */
/*               [      ...  ...  ...                        ] */
/*               [ ...  ...  ... t_k-1,k-2 t_k-1,k-1 t_k-1,k ] */
/*               [ 0    0    ... 0         t_k,k-1   t_kk    ] */
#ifdef USE_SPARSE_MATRIX
int lanczos_drsmatrix(DBMatrix tridiag, DMatrix pmat, DMatrix qmat, long int dim_k, DRSMatrix mat, DVector init_p, DVector init_q)
#else
int lanczos_dmatrix(DBMatrix tridiag, DMatrix pmat, DMatrix qmat, long int dim_k, DMatrix mat, DVector init_p, DVector init_q)
#endif
{
	long int i, j, k, row_dim, col_dim;
	double alpha, beta, gamma, tmp;
	DVector tmp_p, tmp_q, p, q, p_old, q_old, aq, ap, r, s;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	if(dim_k <= 0)
		return BNC_SUCCESS;

	if(dim_k > row_dim)
	{
		fprintf(stderr, "Warning!: dim_k(=%d) is larger than row_dim(=%d)!\n", dim_k, row_dim);
		dim_k = row_dim;
		fprintf(stderr, "Reset dim_k = %d\n", dim_k);
	}

	p = init_dvector(col_dim);
	q = init_dvector(col_dim);
	p_old = init_dvector(col_dim);
	q_old = init_dvector(col_dim);
	tmp_p = init_dvector(col_dim);
	tmp_q = init_dvector(col_dim);
	ap = init_dvector(col_dim);
	aq = init_dvector(col_dim);
	r = init_dvector(col_dim);
	s = init_dvector(col_dim);

	tmp = 1.0 / norm2_dvector(init_q);
	cmul_dvector(r, tmp, init_q);
	tmp = 1.0 / norm2_dvector(init_p);
	cmul_dvector(s, tmp, init_p);

	for(i = 0; i < dim_k; i++)
	{
		beta = norm2_dvector(r);
		gamma = ip_dvector(s, r) / beta;

		if(((i - 1) >= 0) && (i <= (dim_k - 1)))
		{
			set_dbmatrix_ij(tridiag, i, i - 1, beta);
			set_dbmatrix_ij(tridiag, i - 1, i, gamma);
		}

		subst_dvector(q_old, q);
		subst_dvector(p_old, p);

		tmp = 1.0 / beta;
		cmul_dvector(q, tmp, r);
		tmp = 1.0 / gamma;
		cmul_dvector(p, tmp, s);

		set_dmatrix_i_dvec(qmat, i, q);
		set_dmatrix_i_dvec(pmat, i, p);

		/* aq := Aq_i */
#ifdef USE_SPARSE_MATRIX
		mul_drsmatrix_dvec(aq, mat, q);
#else
		mul_dmatrix_dvec(aq, mat, q);
#endif

		/* alpha := (p_i, Aq_i) */
		alpha = ip_dvector(p, aq);
		set_dbmatrix_ij(tridiag, i, i, alpha);

		/* r_i+1 := Aq_i - alpha_i q_i - gamma * q_i-1 */
		cmul_dvector(tmp_q, alpha, q);
		sub_dvector(aq, aq, tmp_q);
		cmul_dvector(tmp_q, gamma, q_old);
		sub_dvector(r, aq, tmp_q);

		/* ap := A^T p_i */
#ifdef USE_SPARSE_MATRIX
		mul_drsmatrixt_dvec(ap, mat, p);
#else
		mul_dmatrixt_dvec(ap, mat, p);
#endif

		/* s_i+1 := Ap_i - alpla_i * p_i - beta * p_i-1 */
		cmul_dvector(tmp_p, alpha, p);
		sub_dvector(ap, ap, tmp_p);
		cmul_dvector(tmp_p, beta, p_old);
		sub_dvector(s, ap, tmp_p);
	}

	free_dvector(p);
	free_dvector(q);
	free_dvector(p_old);
	free_dvector(q_old);
	free_dvector(tmp_p);
	free_dvector(tmp_q);
	free_dvector(ap);
	free_dvector(aq);
	free_dvector(r);
	free_dvector(s);
}

/* DKA for tridiagonal matrix */
/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
double ddka_center_dtridiag(DBMatrix mat)
{
	double ret;

/*
	ret = get_dpoly_i(func, func->deg - 1);
	ret /= get_dpoly_i(func, func->deg);
	ret /= func->deg;
	ret = -ret;
*/
	return ret;

}
/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
double ddka_radius_dtridiag(DPoly func)
{
	long int i;
	double ret, tmp, num_nonzero, an;

	num_nonzero = num_nonzero_dpoly(func);
	an = get_dpoly_i(func, func->deg);
	ret = num_nonzero;
	for(i = func->deg - 1; i >= 0; i--)
	{
		tmp = get_dpoly_i(func, i) / an;
		tmp *= num_nonzero;
		tmp = fabs(tmp);
		tmp = pow(tmp, 1.0/(func->deg - i));
		if(ret < tmp)
			ret = tmp;
	}

	return ret;
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void ddka_init_dtridiag(CDArray x_init, DPoly func)
{
	long int i, itmp;
	double rad, cen, an, tmp, re_cinit, im_cinit;
	DCmplx cinit;

	rad = ddka_radius(func);
	cen = ddka_center(func);

//	printf("%f, %f\n", rad, cen);

	cinit = init_dcmplx();
	for(i = 0; i < func->deg; i++)
	{
		set0_dcmplx(cinit);
		tmp = (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg));
		iexp_dcmplx(cinit, tmp);
		re_cinit = get_real_dcmplx(cinit);
		im_cinit = get_image_dcmplx(cinit);

		re_cinit = cen + rad * re_cinit;
		im_cinit = rad * im_cinit;

		set_real_dcmplx(cinit, re_cinit);
		set_image_dcmplx(cinit, im_cinit);

//		printf("%5d(%f) ", i, abs_dcmplx(cinit)); print_dcmplx(cinit);

		set_cdarray_i(x_init, i, cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int ddka_dtridiag(CDArray ans, CDArray x_init, DPoly func, long int maxtimes, double abs_eps, double rel_eps)
{
	long int times, i, j, deg, flag;
	double absmodval, abs_x, abs_newx;
	DCmplx modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	modval = init_dcmplx();
	low_modval = init_dcmplx();
	up_modval = init_dcmplx();
	tmp = init_dcmplx();
	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			set_real_dcmplx(low_modval, 1.0);
			set_image_dcmplx(low_modval, 0.0);
			for(j = 0; j < i; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, i),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, i),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			mul_dcmplx_d(low_modval, low_modval, get_dpoly_i(func, func->deg));
			ceval_dpoly(up_modval, func, get_cdarray_i(x_init, i));

			div_dcmplx(modval, up_modval, low_modval);
			sub_dcmplx(tmp, get_cdarray_i(x_init, i), modval);
			set_cdarray_i(ans, i, tmp);
//			printf("%25.17e, ", get_real_dcmplx(get_cfarray_i(ans, i)));

			/* check convergence */
			absmodval = abs_dcmplx(modval);
			abs_x = abs_dcmplx(get_cdarray_i(x_init, i));
			abs_newx = abs_dcmplx(get_cdarray_i(ans, i));
			if( absmodval > (abs_x + abs_newx) * rel_eps + abs_eps )
				flag = 1;

		}
//		printf("\n");

		/* check convergence */
		if(flag == 0)
			break;

		subst_cdarray(x_init, ans);
//		printf("%d: ", times); print_cdarray(x_init);

	}

	return times;
}

/* MPF */
#ifdef USE_GMP

/* set_mpfmatrix_i_dvec: set one vector to i-th column of matrix */
void set_mpfmatrix_i_mpfvec(MPFMatrix mat, long int col_index, MPFVector vec)
{
	long int col_dim, i;

	col_dim = mat->col_dim;
	if(mat->col_dim != vec->dim)
	{
		fprintf(stderr, "Warning!: matrix column dimension(%d) is not the same as vector dimension(%d)!\n", mat->col_dim, vec->dim);
		if(col_dim > vec->dim)
			col_dim = vec->dim;
	}
	if((col_index < 0) || (col_index >= mat->row_dim))
	{
		fprintf(stderr, "Error!: col_index(%d) exceed matrix row dimension(%d)!\n", col_index, mat->row_dim);
		return;
	}

	for(i = 0; i < col_dim; i++)
		set_mpfmatrix_ij(mat, i, col_index, get_mpfvector_i(vec, i));
}

/* set random vector */
void set_random_mpfvector(MPFVector vec, int seed)
{
	long int i;

	srand(seed);

	for(i = 0; i < vec->dim; i++)
		set_mpfvector_i_d(vec, i, (double)rand());
}

/* Arnoldi Method to transform Hessenberg Matrix */
/* Q^T * A * Q = H                           */
/* Q = q = {q_1, q_2, ..., q_k}              */
/* H = hess = [ h_11 h_12 ... h_1,k-1 h_1k ] */
/*            [ h_21 h_22 ... h_2,k-1 h_2k ] */
/*            [ 0    h_32 ... h_3,k-1 h_3k ] */
/*            [ ...  ...  ... .....   ...  ] */
/*            [ 0    0    ... h_k,k-1 h_kk ] */
#ifdef USE_SPARSE_MATRIX
int arnoldi_mpfrsmatrix(MPFMatrix hess, MPFMatrix q, long int dim_k, MPFRSMatrix mat, MPFVector init_q)
#else
int arnoldi_mpfmatrix(MPFMatrix hess, MPFMatrix q, long int dim_k, MPFMatrix mat, MPFVector init_q)
#endif
{
	unsigned long prec;
	long int i, j, k, row_dim, col_dim;
	mpf_t htmp, tmp;
	MPFVector r, vtmp, aq;

	prec = hess->prec;
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	if(dim_k <= 0)
		return BNC_SUCCESS;

	if(dim_k > row_dim)
	{
		fprintf(stderr, "Warning!: dim_k(=%d) is larger than row_dim(=%d)!\n", dim_k, row_dim);
		dim_k = row_dim;
		fprintf(stderr, "Reset dim_k = %d\n", dim_k);
	}

	mpf_init2(htmp, prec);
	mpf_init2(tmp, prec);

	r = init2_mpfvector(col_dim, prec);
	vtmp = init2_mpfvector(col_dim, prec);
	aq = init2_mpfvector(col_dim, prec);

	/* r := init_q  */
	subst_mpfvector(r, init_q);

	norm2_mpfvector(htmp, r);
	for(i = 0; i < dim_k; i++)
	{
		if(mpf_cmp_ui(htmp, 0UL) == 0)
		{
			fprintf(stderr, "Error!: h[%d][%d] is zero!\n", i + 1, i);

			mpf_clear(htmp);
			mpf_clear(tmp);
			free_mpfvector(r);
			free_mpfvector(vtmp);
			free_mpfvector(aq);

			return BNC_ERROR;
		}

		/* q_i := r_i / htmp */
		//tmp = 1.0 / htmp;
		mpf_ui_div(tmp, 1UL, htmp);
		cmul_mpfvector(vtmp, tmp, r);
		set_mpfmatrix_i_mpfvec(q, i, vtmp);
		
		/* r := Aq_i */
#ifdef USE_SPARSE_MATRIX
		mul_mpfrsmatrix_mpfvec(aq, mat, vtmp);
#else
		mul_mpfmatrix_mpfvec(aq, mat, vtmp);
#endif
		subst_mpfvector(r, aq);

		/* h_1i, h_2i, ..., h_ii */
		for(j = 0; j <= i; j++)
		{
			/* h_ji := (q_j, Aq_i) */
			mpf_set_ui(htmp, 0UL);
			for(k = 0; k < col_dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(q, k, j), get_mpfvector_i(aq, k));
				mpf_add(htmp, htmp, tmp);
			}
			set_mpfmatrix_ij(hess, j, i, htmp);

			/* r := r - h_ji * q_j */
			for(k = 0; k < col_dim; k++)
			{
				//tmp = get_mpfvector_i(r, k) - htmp * get_mpfmatrix_ij(q, k, j);
				mpf_mul(tmp, htmp, get_mpfmatrix_ij(q, k, j));
				mpf_sub(tmp, get_mpfvector_i(r, k), tmp);
				set_mpfvector_i(r, k, tmp);
			}
		}

		if(i == (dim_k - 1))
			break;

		/* h_i+1,i := ||r||_2 */
		norm2_mpfvector(htmp, r);
		set_mpfmatrix_ij(hess, i + 1, i, htmp);
	}

	mpf_clear(htmp);
	mpf_clear(tmp);

	free_mpfvector(r);
	free_mpfvector(vtmp);
	free_mpfvector(aq);

	return BNC_SUCCESS;
}

/* Lanczos Method for Unsymmetric Real square matrix */
/* P^T * A * Q = T                                             */
/* P = p = {p_1, p_2, ..., p_k}                                */
/* Q = q = {q_1, q_2, ..., q_k}                                */
/* T = tridiag = [ t_11 t_12 0    0 ...                    0 ] */
/*               [ t_21 t_22 t_23 0 ...                    0 ] */
/*               [      ...  ...  ...                        ] */
/*               [ ...  ...  ... t_k-1,k-2 t_k-1,k-1 t_k-1,k ] */
/*               [ 0    0    ... 0         t_k,k-1   t_kk    ] */
#ifdef USE_SPARSE_MATRIX
int lanczos_mpfrsmatrix(MPFBMatrix tridiag, MPFMatrix pmat, MPFMatrix qmat, long int dim_k, MPFRSMatrix mat, MPFVector init_p, MPFVector init_q)
#else
int lanczos_mpfmatrix(MPFBMatrix tridiag, MPFMatrix pmat, MPFMatrix qmat, long int dim_k, MPFMatrix mat, MPFVector init_p, MPFVector init_q)
#endif
{
	unsigned long prec;
	long int i, j, k, row_dim, col_dim;
	mpf_t alpha, beta, gamma, tmp;
	MPFVector tmp_p, tmp_q, p, q, p_old, q_old, aq, ap, r, s;

	prec = tridiag->prec;
	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	if(dim_k <= 0)
		return BNC_SUCCESS;

	if(dim_k > row_dim)
	{
		fprintf(stderr, "Warning!: dim_k(=%d) is larger than row_dim(=%d)!\n", dim_k, row_dim);
		dim_k = row_dim;
		fprintf(stderr, "Reset dim_k = %d\n", dim_k);
	}

	mpf_init2(alpha, prec);
	mpf_init2(beta, prec);
	mpf_init2(gamma, prec);
	mpf_init2(tmp, prec);

	p = init2_mpfvector(col_dim, prec);
	q = init2_mpfvector(col_dim, prec);
	p_old = init2_mpfvector(col_dim, prec);
	q_old = init2_mpfvector(col_dim, prec);
	tmp_p = init2_mpfvector(col_dim, prec);
	tmp_q = init2_mpfvector(col_dim, prec);
	ap = init2_mpfvector(col_dim, prec);
	aq = init2_mpfvector(col_dim, prec);
	r = init2_mpfvector(col_dim, prec);
	s = init2_mpfvector(col_dim, prec);

	//tmp = 1.0 / norm2_mpfvector(init_q);
	norm2_mpfvector(tmp, init_q);
	mpf_ui_div(tmp, 1UL, tmp);
	cmul_mpfvector(r, tmp, init_q);

	//tmp = 1.0 / norm2_mpfvector(init_p);
	norm2_mpfvector(tmp, init_p);
	mpf_ui_div(tmp, 1UL, tmp);
	cmul_mpfvector(s, tmp, init_p);

	for(i = 0; i < dim_k; i++)
	{
		norm2_mpfvector(beta, r);
		//gamma = ip_mpfvector(s, r) / beta;
		ip_mpfvector(gamma, s, r);
		mpf_div(gamma, gamma, beta);

		if(((i - 1) >= 0) && (i <= (dim_k - 1)))
		{
			set_mpfbmatrix_ij(tridiag, i, i - 1, beta);
			set_mpfbmatrix_ij(tridiag, i - 1, i, gamma);
		}

		subst_mpfvector(q_old, q);
		subst_mpfvector(p_old, p);

		//tmp = 1.0 / beta;
		mpf_ui_div(tmp, 1UL, beta);
		cmul_mpfvector(q, tmp, r);
		//tmp = 1.0 / gamma;
		mpf_ui_div(tmp, 1UL, gamma);
		cmul_mpfvector(p, tmp, s);

		set_mpfmatrix_i_mpfvec(qmat, i, q);
		set_mpfmatrix_i_mpfvec(pmat, i, p);

		/* aq := Aq_i */
#ifdef USE_SPARSE_MATRIX
		mul_mpfrsmatrix_mpfvec(aq, mat, q);
#else
		mul_mpfmatrix_mpfvec(aq, mat, q);
#endif

		/* alpha := (p_i, Aq_i) */
		ip_mpfvector(alpha, p, aq);
		set_mpfbmatrix_ij(tridiag, i, i, alpha);

		/* r_i+1 := Aq_i - alpha_i q_i - gamma * q_i-1 */
		cmul_mpfvector(tmp_q, alpha, q);
		sub_mpfvector(aq, aq, tmp_q);
		cmul_mpfvector(tmp_q, gamma, q_old);
		sub_mpfvector(r, aq, tmp_q);

		/* ap := A^T p_i */
#ifdef USE_SPARSE_MATRIX
		mul_mpfrsmatrixt_mpfvec(ap, mat, p);
#else
		mul_mpfmatrixt_mpfvec(ap, mat, p);
#endif

		/* s_i+1 := Ap_i - alpla_i * p_i - beta * p_i-1 */
		cmul_mpfvector(tmp_p, alpha, p);
		sub_mpfvector(ap, ap, tmp_p);
		cmul_mpfvector(tmp_p, beta, p_old);
		sub_mpfvector(s, ap, tmp_p);
	}

	mpf_clear(alpha);
	mpf_clear(beta);
	mpf_clear(gamma);
	mpf_clear(tmp);

	free_mpfvector(p);
	free_mpfvector(q);
	free_mpfvector(p_old);
	free_mpfvector(q_old);
	free_mpfvector(tmp_p);
	free_mpfvector(tmp_q);
	free_mpfvector(ap);
	free_mpfvector(aq);
	free_mpfvector(r);
	free_mpfvector(s);
}
#endif

//#define DIM 3
#define DIM 5
//#define DIM 10 

int main()
{
	DMatrix dmat, dmat_tmp, dmat_tmp1, dpmat, dqmat, dhmat;
	DVector dinit_p, dinit_q;
	DBMatrix dtrimat;
#ifdef USE_GMP
	mpf_t tmp;
	MPFMatrix mpfmat, mpfmat_tmp, mpfmat_tmp1, mpfpmat, mpfqmat, mpfhmat;
	MPFVector mpfinit_p, mpfinit_q;
	MPFBMatrix mpftrimat;
#endif

/* double */

	dmat = init_dmatrix(DIM, DIM);
	dmat_tmp = init_dmatrix(DIM, DIM);
	dmat_tmp1 = init_dmatrix(DIM, DIM);
	dpmat = init_dmatrix(DIM, DIM);
	dqmat = init_dmatrix(DIM, DIM);
	dhmat = init_dmatrix(DIM, DIM);
	dinit_p = init_dvector(DIM);
	dinit_q = init_dvector(DIM);
	dtrimat = init_dbmatrix(DIM, 1, 1);

/***************************************/
/* Arnoldi                             */
/***************************************/
	//hilbert_dmatrix(dmat, DIM);
	//frank_dmatrix(dmat, DIM);
	lotkin_dmatrix(dmat, DIM);
	set_random_dvector(dinit_q, DIM);

	printf("A: \n");
	print_dmatrix(dmat);

	arnoldi_dmatrix(dhmat, dqmat, DIM, dmat, dinit_q);

	printf("H: \n");
	print_dmatrix(dhmat);
	printf("Q: \n");
	print_dmatrix(dqmat);

	/* Q^T * Q */
	printf("Q^T * Q:\n");
	transpose_dmatrix(dmat_tmp, dqmat);
	mul_dmatrix(dmat_tmp1, dmat_tmp, dqmat);
	print_dmatrix(dmat_tmp1);
	printf("||Q^T * Q||_F = %25.17e\n", normf_dmatrix(dmat_tmp1));

	/* Q^T * mat * Q */
	printf("Q^T * A * Q:\n");
	transpose_dmatrix(dmat_tmp, dqmat);
	mul_dmatrix(dmat_tmp1, dmat_tmp, dmat);
	mul_dmatrix(dmat_tmp, dmat_tmp1, dqmat);
	print_dmatrix(dmat_tmp);

	/* H - Q^T * mat * Q */
	sub_dmatrix(dmat_tmp1, dhmat, dmat_tmp);
	printf("||H - Q^T * A * Q||_F = %25.17e\n", normf_dmatrix(dmat_tmp1));

/***************************************/
/* Lanczos                             */
/***************************************/
	//hilbert_dmatrix(dmat, DIM);
	//frank_dmatrix(dmat, DIM);
	lotkin_dmatrix(dmat, DIM);
	set_random_dvector(dinit_q, DIM);
	set_random_dvector(dinit_p, DIM);

	printf("A: \n");
	print_dmatrix(dmat);

	lanczos_dmatrix(dtrimat, dpmat, dqmat, DIM, dmat, dinit_p, dinit_q);

	printf("H: \n");
	print_dbmatrix(dtrimat);
	printf("Q: \n");
	print_dmatrix(dqmat);

	/* P^T * Q */
	printf("P^T * Q:\n");
	transpose_dmatrix(dmat_tmp1, dpmat);
	mul_dmatrix(dmat_tmp, dmat_tmp1, dqmat);
	print_dmatrix(dmat_tmp);
	printf("||P^T * Q||_F = %25.17e\n", normf_dmatrix(dmat_tmp));

	/* P * mat * Q */
	printf("P^T * A * Q:\n");
	transpose_dmatrix(dmat_tmp, dpmat);
	mul_dmatrix(dmat_tmp1, dmat_tmp, dmat);
	mul_dmatrix(dmat_tmp, dmat_tmp1, dqmat);
	print_dmatrix(dmat_tmp);

	free_dmatrix(dmat_tmp);
	free_dmatrix(dmat_tmp1);
	free_dmatrix(dmat);
	free_dmatrix(dpmat);
	free_dmatrix(dqmat);
	free_dmatrix(dhmat);
	free_dvector(dinit_p);
	free_dvector(dinit_q);
	free_dbmatrix(dtrimat);

#ifdef USE_GMP

	set_bnc_default_prec(128);

	mpf_init(tmp);

	mpfmat = init_mpfmatrix(DIM, DIM);
	mpfmat_tmp = init_mpfmatrix(DIM, DIM);
	mpfmat_tmp1 = init_mpfmatrix(DIM, DIM);
	mpfpmat = init_mpfmatrix(DIM, DIM);
	mpfqmat = init_mpfmatrix(DIM, DIM);
	mpfhmat = init_mpfmatrix(DIM, DIM);
	mpfinit_p = init_mpfvector(DIM);
	mpfinit_q = init_mpfvector(DIM);
	mpftrimat = init_mpfbmatrix(DIM, 1, 1);

/***************************************/
/* Arnoldi                             */
/***************************************/
	//hilbert_mpfmatrix(mpfmat, DIM);
	//frank_mpfmatrix(mpfmat, DIM);
	lotkin_mpfmatrix(mpfmat, DIM);
	set_random_mpfvector(mpfinit_q, DIM);

	printf("A: \n");
	print_mpfmatrix(mpfmat);

	arnoldi_mpfmatrix(mpfhmat, mpfqmat, DIM, mpfmat, mpfinit_q);

	printf("H: \n");
	print_mpfmatrix(mpfhmat);
	printf("Q: \n");
	print_mpfmatrix(mpfqmat);

	/* Q^T * Q */
	printf("Q^T * Q:\n");
	transpose_mpfmatrix(mpfmat_tmp, mpfqmat);
	mul_mpfmatrix(mpfmat_tmp1, mpfmat_tmp, mpfqmat);
	print_mpfmatrix(mpfmat_tmp1);
	printf("||Q^T * Q||_F = "); normf_mpfmatrix(tmp, mpfmat_tmp1); mpf_out_str(stdout, 10, 0, tmp); printf("\n");

	/* Q^T * mat * Q */
	printf("Q^T * A * Q:\n");
	transpose_mpfmatrix(mpfmat_tmp, mpfqmat);
	mul_mpfmatrix(mpfmat_tmp1, mpfmat_tmp, mpfmat);
	mul_mpfmatrix(mpfmat_tmp, mpfmat_tmp1, mpfqmat);
	print_mpfmatrix(mpfmat_tmp);

	/* H - Q^T * mat * Q */
	sub_mpfmatrix(mpfmat_tmp1, mpfhmat, mpfmat_tmp);
	printf("||H - Q^T * A * Q||_F = ");normf_mpfmatrix(tmp, mpfmat_tmp1); mpf_out_str(stdout, 10, 0, tmp); printf("\n");

/***************************************/
/* Lanczos                             */
/***************************************/
	//hilbert_mpfmatrix(mpfmat, DIM);
	//frank_mpfmatrix(mpfmat, DIM);
	lotkin_mpfmatrix(mpfmat, DIM);
	set_random_mpfvector(mpfinit_q, DIM);
	set_random_mpfvector(mpfinit_p, DIM);

	printf("A: \n");
	print_mpfmatrix(mpfmat);

	lanczos_mpfmatrix(mpftrimat, mpfpmat, mpfqmat, DIM, mpfmat, mpfinit_p, mpfinit_q);

	printf("H: \n");
	print_mpfbmatrix(mpftrimat);
	printf("Q: \n");
	print_mpfmatrix(mpfqmat);

	/* P^T * Q */
	printf("P^T * Q:\n");
	transpose_mpfmatrix(mpfmat_tmp1, mpfpmat);
	mul_mpfmatrix(mpfmat_tmp, mpfmat_tmp1, mpfqmat);
	print_mpfmatrix(mpfmat_tmp);
	printf("||P^T * Q||_F = "); normf_mpfmatrix(tmp, mpfmat_tmp); mpf_out_str(stdout, 10, 0, tmp); printf("\n");

	/* P * mat * Q */
	printf("P^T * A * Q:\n");
	transpose_mpfmatrix(mpfmat_tmp, mpfpmat);
	mul_mpfmatrix(mpfmat_tmp1, mpfmat_tmp, mpfmat);
	mul_mpfmatrix(mpfmat_tmp, mpfmat_tmp1, mpfqmat);
	print_mpfmatrix(mpfmat_tmp);

	mpf_clear(tmp);

	free_mpfmatrix(mpfmat_tmp);
	free_mpfmatrix(mpfmat_tmp1);
	free_mpfmatrix(mpfmat);
	free_mpfmatrix(mpfpmat);
	free_mpfmatrix(mpfqmat);
	free_mpfmatrix(mpfhmat);
	free_mpfvector(mpfinit_p);
	free_mpfvector(mpfinit_q);
	free_mpfbmatrix(mpftrimat);
#endif
}
