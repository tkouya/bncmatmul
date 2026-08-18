/* Test Program for reading and writing MatrixMarket-formatted files including complex elements */
/* 2011-12-06 (Tue) Tomonori Kouya */

#include "bnc.h"
#include "bncmm.h"
#include "bncmm_c.h"

int main()
{
	long int i;
	MPFMatrix a;
	CMPFMatrix mat_tmp, mat_tmp2, p, q;
	CMPFVector lambda;
	mpf_t norm;
	MPFCmplx ctmp;

	set_bnc_default_prec_decimal(500);

	mpf_init(norm);

// IRK G10
	a = init_mpfmatrix_readMMcoordinate("irkg10_d500_a.mtx");
	p = init_cmpfmatrix_readMMcoordinate("irkg10_d500_p.mtx");
	q = init_cmpfmatrix_readMMcoordinate("irkg10_d500_q.mtx");

	mat_tmp = init_cmpfmatrix(a->row_dim, a->col_dim);
	mat_tmp2 = init_cmpfmatrix(a->row_dim, a->col_dim);
	ctmp = init_mpfcmplx();

	lambda = init_cmpfvector_readMMcoordinate("irkg10_d500_lambda.mtx");
//	print_cmpfvector(lambda);
//	return 0;

	/* p * q */
	mul_cmpfmatrix(mat_tmp, p, q);
	//print_cmpfmatrix(mat_tmp);
	normi_cmpfmatrix(norm, mat_tmp);
	printf("||p * q||_inf = "); mpf_out_str(stdout, 10, 0, norm); printf("\n");

	/* q * a * p */
	subst_cmpfmatrix_mpfmat(mat_tmp, a);
	mul_cmpfmatrix(mat_tmp2, q, mat_tmp);
	mul_cmpfmatrix(mat_tmp, mat_tmp2, p);
//	print_cmpfmatrix(mat_tmp);
	for(i = 0; i < lambda->dim; i++)
	{
		sub_mpfcmplx(ctmp, get_cmpfmatrix_ij(mat_tmp, i, i), get_cmpfvector_i(lambda, i));
		abs_mpfcmplx(norm, ctmp);
		printf("%5ld: ", i);
		mpf_out_str(stdout, 10, 10, norm);
//		print_mpfcmplx(get_cmpfmatrix_ij(mat_tmp, i, i));
//		print_mpfcmplx(get_cmpfvector_i(lambda, i));
		printf("\n");
	}

	/* free */
	free_cmpfmatrix(mat_tmp);
	free_cmpfmatrix(mat_tmp2);
	free_mpfmatrix(a);
	free_cmpfmatrix(p);
	free_cmpfmatrix(q);
	mpf_clear(norm);
	free_mpfcmplx(ctmp);

	return 0;
}
