/* test_hqr.c : test program for hessenberg.c */

#include <stdio.h>
#include <math.h>

#include "bnc.h"

#include "fread_write_opt.c"

int main(int argc, char *argv[])
{
	long int dqr_times, mpfqr_times;
	long int row_dim, col_dim;
	DMatrix dmat, dmat_org, dproj, dproj_t;
#ifdef USE_GMP
	unsigned long prec, dprec;
	mpf_t reps;
	MPFMatrix mpfmat, mpfmat_org, mpfproj, mpfproj_t;
	CMPFArray cmpfeig;
#endif

	if(argc <= 1)
	{
		printf("Usage: %s [fname of matrix data]\n", argv[0]);
		return 0;
	}

//	dmat = init_dmatrix(DIM, DIM);
	dmat = init_read_dmatrix(argv[1]);
	row_dim = dmat->row_dim;
	col_dim = dmat->col_dim;

	dmat_org = init_dmatrix(row_dim, col_dim);
	dproj = init_dmatrix(row_dim, col_dim);
	dproj_t = init_dmatrix(row_dim, col_dim);

	subst_dmatrix(dmat_org, dmat);

	printf("Orginal Matrix: \n");
	print_dmatrix(dmat_org);

//	dstrimat(dmat, dproj, 1);
	dhessenberg(dmat, dproj, 1);

	printf("\nHessenberg Matrix: \n");
	print_dmatrix(dmat);
/*
	printf("\nProjection Matrix: \n");
	print_dmatrix(dproj);

	printf("\nProjection Matrix: \n");
	transpose_dmatrix(dproj_t, dproj);
	print_dmatrix(dproj_t);

	printf("\nP^T * A * P: \n");
	mul_dmatrix(dmat, dmat_org, dproj);
	mul_dmatrix(dproj, dproj_t, dmat);
	print_dmatrix(dproj);
*/

	dqr_times = dqrh_iteration(dmat, dproj, 1, 5 * row_dim, 1.0e-8);

	printf("\nAfter QR iteration (%d times): \n", dqr_times);
	print2_dmatrix(dmat, 10);
	printdiag2_dmatrix(dmat, 15, 1);

	free_dmatrix(dmat);
	free_dmatrix(dmat_org);
	free_dmatrix(dproj);
	free_dmatrix(dproj_t);

//	goto end;

/***************************************************/
/* MPFR or GMP                                     */
/***************************************************/
#ifdef USE_GMP
	dprec = 100;

	set_bnc_default_prec_decimal(dprec);

	mpfmat = init_read_mpfmatrix(argv[1]);
	row_dim = mpfmat->row_dim;
	col_dim = mpfmat->col_dim;

	mpfproj = init_mpfmatrix(row_dim, col_dim);
	cmpfeig = init_cmpfarray(row_dim);

//	hilbert_mpfmatrix(mpfmat, DIM);
//	frank_mpfmatrix(mpfmat, DIM);
//	lotkin_mpfmatrix(mpfmat, DIM);
//	print2_mpfmatrix(mpfmat, 10);

	mpfhessenberg(mpfmat, mpfproj, 1);

	printf("\nHessenberg Matrix: \n");
	print2_mpfmatrix(mpfmat, 10);

	mpf_init_set_str(reps, "1.0e-100", 10);

	mpfqr_times = mpfqrh_iteration(mpfmat, mpfproj, 1, row_dim * col_dim, reps);
//	mpf_qr(mpfmat, DIM * DIM);

	printf("\nAfter QR iteration (%d times): \n", mpfqr_times);
	//print2_mpfmatrix(mpfmat, 10);
	printdiag2_mpfmatrix(mpfmat, 20, 2);

	get_ceig_mpfqrh(cmpfeig, mpfmat, reps);
	print2_cmpfarray(cmpfeig, 20);

	printlog10_mpfmatrix(mpfmat, 7);

	mpf_clear(reps);
	free_mpfmatrix(mpfmat);
	free_mpfmatrix(mpfproj);
	free_cmpfarray(cmpfeig);

#endif

end:
	return 0;
}
