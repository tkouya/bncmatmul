#include "gddlinear.h"

using namespace std;

int main(int argc, char *argv[])
{
	int num_threads, num_blocks;
	long int i, j, dim;
	unsigned long prec = 128;
#ifdef __cplusplus
	dd_real ddtmp[2];
#else // __cplusplus
	double ddtmp[2][DDSIZE];
#endif // __cplusplus
	DDMatrix mpfc, mpfa, mpfb;
	DDVector mpfdiag_left, mpfdiag_right;
	double stime, etime[4], reldiff[4];

//	dim = 128;
	if(argc <= 1)
	{
		fprintf(stderr, "Usage: %s [dim] [#thread] [#blocks]\n", argv[0]);
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

	num_threads = 64;  // #blocks per threads
	if(argc >= 3)
	{
		num_threads = (unsigned long)atol(argv[2]);
		if(num_threads < 1)
			num_threads = 1;
	}

	num_blocks = 4;  // #blocks per grid
	if(argc >= 4)
	{
		num_blocks = (unsigned long)atol(argv[3]);
		if(num_blocks < 1)
			num_blocks = 1;
	}

/* double-double precision */
//	set_bncomp_num_threads(4);
//	set_bncomp_num_threads(num_threads);
//	printf("OpenMP #Threads = %ld\n", omp_get_num_threads());
	// GDD start!
	GDDStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
//	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
//	num_threads = num_threads; // #threads per block

	set0_dd(ddtmp[0]);
	set0_dd(ddtmp[1]);

	mpfc = init_ddmatrix(dim, dim);
	mpfa = init_ddmatrix(dim, dim);
	mpfb = init_ddmatrix(dim, dim);

	mpfdiag_left = init_ddvector(dim);
	mpfdiag_right = init_ddvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			rdd_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[0], ddtmp[0]);

			rdd_set_d(ddtmp[1], (double)rand());
			rdd_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[1], ddtmp[1]);

			set_ddmatrix_ij(mpfa, i, j, ddtmp[0]);
			set_ddmatrix_ij(mpfb, i, j, ddtmp[1]);
		}
	}
	gddmat_a = init_gddmatrix_dev(dim, dim);
	gddmat_b = init_gddmatrix_dev(dim, dim);
	gddmat_c = init_gddmatrix_dev(dim, dim);

	// normal matrix mul
	printf("_bncomp_mul_ddmatrix_simple...\n");
	stime = get_real_secv();
	_bncomp_mul_ddmatrix(mpfc_normal, mpfa, mpfb);
	etime[0] = get_real_secv() - stime;

	//left_scaling_ddmatrix(mpfa, mpfa, mpfdiag_left, NULL);
	//right_scaling_ddmatrix(mpfb, mpfb, mpfdiag_right, NULL);

	// blocked matrix mul
	printf("_bncomp_mul_ddmatrix_block...\n");
	//stime = get_secv();
	stime = get_real_secv();
	_bncomp_mul_ddmatrix_block(mpfc_block, mpfa, mpfb, 32);
	//etime[1] = get_secv() - stime;
	etime[1] = get_real_secv() - stime;

	// Strassen 
	stime = get_real_secv();
	mul_ddmatrix_strassen(mpfc, mpfa, mpfb, 8);
//	mul_ddmatrix_strassen(mpfc, mpfa, mpfb, 16);
//	mul_ddmatrix_strassen(mpfc, mpfa, mpfb, 32);
	etime[2] = get_real_secv() - stime;

	//mul_ddmatrix_dddiag(mpfc, mpfdiag_left, 0, mpfc, mpfdiag_right, 0);

	// difference
	sub_ddmatrix(mpfa, mpfc_normal, mpfc);

	// print
	printf("dim        : %ld\n", dim);
	printf("normal     : %f\n", etime[0]);
	printf("block      : %f\n", etime[1]);
	printf("strassen   : %f\n", etime[2]);
#ifdef __cplusplus
	normi_ddmatrix(&ddtmp[0], mpfa);
	normi_ddmatrix(&ddtmp[1], mpfc_normal);
#else // __cplusplus
	normi_ddmatrix(ddtmp[0], mpfa);
	normi_ddmatrix(ddtmp[1], mpfc_normal);
#endif // __cplusplus
	rdd_div(ddtmp[0], ddtmp[0], ddtmp[1]);
	printf("||reldiff||: "); rdd_out_str(ddtmp[0]); printf("\n");

/* Inverse */

	frank_ddmatrix(mpfa, dim);
	frank_ddmatrix(mpfb, dim);
//	lotkin_ddmatrix(mpfa, dim);
//	lotkin_ddmatrix(mpfb, dim);

	free_ddmatrix(mpfc);
	free_ddmatrix(mpfc_normal);
	free_ddmatrix(mpfc_block);
	free_ddmatrix(mpfa);
	free_ddmatrix(mpfb);

	// GQD end!
	GQDEnd();

	return 0;
}