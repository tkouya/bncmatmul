/* test_gddmatmul.cu : verify GPU gdd matmul (mul_gddmatrix_dev) vs CPU dd matmul */
#include <cstdio>
#include <cmath>
#include "gddlinear.h"

int main(void)
{
	long int i, j, dim = 63;
	int blocks = 64, threads = 128;

	fpu_fix_start(NULL);

	/* CPU dd matrices */
	DDMatrix a = init_ddmatrix(dim, dim), b = init_ddmatrix(dim, dim);
	DDMatrix c_cpu = init_ddmatrix(dim, dim), c_gpu = init_ddmatrix(dim, dim);
	double t[DDSIZE], rel[3][DDSIZE];

	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			rdd_set_ui(t, (unsigned long)((i + j) % 7 + 1)); rdd_sqrt(t, t);
			set_ddmatrix_ij(a, i, j, t);
			rdd_set_ui(t, (unsigned long)((i * 2 + j) % 5 + 1)); rdd_sqrt(t, t);
			set_ddmatrix_ij(b, i, j, t);
		}

	/* CPU reference */
	mul_ddmatrix(c_cpu, a, b);

	/* GPU gdd */
	GDDMatrix ga = init_gddmatrix_dev(dim, dim);
	GDDMatrix gb = init_gddmatrix_dev(dim, dim);
	GDDMatrix gc = init_gddmatrix_dev(dim, dim);
	subst_gddmatrix_dev_ddmat(ga, a);
	subst_gddmatrix_dev_ddmat(gb, b);
	mul_gddmatrix_dev(gc, ga, gb, blocks, threads);
	cudaDeviceSynchronize();
	subst_ddmatrix_gddmat_dev(c_gpu, gc);   /* device -> host */

	(void)rel;
	double maxrel = 0.0, d[DDSIZE], q[DDSIZE], cc[DDSIZE];
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			double *g = get_ddmatrix_ij(c_gpu, i, j);
			double *c = get_ddmatrix_ij(c_cpu, i, j);
			rdd_sub(d, g, c);        /* g - c */
			rdd_abs(d, d);
			rdd_abs(cc, c);
			if(cc[0] != 0.0) { rdd_div(q, d, cc); if(q[0] > maxrel) maxrel = q[0]; }
		}
	printf("GPU gdd matmul vs CPU dd (dim=%ld): max relative error = %10.3e\n", dim, maxrel);
	printf("%s\n", (maxrel < 1e-28) ? "PASS (full double-double precision)" : "FAIL");

	free_ddmatrix(a); free_ddmatrix(b); free_ddmatrix(c_cpu); free_ddmatrix(c_gpu);
	free_gddmatrix_dev(ga); free_gddmatrix_dev(gb); free_gddmatrix_dev(gc);
	return 0;
}
