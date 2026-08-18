/*******************************************************/ 
/* cpi-gmp.c: original version is "cpi.c"              */
/* modified by Tomonori Kouya                          */
/* Version 0.0: 2003.04/01                             */
/*                                                     */
/* Usage:                                              */
/* % mpicc cpi-gmp.c mpi_gmp.c -lmpfr -lgmp            */
/*   or                                                */
/* % mpicc -DUSE_MPFR cpi-gmp.c mpi_gmp.c -lmpfr -lgmp */
/*******************************************************/
#include "mpi.h"
#include <stdio.h>
#include <math.h>

#ifdef __GMP_H
#include "mpi_gmp.h"

// f(a) = 4/(1 + a^2)
void mpf_f(mpf_t ret, mpf_t a)
{
	mpf_t tmp;

	mpf_init(tmp);

	mpf_mul(tmp, a, a);
	mpf_add_ui(tmp, tmp, 1UL);
	mpf_ui_div(ret, 4UL, tmp);

	mpf_clear(tmp);

	return;
}
#endif

int main(int argc, char *argv[])
{
	int done = 0, n, myid, numprocs, i;
	double startwtime, endwtime;
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

#ifdef __GMP_H
	mpf_t mpf_mypi, mpf_pi, mpf_h, mpf_sum, mpf_x, mpf_fret, mpf_tmp;
	void * packed_mpf_mypi, * packed_mpf_pi;
	int mpf_size, pos;
#endif

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(processor_name, &namelen);

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);

#ifdef __GMP_H
/* MPF or MPFR */
#define MPF_PREC 256

	// typedef for MPI_COMM_WORLD
	mpf_set_default_prec(MPF_PREC);
	commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	if(myid == 0)
	{
		printf("----- Start (MPF or MPFR) -----\n");
		n = 16384;
		startwtime = MPI_Wtime();
	}

	/* is needed ? */
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	mpf_init(mpf_h);
	mpf_init(mpf_x);
	mpf_init(mpf_pi);
	mpf_init(mpf_sum);
	mpf_init(mpf_mypi);
	mpf_init(mpf_tmp);
	mpf_init(mpf_fret);

	mpf_set_ui(mpf_h, n);
	mpf_ui_div(mpf_h, 1UL, mpf_h);

	mpf_set_ui(mpf_sum, 0UL);

	// mpf_tmp = 0.5
	mpf_set_ui(mpf_tmp, 1UL);
	mpf_div_ui(mpf_tmp, mpf_tmp, 2UL);

	for (i = myid + 1; i <= n; i += numprocs)
	{
		mpf_set_ui(mpf_x, (unsigned long)i);
		mpf_sub(mpf_x, mpf_x, mpf_tmp);
		mpf_mul(mpf_x, mpf_x, mpf_h);

		mpf_f(mpf_fret, mpf_x);
		mpf_add(mpf_sum, mpf_sum, mpf_fret);
	}

	mpf_mul(mpf_mypi, mpf_sum, mpf_h);

	packed_mpf_mypi = allocbuf_mpf(mpf_get_prec(mpf_mypi), 1);
	packed_mpf_pi = allocbuf_mpf(mpf_get_prec(mpf_pi), 1);
	pack_mpf(mpf_mypi, 1, packed_mpf_mypi);
	pack_mpf(mpf_pi, 1, packed_mpf_pi);
	MPI_Reduce(packed_mpf_mypi, packed_mpf_pi, 1, MPI_MPF, MPI_MPF_SUM, 0, MPI_COMM_WORLD);
	unpack_mpf(packed_mpf_pi, mpf_pi, 1);

	if (myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("mpf_pi :"); mpf_out_str(stdout, 10, 0, mpf_pi); printf("\n");
		printf("mpf_h :"); mpf_out_str(stdout, 10, 0, mpf_h); printf("\n");
		printf("wall clock time = %f\n", endwtime - startwtime);		
		printf("----- End (MPF or MPFR) -----\n");
	}

	// free
	mpf_clear(mpf_mypi);
	mpf_clear(mpf_sum);
	mpf_clear(mpf_x);
	mpf_clear(mpf_h);
	mpf_clear(mpf_pi);
	mpf_clear(mpf_fret);
	mpf_clear(mpf_tmp);

	// free typedef
	free_mpf(&(MPI_MPF));
	free_mpf_op(&(MPI_MPF_SUM));
#endif
	MPI_Finalize();

}
