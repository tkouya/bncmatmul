/************************************************/
/* Test program for ex_*.c                      */
/* Copyleft 2004, Tomonori Kouya                */
/************************************************/
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

#define DIM 5 // DIM == PROCS
#define TOTAL_DIM (2 * DIM * DIM)

/* for x(t) */
#define P_X 3
#define B_X 1000.0
#define STR_B_X "1000.0"
#define A_X 10.0
#define STR_A_X "10.0"

/* for y(t) */
#define P_Y 1
#define P_F 3
#define P_G 3
#define P_H 2
#define B_Y (pow(10.0, -9.0))
#define STR_B_Y "0.000000001"
#define A_Y (pow(10.0, -1.0))
#define STR_A_Y "0.1"
#define C_F (pow(10.0, -1.0))
#define STR_C_F "0.1"
#define C_G (pow(10.0,  0.0))
#define STR_C_G "1.0"
#define C_H (pow(10.0, -9.0))
#define STR_C_H "0.000000001"

/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_dvector_ij(DVector ret, long int index_start, long int index_end, DVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_dvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_dvector_i(ret, index_start + i, get_dvector_i(src, src_index_start + i));
//		printf("%d <----------------------------------> %d\n", index_start + i, src_index_start + i);
	}
}

/* double */
void ddf(DVector local_y, double x0, DVector y0, int myrank)
{
	long int i, j, k;
	long int im1, ip1, jm2, jp2;
	double tmp_f, tmp_g, tmp_h, num_xi, den_xi, num_yi, den_yi;
	static int init_flag = 0;
	static DVector tmp_xim1, tmp_xip1, tmp_xi, tmp_tmp_xi;
	static DVector tmp_yim1, tmp_yip1, tmp_yi, tmp_tmp_yi;

	if(init_flag == 0)
	{
		tmp_xim1   = init_dvector(DIM);
		tmp_xip1   = init_dvector(DIM);
		tmp_xi     = init_dvector(DIM);
		tmp_tmp_xi = init_dvector(DIM);
		tmp_yim1   = init_dvector(DIM);
		tmp_yip1   = init_dvector(DIM);
		tmp_yi     = init_dvector(DIM);
		tmp_tmp_yi = init_dvector(DIM);
		init_flag = 1;
	}

//	goto local_ret;

//	for(i = 0; i < DIM; i++)
//	{

	i = myrank; // for mpi

		im1 = i - 1;
		if(im1 < 0)
			im1 += DIM;
		copy_dvector_ij(tmp_xim1, 0, DIM - 1, y0, (2 * DIM) * im1      , (2 * DIM) * im1 + (DIM - 1));
		copy_dvector_ij(tmp_yim1, 0, DIM - 1, y0, (2 * DIM) * im1 + DIM, (2 * DIM) * (im1 + 1) - 1  );

		ip1 = i + 1;
		if(ip1 >= DIM)
			ip1 -= DIM;
		copy_dvector_ij(tmp_xip1, 0, DIM - 1, y0, (2 * DIM) * ip1      , (2 * DIM) * ip1 + (DIM - 1));
		copy_dvector_ij(tmp_yip1, 0, DIM - 1, y0, (2 * DIM) * ip1 + DIM, (2 * DIM) * (ip1 + 1) - 1  );

		copy_dvector_ij(tmp_xi, 0, DIM - 1, y0, (2 * DIM) * i      , (2 * DIM) * i + (DIM - 1));
		copy_dvector_ij(tmp_yi, 0, DIM - 1, y0, (2 * DIM) * i + DIM, (2 * DIM) * (i + 1) - 1  );

		for(j = 0; j < DIM; j++)
		{
			jm2 = j - 2;
			if(jm2 < 0)
				jm2 += DIM;
			jp2 = j + 2;
			if(jp2 >= DIM)
				jp2 -= DIM;

			tmp_f  = get_dvector_i(tmp_xim1, jm2);
			tmp_f += get_dvector_i(tmp_xim1, jp2);
			tmp_f += get_dvector_i(tmp_xip1, jm2);
			tmp_f += get_dvector_i(tmp_xip1, jp2);

			tmp_g = 0.0;
			for(k = 0; k < j; k++)
				tmp_g += get_dvector_i(tmp_yi, k);
			for(k = j + 1; k < DIM; k++)
				tmp_g += get_dvector_i(tmp_yi, k);

			tmp_h = get_dvector_i(tmp_yim1, j) + get_dvector_i(tmp_yip1, j);

			num_xi = pow(tmp_h, (double)P_X);
			den_xi = B_X + pow(tmp_h, (double)P_X);
			set_dvector_i(tmp_tmp_xi,
				j,
				num_xi / den_xi - A_X * get_dvector_i(tmp_xi, j)
			);

			num_yi = pow(get_dvector_i(tmp_yi, j), (double)P_Y) + C_F * pow(tmp_f, (double)P_F);
			den_yi = B_Y + pow(get_dvector_i(tmp_yi, j), (double)P_Y) + C_F * pow(tmp_f, (double)P_F) + C_G * pow(tmp_g, (double)P_G) + C_H * pow(tmp_h, (double)P_H);
			set_dvector_i(tmp_tmp_yi,
				j,
				num_yi / den_yi - A_Y * get_dvector_i(tmp_yi, j)
			);
		}

//		print_dvector(tmp_tmp_xi);
		copy_dvector_ij(local_y, 0  , DIM - 1    , tmp_tmp_xi, 0, DIM - 1);
		copy_dvector_ij(local_y, DIM, 2 * DIM - 1, tmp_tmp_yi, 0, DIM - 1);
//	}

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void dans(DVector y, double x)
{
	long int i;

	for(i = 0; i < y->dim; i++)
		set_dvector_i(y, i, exp((i + 1) * x));

	return;
}

void initial_dvalue(double *lf_initx, double *lf_maxx, DVector lf_inity)
{
	long int i, j;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	*lf_initx = 0.0;
	*lf_maxx = 100.0;
	for(i = 0; i < lf_inity->dim; i++)
		set_dvector_i(lf_inity, i, 1.0);

	for(i = 0; i < DIM; i++)
	{
		for(j = 0; j < DIM; j++)
			set_dvector_i(lf_inity, (2 * DIM) * i + DIM + j, 0.1);
		set_dvector_i(lf_inity, (2 * DIM) * i + DIM + i, 15.0);
	}
}

#ifdef USE_GMP
void copy_mpfvector_ij(MPFVector ret, long int index_start, long int index_end, MPFVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_mpfvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_mpfvector_i(ret, index_start + i, get_mpfvector_i(src, src_index_start + i));
//		printf("%d <----------------------------------> %d\n", index_start + i, src_index_start + i);
	}
}
/* mpf_t */
void df(MPFVector local_y, mpf_t x0, MPFVector y0, int myrank)
{
	long int i, j, k;
	long int im1, ip1, jm2, jp2;
	unsigned long prec;
	static mpf_t tmp_f, tmp_g, tmp_h, num_xi, den_xi, num_yi, den_yi;
	static mpf_t tmp_mpf[10];
	static mpf_t b_x, a_x;
	static mpf_t b_y, a_y, c_f, c_g, c_h;
	static long p_x, p_y, p_f, p_g, p_h;
	static int init_flag = 0;
	static MPFVector tmp_xim1, tmp_xip1, tmp_xi, tmp_tmp_xi;
	static MPFVector tmp_yim1, tmp_yip1, tmp_yi, tmp_tmp_yi;

	prec = local_y->prec;

	if(init_flag == 0)
	{
		mpf_init2(tmp_f, prec);
		mpf_init2(tmp_g, prec);
		mpf_init2(tmp_h, prec);
		mpf_init2(num_xi, prec);
		mpf_init2(den_xi, prec);
		mpf_init2(num_yi, prec);
		mpf_init2(den_yi, prec);
		for(i = 0; i < 5; i++)
			mpf_init2(tmp_mpf[i], prec);

		tmp_xim1   = init2_mpfvector(DIM, prec);
		tmp_xip1   = init2_mpfvector(DIM, prec);
		tmp_xi     = init2_mpfvector(DIM, prec);
		tmp_tmp_xi = init2_mpfvector(DIM, prec);
		tmp_yim1   = init2_mpfvector(DIM, prec);
		tmp_yip1   = init2_mpfvector(DIM, prec);
		tmp_yi     = init2_mpfvector(DIM, prec);
		tmp_tmp_yi = init2_mpfvector(DIM, prec);
		init_flag = 1;
	}

//	goto local_ret;

//	for(i = 0; i < DIM; i++)
//	{

	i = myrank;

		im1 = i - 1;
		if(im1 < 0)
			im1 += DIM;
		copy_mpfvector_ij(tmp_xim1, 0, DIM - 1, y0, (2 * DIM) * im1      , (2 * DIM) * im1 + (DIM - 1));
		copy_mpfvector_ij(tmp_yim1, 0, DIM - 1, y0, (2 * DIM) * im1 + DIM, (2 * DIM) * (im1 + 1) - 1  );

		ip1 = i + 1;
		if(ip1 >= DIM)
			ip1 -= DIM;
		copy_mpfvector_ij(tmp_xip1, 0, DIM - 1, y0, (2 * DIM) * ip1      , (2 * DIM) * ip1 + (DIM - 1));
		copy_mpfvector_ij(tmp_yip1, 0, DIM - 1, y0, (2 * DIM) * ip1 + DIM, (2 * DIM) * (ip1 + 1) - 1  );

		copy_mpfvector_ij(tmp_xi, 0, DIM - 1, y0, (2 * DIM) * i      , (2 * DIM) * i + (DIM - 1));
		copy_mpfvector_ij(tmp_yi, 0, DIM - 1, y0, (2 * DIM) * i + DIM, (2 * DIM) * (i + 1) - 1  );

		for(j = 0; j < DIM; j++)
		{
			jm2 = j - 2;
			if(jm2 < 0)
				jm2 += DIM;
			jp2 = j + 2;
			if(jp2 >= DIM)
				jp2 -= DIM;

			mpf_set(tmp_f, get_mpfvector_i(tmp_xim1, jm2));
			mpf_add(tmp_f, tmp_f, get_mpfvector_i(tmp_xim1, jp2));
			mpf_add(tmp_f, tmp_f, get_mpfvector_i(tmp_xip1, jm2));
			mpf_add(tmp_f, tmp_f, get_mpfvector_i(tmp_xip1, jp2));

			mpf_set_ui(tmp_g, 0UL);
			for(k = 0; k < j; k++)
				mpf_add(tmp_g, tmp_g, get_mpfvector_i(tmp_yi, k));
			for(k = j + 1; k < DIM; k++)
				mpf_add(tmp_g, tmp_g, get_mpfvector_i(tmp_yi, k));

			mpf_add(tmp_h, get_mpfvector_i(tmp_yim1, j), get_mpfvector_i(tmp_yip1, j));

			mpf_power(num_xi, tmp_h, (long)P_X);
			mpf_set_str(den_xi, STR_B_X, 10);
			mpf_add(den_xi, den_xi, num_xi);

			mpf_set_str(tmp_mpf[0], STR_A_X, 10);
			mpf_mul(tmp_mpf[0], tmp_mpf[0], get_mpfvector_i(tmp_xi, j));
			mpf_div(tmp_mpf[1], num_xi, den_xi);
			mpf_sub(tmp_mpf[0], tmp_mpf[1], tmp_mpf[0]);
			set_mpfvector_i(tmp_tmp_xi,
				j,
				tmp_mpf[0]
			);

			mpf_power(num_yi, get_mpfvector_i(tmp_yi, j), (long)P_Y);
			mpf_set_str(tmp_mpf[0], STR_C_F, 10);
			mpf_power(tmp_mpf[1], tmp_f, (long)P_F);
			mpf_mul(tmp_mpf[0], tmp_mpf[0], tmp_mpf[1]);
			mpf_add(num_yi, num_yi, tmp_mpf[0]);

			mpf_set_str(den_yi, STR_B_Y, 10);
			mpf_power(tmp_mpf[0], get_mpfvector_i(tmp_yi, j), (long)P_Y);
			mpf_set_str(tmp_mpf[1], STR_C_F, 10);
			mpf_power(tmp_mpf[2],  tmp_f, (long)P_F);
			mpf_mul(tmp_mpf[1], tmp_mpf[1], tmp_mpf[2]);
			mpf_set_str(tmp_mpf[2], STR_C_G, 10);
			mpf_power(tmp_mpf[3], tmp_g, (long)P_G);
			mpf_mul(tmp_mpf[2], tmp_mpf[2], tmp_mpf[3]);
			mpf_set_str(tmp_mpf[3], STR_C_H, 10);
			mpf_power(tmp_mpf[4], tmp_h, (long)P_H);
			mpf_mul(tmp_mpf[3], tmp_mpf[3], tmp_mpf[4]);
			mpf_add(den_yi, den_yi, tmp_mpf[0]);
			mpf_add(den_yi, den_yi, tmp_mpf[1]);
			mpf_add(den_yi, den_yi, tmp_mpf[2]);
			mpf_add(den_yi, den_yi, tmp_mpf[3]);

			mpf_div(tmp_mpf[0], num_yi, den_yi);
			mpf_set_str(tmp_mpf[1], STR_A_Y, 10);
			mpf_mul(tmp_mpf[1], tmp_mpf[1], get_mpfvector_i(tmp_yi, j));
			mpf_sub(tmp_mpf[0], tmp_mpf[0], tmp_mpf[1]);
			set_mpfvector_i(tmp_tmp_yi,
				j,
				tmp_mpf[0]
			);
		}

//		print_mpfvector(tmp_tmp_xi);
		copy_mpfvector_ij(local_y, 0  , DIM - 1    , tmp_tmp_xi, 0, DIM - 1);
		copy_mpfvector_ij(local_y, DIM, 2 * DIM - 1, tmp_tmp_yi, 0, DIM - 1);

//	}

	return;
}

/* y = [exp(x), ...., exp(x)]^T */
void ans(MPFVector y, mpf_t x)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, y->prec);
	for(i = 0; i < y->dim; i++)
	{
		mpf_mul_ui(tmp, x, i + 1);
		mpf_exp(tmp, tmp);
		set_mpfvector_i(y, i, tmp);
	}
	mpf_clear(tmp);
	return;
}


void initial_value(mpf_t lf_initx, mpf_t lf_maxx, MPFVector lf_inity)
{
	long int i, j;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	mpf_init_set_str(lf_initx, "0.0", 10);
	mpf_init_set_str(lf_maxx, "100.0", 10);
	for(i = 0; i < lf_inity->dim; i++)
		set_mpfvector_i_str(lf_inity, i, "1.0", 10);

	for(i = 0; i < DIM; i++)
	{
		for(j = 0; j < DIM; j++)
			set_mpfvector_i_str(lf_inity, (2 * DIM) * i + DIM + j, "0.1", 10);
		set_mpfvector_i_str(lf_inity, (2 * DIM) * i + DIM + i, "15.0", 10);
	}
}
#endif

#define DPREC 500

main(int argc, char *argv[])
{
	DVector dy, init_dy, lf_dtmp;
	long int div_num;
	double dx, init_dx, dstepsize, dabs_tol, drel_tol;
	double stime, etime;
#ifdef USE_GMP
	MPFVector y, init_y, lf_tmp;
	mpf_t x, init_x, stepsize, abs_tol, rel_tol;
#endif
	int myrank, num_procs;

/* for MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

#ifdef USE_GMP
	_mpi_set_bnc_default_prec_decimal(DPREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), ceil(DPREC/log10(2.0)), MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	/* x := 1 */
	mpf_init(init_x);
	mpf_init(x);

	y = init_mpfvector(TOTAL_DIM);
	lf_tmp = init_mpfvector(TOTAL_DIM);
	init_y = init_mpfvector(TOTAL_DIM);
	initial_value(init_x, x, init_y);
#endif

	dy = init_dvector(TOTAL_DIM);
	lf_dtmp = init_dvector(TOTAL_DIM);
	init_dy = init_dvector(TOTAL_DIM);
	initial_dvalue(&init_dx, &dx, init_dy);

	dstepsize = 0.1;
	dabs_tol = 0.0;
	drel_tol = 1.0e-10;
//	drel_tol = 1.0e-30;
//	drel_tol = 0.0;
#ifdef USE_GMP
	mpf_init_set_d(stepsize, dstepsize);
	mpf_init_set_d(abs_tol, dabs_tol);
	mpf_init_set_d(rel_tol, drel_tol);
//	mpf_init_set_d(rel_tol, 1.0e-20);
#endif

	goto HARMONIC;

	/* Extrapolation */
	printf("-- ex_nim -- \n");

	for(div_num = 2; div_num <= 128; div_num *= 2)
	{
		initial_dvalue(&init_dx, &dx, init_dy);
		stime = MPI_Wtime();
		_mpi_dex_nim_fs(NULL, dx, dy, init_dx, init_dy, div_num, ddf, drel_tol, dabs_tol, 4, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
/*			dans(init_dy, dx);
			dx = normi_dvector(init_dy);
			sub_dvector(init_dy, init_dy, dy);
			init_dx = normi_dvector(init_dy);
			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", init_dx, init_dx/dx);
*/
		}
#ifdef USE_GMP
		initial_value(init_x, x, init_y);
		stime = MPI_Wtime();
		_mpi_mpf_ex_nim_fs(NULL, x, y, init_x, init_y, div_num, df, rel_tol, abs_tol, 4, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
			ans(init_y, x);
			normi_mpfvector(x, init_y);
			sub_mpfvector(init_y, init_y, y);
			normi_mpfvector(init_x, init_y);
			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", mpf_get_d(init_x), mpf_get_d(init_x)/mpf_get_d(x));
		}
#endif
	}
	initial_dvalue(&init_dx, &dx, init_dy);
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 4, dans, MPI_COMM_WORLD);
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 16, dans, MPI_COMM_WORLD);
#ifdef USE_GMP
/* 4stages, 128bits */
/*
-- ex_nim ended --

real    2m57.379s
user    2m53.577s
sys     0m0.249s
*/
	initial_value(init_x, x, init_y);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 4, ans, 100, MPI_COMM_WORLD);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 8, ans, 1, MPI_COMM_WORLD);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_nim ended -- \n");

//	goto END;
HARMONIC:

	/* Extrapolation */
	printf("-- ex_harmonic -- \n");
	for(div_num = 256; div_num <= 128; div_num *= 2)
	{
		initial_dvalue(&init_dx, &dx, init_dy);
		if(myrank == 0)
			print_dvector(init_dy);
		stime = MPI_Wtime();
		_mpi_dex_harmonic_fs(NULL, dx, dy, init_dx, init_dy, div_num, ddf, drel_tol, dabs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
/*			dans(init_dy, dx);
			dx = normi_dvector(init_dy);
			sub_dvector(init_dy, init_dy, dy);
			init_dx = normi_dvector(init_dy);
			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", init_dx, init_dx/dx);
*/
		}
#ifdef USE_GMP
		initial_value(init_x, x, init_y);
		stime = MPI_Wtime();
		_mpi_mpf_ex_harmonic_fs(NULL, x, y, init_x, init_y, div_num, df, rel_tol, abs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
/*			ans(init_y, x);
			normi_mpfvector(x, init_y);
			sub_mpfvector(init_y, init_y, y);
			normi_mpfvector(init_x, init_y);
			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", mpf_get_d(init_x), mpf_get_d(init_x)/mpf_get_d(x));
*/
		}
#endif
	}
	initial_dvalue(&init_dx, &dx, init_dy);
	stime = MPI_Wtime();
	_mpi_dex_harmonic(NULL, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
	etime = MPI_Wtime();
	if(myrank == 0)
		printf("time(sec): %f\n", etime - stime);
//	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
//	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 16, dans, MPI_COMM_WORLD);
#ifdef USE_GMP
/* 4stages, 128bits */
/*-- ex_harmonic ended --

real    2m35.402s
user    2m29.843s
sys     0m0.531s
*/
	initial_value(init_x, x, init_y);
	stime = MPI_Wtime();
	_mpi_mpf_ex_harmonic(NULL, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 8, ans, 1, MPI_COMM_WORLD);
	etime = MPI_Wtime();
	if(myrank == 0)
		printf("time(sec): %f\n", etime - stime);
//	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 8, ans, 1, MPI_COMM_WORLD);
//	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_harmonic ended -- \n");

END:

	MPI_Finalize();
}
