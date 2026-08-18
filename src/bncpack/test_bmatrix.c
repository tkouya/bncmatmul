/********************************************************************************/
/* test bmatrix.c                                                               */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc.h"

#define DIM 15
#define PREC 256

int main()
{
	long int i, j;
	long int dim, upper_dim, lower_dim;
	FBMatrix fbmat;
	FVector fb, fx, fans;
	DBMatrix dbmat;
	DVector db, dx, dans;
#ifdef USE_GMP
	unsigned long prec;
	MPFBMatrix mpfbmat;
	MPFVector mpfb, mpfx, mpfans;
#endif

/***************************************/
/* Single Precision                    */
/***************************************/

	dim = DIM;

//	upper_dim = dim / 2;
	upper_dim = 0;
//	lower_dim = dim / 2;
	lower_dim = 0;

	//for(lower_dim = 0; lower_dim < dim / 2; lower_dim++)
	for(upper_dim = 0; upper_dim < dim / 2; upper_dim++)
	{
		// Initialize and set
		fbmat = init_fbmatrix(dim, upper_dim, lower_dim);
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
				set_fbmatrix_ij(fbmat, i, j, (float)rand());
		}
		print_fbmatrix(fbmat);

		// Initialize & set
		fx = init_fvector(dim);
		fb = init_fvector(dim);
		fans = init_fvector(dim);
		for(i = 0; i < dim; i++)
			set_fvector_i(fx, i, (float)(dim - i));

		// b := mat * x
		printf("b := mat * x\n");
		mul_fbmatrix_fvec(fb, fbmat, fx);

		// LUdecomp & solve
		printf("LUdecomp\n");
		FBLUdecomp(fbmat);
		printf("SolveLS\n");
		SolveFBLS(fans, fbmat, fb);

		for(i = 0; i < dim; i++)
			printf("%5d: %15.7e %15.7e %10.3e\n", i, get_fvector_i(fx, i), get_fvector_i(fans, i), fabs(get_fvector_i(fx, i) - get_fvector_i(fans, i)));

		// Free
		free_fbmatrix(fbmat);
		free_fvector(fx);
		free_fvector(fb);
		free_fvector(fans);
	}

/***************************************/
/* Double Precision                    */
/***************************************/

	dim = DIM;

//	upper_dim = dim / 2;
	upper_dim = 0;
//	lower_dim = dim / 2;
	lower_dim = 0;

	//for(lower_dim = 0; lower_dim < dim / 2; lower_dim++)
	for(upper_dim = 0; upper_dim < dim / 2; upper_dim++)
	{
		// Initialize and set
		dbmat = init_dbmatrix(dim, upper_dim, lower_dim);
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
				set_dbmatrix_ij(dbmat, i, j, (double)rand());
		}
		print_dbmatrix(dbmat);

		// Initialize & set
		dx = init_dvector(dim);
		db = init_dvector(dim);
		dans = init_dvector(dim);
		for(i = 0; i < dim; i++)
			set_dvector_i(dx, i, (double)(dim - i));

		// b := mat * x
		printf("b := mat * x\n");
		mul_dbmatrix_dvec(db, dbmat, dx);

		// LUdecomp & solve
		printf("LUdecomp\n");
		DBLUdecomp(dbmat);
		printf("SolveLS\n");
		SolveDBLS(dans, dbmat, db);

		for(i = 0; i < dim; i++)
			printf("%5d: %25.17e %25.17e %10.3e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i), fabs(get_dvector_i(dx, i) - get_dvector_i(dans, i)));

		// Free
		free_dbmatrix(dbmat);
		free_dvector(dx);
		free_dvector(db);
		free_dvector(dans);
	}

#ifdef USE_GMP

/***************************************/
/* Multiple Precision                  */
/***************************************/

	dim = DIM;
	prec = PREC;

//	upper_dim = dim / 2;
	upper_dim = 0;
//	lower_dim = dim / 2;
	lower_dim = 0;

	//for(lower_dim = 0; lower_dim < dim / 2; lower_dim++)
	for(upper_dim = 0; upper_dim < dim / 2; upper_dim++)
	{
		set_bnc_default_prec(prec);
		
		// Initialize and set
		mpfbmat = init_mpfbmatrix(dim, upper_dim, lower_dim);
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
				set_mpfbmatrix_ij_d(mpfbmat, i, j, (double)rand());
		}
		print_mpfbmatrix(mpfbmat);

		// Initialize & set
		mpfx = init_mpfvector(dim);
		mpfb = init_mpfvector(dim);
		mpfans = init_mpfvector(dim);
		for(i = 0; i < dim; i++)
			set_mpfvector_i_ui(mpfx, i, (unsigned long)(dim - i));

		// b := mat * x
		printf("b := mat * x\n");
		mul_mpfbmatrix_mpfvec(mpfb, mpfbmat, mpfx);

		// LUdecomp & solve
		printf("LUdecomp\n");
		MPFBLUdecomp(mpfbmat);
		printf("SolveLS\n");
		SolveMPFBLS(mpfans, mpfbmat, mpfb);

		for(i = 0; i < dim; i++)
			printf("%5d: %25.17e %25.17e %10.3e\n", i, mpf2double(get_mpfvector_i(mpfx, i)), mpf2double(get_mpfvector_i(mpfans, i)), fabs(mpf2double(get_mpfvector_i(mpfx, i)) - mpf2double(get_mpfvector_i(mpfans, i))));

		// Free
		free_mpfbmatrix(mpfbmat);
		free_mpfvector(mpfx);
		free_mpfvector(mpfb);
		free_mpfvector(mpfans);
	}
#endif

	return BNC_SUCCESS;
}
