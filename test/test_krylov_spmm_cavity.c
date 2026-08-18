/********************************************************************************/
/* test_krylov_spmm_cavity.c:                                                   */
/* Copyright (C) 2004-2012 Tomonori Kouya                                       */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
/****************************************************/
/* Test Program for *BiCG, *CGS, *BiCGSTAB, *GPBiCG */
/****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncmatmul.h"
#include "get_secv.h" // get_secv
//#include "bnc.h"
//#include "bncsparse.h"
//#include "bncmm.h"

/* Downloaded file from UF Sparse Matrix Collection */
#define MATRIX_MM_FILE "cavity04/cavity04.mtx"
#define VEC_ANS_MM_FILE "cavity04/cavity04_dx.mtx"
#define VEC_B_MM_FILE "cavity04/cavity04_db.mtx"

#ifdef USE_GMP
void free_mpfproblem(MPFMatrix a, MPFVector b, MPFVector ans)
{
	if(a != NULL)
		free_mpfmatrix(a);
	if(b != NULL)
		free_mpfvector(b);
	if(ans != NULL)
		free_mpfvector(ans);
}

void free_mpfproblem_sp(MPFRSMatrix a, MPFVector b, MPFVector ans)
{
	if(a != NULL)
		free_mpfrsmatrix(a);
	if(b != NULL)
		free_mpfvector(b);
	if(ans != NULL)
		free_mpfvector(ans);
}
#endif // USE_GMP

#define MAXTIMES 1000
#define PREC 512

// read MatrixMarket format (coodinate or array type only) as vector if possible
DVector test_init_dvector_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	double val;
	DVector dvec;

	if(fname == NULL)
	{
		fprintf(stderr, "(init_dvector_readMMcoordinate) fname is not set\n");
		return NULL;
	}
    else
    { 
        if ((f = fopen(fname, "r")) == NULL)
		{
			fprintf(stderr, "(init_dvector_readMMcoordinate) cannot read %s!!\n", fname);
            return NULL;
		}
    }

	printf("(init_dvector_readMMcoodinate) Reading %s ...\n", fname);
    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        return NULL;
    }
	// print information about MM file
    mm_write_banner(stdout, matcode);

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
	if(mm_is_complex(matcode))
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

/*    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }
*/
	// symmetric matrix
	if(mm_is_symmetric(matcode))
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) Symmetrix matrix is not supported in current BNCpack!\n");
		return NULL;
	}

    /* find out size of sparse matrix .... */

	if(mm_is_coordinate(matcode))
	{
		ret_code = mm_read_mtx_crd_size(f, &row_dim, &col_dim, &nzero_total_num);
	    if (ret_code !=0)
	        return NULL;

	    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);
	}
	else if(mm_is_array(matcode))
	{
		ret_code = mm_read_mtx_array_size(f, &row_dim, &col_dim);
		if(ret_code != 0)
			return NULL;
		mm_write_mtx_array_size(stdout, row_dim, col_dim);
	}
	else
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) %s is not supported format\n", fname);
		return NULL;
	}

	// check size of vector
	if(col_dim >= 2)
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) %s is not a vector(rod_dim = %d, col_dim = %d)\n", fname, row_dim, col_dim);
		return NULL;
	}

	// initialize dmatrix
	dvec = init_dvector(row_dim);

	if(dvec == NULL)
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) ERROR: cannot allocate vector area(dim= %d x %d\n", row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	if(mm_is_coordinate(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
	        row_index--;
	        col_index--;

			// set
			set_dvector_i(dvec, row_index, val);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
	else if(mm_is_array(matcode))
	{
		nzero_total_num = row_dim * col_dim;
	    for (i = 0; i < nzero_total_num; i++)
	    {

	        fscanf(f, "%lg\n", &val);
//	        printf("%d: %e\n", i, val);

			// set
			set_dvector_i(dvec, i, val);
		}
	}
	else
	{
		fprintf(stderr, "(init_dvector_readMMcoodinate) %s is not supported format\n", fname);
		free_dvector(dvec);
		return NULL;
	}

    if (f !=stdin) fclose(f);

	return dvec;
}

// read MatrixMarket format (coodinate type only)
MPFRSMatrix test_init2_mpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num, new_nzero_total_num, nzero_diagonal_num;
    int row_index, col_index;
    int symmetric_flag;
    long int *nzero_col_dim;
    long int i, j, total_index;
	mmcoordinate_str *buf, *new_buf;
	int flag_new_buf = 0;
	char val_str[MAX_VAL_STR];
	MPFRSMatrix dsmat;

	if(fname == NULL)
	{
		fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL)
		{
			fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) cannot read %s!!\n", fname);
            return NULL;
		}
    }
	printf("mm_read_banner...!\n");
    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) Could not process Matrix Market banner.\n");
        return NULL;
    }
	printf("mm_write_banner...!\n");
	// print information about MM file
    mm_write_banner(stdout, matcode);

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
	if(mm_is_complex(matcode))
	{
		fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) Complex number is not supported in current BNCmatmul!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }
	printf("mm_is_coordinate...!\n");
	symmetric_flag = 0;
	if(mm_is_symmetric(matcode))
	{
		symmetric_flag = 1;
//		fprintf(stderr, "Symmetrix matrix is not supported in current BNCpack!\n");
//		return NULL;
	}

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &row_dim, &col_dim, &nzero_total_num)) !=0)
        return NULL;

    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);

	if(row_dim != col_dim)
	{
		fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) BNCmatmul supports only square sparse matrices!(%d, %d)\n", row_index, col_index);
		return NULL;
	}

	// initialize buffers
	printf("initialize buffers...!\n");
	nzero_col_dim = (long int *)calloc(sizeof(long int), row_dim);
	//row_index = (long int *)calloc(sizeof(long int), nzero_total_num);
	//col_index = (long int *)calloc(sizeof(long int), nzero_total_num);
	//val = (double *)calloc(sizeof(double), nzero_total_num);
	buf = (mmcoordinate_str *)calloc(sizeof(mmcoordinate_str), nzero_total_num);

	if((nzero_col_dim == NULL) || (buf == NULL))
		return NULL;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	nzero_diagonal_num = 0;

	// fix! : 2012-07-13
	for(i = 0; i < row_dim; i++)
		nzero_col_dim[i] = 0;

    for (i = 0; i < nzero_total_num; i++)
    {
        fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
        row_index--;
        col_index--;
//		dsmat->nzero_index[row_index][dsmat->nzero_col_dim[row_index]] = col_index;
		//printf("%d %d %d->%d\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
		nzero_col_dim[row_index]++;

		// set
		buf[i].row_index = row_index;
		buf[i].col_index = col_index;
		strcpy(buf[i].val_str, val_str);
//		printf("val_str -> %s-->%s--\n", val_str, buf[i].val_str);

		// count nonzero diagonal elements
		if(row_index == col_index)
			nzero_diagonal_num++;

		/* if(i % 100 == 0)
			fprintf(stderr, "reading %d -th row...\n", row_index);
		*/
	}

	printf("1st fclose...!\n");
    if (f !=stdin) fclose(f);

	new_buf = buf;
	new_nzero_total_num = nzero_total_num;

	// if symmetric matrix...
	if(symmetric_flag == 1)
	{
		new_nzero_total_num = (nzero_total_num - nzero_diagonal_num) * 2 + nzero_diagonal_num;
		if(new_nzero_total_num > nzero_total_num)
		{
			fprintf(stderr, "Reallocation...%d -> %d\n", nzero_total_num, new_nzero_total_num);
		//	new_buf = (mmcoordinate_str *)realloc(buf, sizeof(mmcoordinate_str) * new_nzero_total_num);
			new_buf = (mmcoordinate_str *)malloc(sizeof(mmcoordinate_str) * new_nzero_total_num);
		//	fprintf(stderr, "reallocation is success!\n");
			flag_new_buf = 1; // newly malloc
			if(new_buf == NULL)
			{
				fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) Error: cannot allocate buf or nzero_col_dim!\n");
				return NULL;
			}

			total_index = nzero_total_num;

			// fix! : 2012-07-13
			for(i = 0; i < row_dim; i++)
				nzero_col_dim[i] = 0;

			for(i = 0; i < nzero_total_num; i++)
			{
				// set
			//	printf("set %d(%s) -> ", i, buf[i].val_str);
				// fix! : 2012-07-13
				nzero_col_dim[buf[i].row_index]++;

				new_buf[i].row_index = buf[i].row_index;
				new_buf[i].col_index = buf[i].col_index;
				strcpy(new_buf[i].val_str, buf[i].val_str);
			//	printf("val_str -> %s-->%s--\n", buf[i].val_str, new_buf[i].val_str);

				// copy to opposite triangular area except diagonal element
				if(buf[i].row_index != buf[i].col_index)
				{
			//		printf("set %d -> ", total_index);
					// fix! : 2012-07-13
					nzero_col_dim[buf[i].col_index]++;

					new_buf[total_index].row_index = buf[i].col_index;
					new_buf[total_index].col_index = buf[i].row_index;
					strcpy(new_buf[total_index].val_str, buf[i].val_str);
					total_index++;
				}
			}
			//buf = new_buf;
			//nzero_total_num = new_nzero_total_num;
		}
	}
	printf("qsort...!\n");
	// sorting
//	qsort(buf, nzero_total_num, sizeof(mmcoordinate_str), compare_mmcoordinate_str);
	qsort(new_buf, new_nzero_total_num, sizeof(mmcoordinate_str), compare_mmcoordinate_str);
//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %s %d\n", buf[i].row_index, buf[i].col_index, buf[i].val_str, nzero_col_dim[buf[i].row_index]);


	printf("init2_mpfrsmatrix...!\n");
	// initialize drsmatrix
	dsmat = init2_mpfrsmatrix(row_dim, nzero_col_dim, new_nzero_total_num, prec);
	if(dsmat == NULL)
	{
		fprintf(stderr, "(_init2_mpfrsmatrix_readMMcoordinate) ERROR: cannot allocate matrix area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
		return NULL;
	}

	// copy buffers to drsmatrix
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < nzero_col_dim[i]; j++)
		{
			mpf_set_str(dsmat->element[total_index], new_buf[total_index].val_str, 10);
			dsmat->nzero_index[i][j] = new_buf[total_index].col_index;
			total_index++;
		}
	}

	printf("free bufs...!\n");
	if(buf != NULL)
		free(buf);

	buf = NULL;
	printf("free new_buf...!\n");
	if(flag_new_buf == 1)
		free(new_buf);

	printf("return!!");
	return dsmat;
}

int main(void)
{
	long int dim;
	long int maxtimes = MAXTIMES;
	DMatrix da;
	DRSMatrix da_sp;
	DVector db, dx, dans;
	double start, dtime[5], dtime_sp[5];
	long int itimes_d[5], itimes_d_sp[5];
	DDMatrix dda;
	DDRSMatrix dda_sp;
	DDVector ddb, ddx, ddans;
	double ddrel[DDSIZE], ddabs[DDSIZE];
	double ddtime[5], ddtime_sp[5];
	long int itimes_dd[5], itimes_dd_sp[5];
	TDMatrix tda;
	TDRSMatrix tda_sp;
	TDVector tdb, tdx, tdans;
	double tdrel[TDSIZE], tdabs[TDSIZE];
	double tdtime[5], tdtime_sp[5];
	long int itimes_td[5], itimes_td_sp[5];
	QDMatrix qda;
	QDRSMatrix qda_sp;
	QDVector qdb, qdx, qdans;
	double qdrel[QDSIZE], qdabs[QDSIZE];
	double qdtime[5], qdtime_sp[5];
	long int itimes_qd[5], itimes_qd_sp[5];
	long int i, j;

#ifdef USE_GMP
	MPFMatrix mpfa;
	MPFVector mpfb, mpfx, mpfans;
	mpf_t reps, aeps;
	MPFMatrix mpfa2;
	MPFVector mpfb2, mpfx2, mpfans2;
	mpf_t reps2, aeps2;
	MPFMatrix mpfa3;
	MPFVector mpfb3, mpfx3, mpfans3;
	mpf_t reps3, aeps3;
	MPFMatrix mpfa4;
	MPFVector mpfb4, mpfx4, mpfans4;
	mpf_t reps4, aeps4;
	MPFMatrix mpfa5;
	MPFVector mpfb5, mpfx5, mpfans5;
	mpf_t reps5, aeps5;
	long int itimes_mpf[4][5], itimes_mpf_sp[4][5];
	double mpftime[4][5], mpftime_sp[4][5];

	MPFRSMatrix mpfa_sp;
	MPFRSMatrix mpfa2_sp;
	MPFRSMatrix mpfa3_sp;
	MPFRSMatrix mpfa4_sp;
	MPFRSMatrix mpfa5_sp;
#endif // USE_GMP

//goto MPFR;
//goto DD;

/* Double */
	/* initialize & get problem */
	//rintf("Reading %s...\n", VEC_ANS_MM_FILE);
	dans = test_init_dvector_readMMcoordinate(VEC_ANS_MM_FILE);
	print_dvector(dans);

	dim = dans->dim;
	dx = init_dvector(dim);

	/* run DBiCG_sp */
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DBiCG_sp\n", dim);
	start = get_secv();
	itimes_d_sp[0] = DBiCG_sp(dx, da_sp, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime_sp[0] = get_secv() - start;

	free_dvector(db);
	free_drsmatrix(da_sp);

	/* run DBiCG_sp */
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DBiCG\n", dim);
	start = get_secv();
	itimes_d[0]    = DBiCG(dx, da, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime[0] = get_secv() - start;

	free_dmatrix(da);
	free_dvector(db);

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run DCGS_sp */
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DCGS_sp\n", dim);
	start = get_secv();
	itimes_d_sp[1] = DCGS_sp(dx, da_sp, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime_sp[1] = get_secv() - start;

	// free problem
	free_dvector(db);
	free_drsmatrix(da_sp);

	/* run DCGS */
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DCGS\n", dim);
	start = get_secv();
	itimes_d[1] = DCGS(dx, da, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime[1] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_dmatrix(da);
	free_dvector(db);

	/* run DBiCGSTAB_sp */
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DBiCGSTAB_sp\n", dim);
	start = get_secv();
	itimes_d_sp[2] = DBiCGSTAB_sp(dx, da_sp, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime_sp[2] = get_secv() - start;

	// free problem
	free_dvector(db);
	free_drsmatrix(da_sp);

	/* run DBiCGSTAB */
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DBiCGSTAB\n", dim);
	start = get_secv();
	itimes_d[2] = DBiCGSTAB(dx, da, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime[2] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_dvector(db);
	free_dmatrix(da);

	/* run DGPBiCG_sp */
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);

	printf("\n-- dim = %ld, DGPBiCG_sp\n", dim);
	start = get_secv();
	itimes_d_sp[3] = DGPBiCG_sp(dx, da_sp, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime_sp[3] = get_secv() - start;

	// free problem
	free_dvector(db);
	free_drsmatrix(da_sp);

	/* run DGPBiCG */
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
	db = init_dvector_readMMcoordinate(VEC_B_MM_FILE);

	printf("\n-- dim = %ld, DGPBiCG\n", dim);
	start = get_secv();
	itimes_d[3] = DGPBiCG(dx, da, db, 1.0e-13, 1.0e-99, maxtimes);
	dtime[3] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_dvector(db);
	free_dmatrix(da);

	/* end */
	free_dvector(dans);
	free_dvector(dx);

// DD
DD:
	/* initialize & get problem */
	printf("DD Reading %s...\n", VEC_ANS_MM_FILE);fflush(stdout);
	mpfans = init_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE);
	mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfa = init_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfb = init_mpfvector_readMMcoordinate(VEC_B_MM_FILE);
	dim = mpfans->dim;
	ddx = init_ddvector(dim);
	ddb = init_ddvector(dim);
	ddans = init_ddvector(dim);
    dda_sp = init_set_ddrsmatrix_mpfrsmatrix(mpfa_sp);
	dda = init_ddmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_ddmatrix_mpfmat(dda, mpfa);

	subst_ddvector_mpfvec(ddans, mpfans);
	subst_ddvector_mpfvec(ddb, mpfans);
	//print_ddvector(ddans);

	ddrel[0] = 1.0e-13; ddrel[1] = 0.0;
	ddabs[0] = 1.0e-99; ddabs[1] = 0.0;

	/* run DDBiCG_sp */
	printf("\n-- dim = %ld, DDBiCG_sp\n", dim);
	start = get_secv();
	itimes_dd_sp[0] = DDBiCG_sp(ddx, dda_sp, ddb, ddrel, ddabs, maxtimes);
	ddtime_sp[0] = get_secv() - start;

	//free_ddvector(ddb);
	//free_ddrsmatrix(dda_sp);

	/* run DBiCG_sp */
	printf("\n-- dim = %ld, DBiCG\n", dim);
	start = get_secv();
	itimes_dd[0]    = DDBiCG(ddx, dda, ddb, ddrel, ddabs, maxtimes);
	ddtime[0] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run DCGS_sp */
	printf("\n-- dim = %ld, DCGS_sp\n", dim);
	start = get_secv();
	itimes_dd_sp[1] = DDCGS_sp(ddx, dda_sp, ddb, ddrel, ddabs, maxtimes);
	ddtime_sp[1] = get_secv() - start;

	/* run DCGS */
	printf("\n-- dim = %ld, DCGS\n", dim);
	start = get_secv();
	itimes_dd[1] = DDCGS(ddx, dda, ddb, ddrel, ddabs, maxtimes);
	ddtime[1] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	//free_ddmatrix(dda);
	//free_ddvector(ddb);

	/* run DDBiCGSTAB_sp */
	printf("\n-- dim = %ld, DDBiCGSTAB_sp\n", dim);
	start = get_secv();
	itimes_dd_sp[2] = DDBiCGSTAB_sp(ddx, dda_sp, ddb, ddrel, ddabs, maxtimes);
	ddtime_sp[2] = get_secv() - start;

	/* run DBiCGSTAB */
	printf("\n-- dim = %ld, DDBiCGSTAB\n", dim);
	start = get_secv();
	itimes_dd[2] = DDBiCGSTAB(ddx, dda, ddb, ddrel, ddabs, maxtimes);
	ddtime[2] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run DDGPBiCG_sp */
	printf("\n-- dim = %ld, DDGPBiCG_sp\n", dim);
	start = get_secv();
	itimes_dd_sp[3] = DDGPBiCG_sp(ddx, dda_sp, ddb, ddrel, ddabs, maxtimes);
	ddtime_sp[3] = get_secv() - start;

	/* run DDGPBiCG */
	printf("\n-- dim = %ld, DDGPBiCG\n", dim);
	start = get_secv();
	itimes_dd[3] = DDGPBiCG(ddx, dda, ddb, ddrel, ddabs, maxtimes);
	ddtime[3] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_ddvector(ddb);
	free_ddmatrix(dda);
	free_mpfvector(mpfb);
	free_mpfvector(mpfans);
	free_mpfrsmatrix(mpfa_sp);
	free_mpfmatrix(mpfa);

	/* end */
	free_ddvector(ddans);
	free_ddvector(ddx);

// TD
TD:
	/* initialize & get problem */
	printf("TD Reading %s...\n", VEC_ANS_MM_FILE);fflush(stdout);
	mpfans = init_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE);
	mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfa = init_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfb = init_mpfvector_readMMcoordinate(VEC_B_MM_FILE);
	dim = mpfans->dim;
	tdx = init_tdvector(dim);
	tdb = init_tdvector(dim);
	tdans = init_tdvector(dim);
    tda_sp = init_set_tdrsmatrix_mpfrsmatrix(mpfa_sp);
	tda = init_tdmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_tdmatrix_mpfmat(tda, mpfa);

	subst_tdvector_mpfvec(tdans, mpfans);
	subst_tdvector_mpfvec(tdb, mpfans);
	//print_tdvector(tdans);

	tdrel[0] = 1.0e-13; tdrel[1] = 0.0; tdrel[2] = 0.0;
	tdabs[0] = 1.0e-99; tdabs[1] = 0.0; tdabs[2] = 0.0;

	/* run TDBiCG_sp */
	printf("\n-- dim = %ld, TDBiCG_sp\n", dim);
	start = get_secv();
	itimes_td_sp[0] = TDBiCG_sp(tdx, tda_sp, tdb, tdrel, tdabs, maxtimes);
	tdtime_sp[0] = get_secv() - start;

	//free_tdvector(ddb);
	//free_tdrsmatrix(dda_sp);

	/* run TDBiCG_sp */
	printf("\n-- dim = %ld, TDBiCG\n", dim);
	start = get_secv();
	itimes_td[0]    = TDBiCG(tdx, tda, tdb, tdrel, tdabs, maxtimes);
	tdtime[0] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run TDCGS_sp */
	printf("\n-- dim = %ld, TDCGS_sp\n", dim);
	start = get_secv();
	itimes_td_sp[1] = TDCGS_sp(tdx, tda_sp, tdb, tdrel, tdabs, maxtimes);
	tdtime_sp[1] = get_secv() - start;

	/* run TDCGS */
	printf("\n-- dim = %ld, TDCGS\n", dim);
	start = get_secv();
	itimes_td[1] = TDCGS(tdx, tda, tdb, tdrel, tdabs, maxtimes);
	tdtime[1] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	//free_tdmatrix(dda);
	//free_tdvector(ddb);

	/* run TDBiCGSTAB_sp */
	printf("\n-- dim = %ld, TDBiCGSTAB_sp\n", dim);
	start = get_secv();
	itimes_td_sp[2] = TDBiCGSTAB_sp(tdx, tda_sp, tdb, tdrel, tdabs, maxtimes);
	tdtime_sp[2] = get_secv() - start;

	/* run TDBiCGSTAB */
	printf("\n-- dim = %ld, TDBiCGSTAB\n", dim);
	start = get_secv();
	itimes_td[2] = TDBiCGSTAB(tdx, tda, tdb, tdrel, tdabs, maxtimes);
	tdtime[2] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run TDGPBiCG_sp */
	printf("\n-- dim = %ld, TDGPBiCG_sp\n", dim);
	start = get_secv();
	itimes_td_sp[3] = TDGPBiCG_sp(tdx, tda_sp, tdb, tdrel, tdabs, maxtimes);
	tdtime_sp[3] = get_secv() - start;

	/* run DDGPBiCG */
	printf("\n-- dim = %ld, TDGPBiCG\n", dim);
	start = get_secv();
	itimes_td[3] = TDGPBiCG(tdx, tda, tdb, tdrel, tdabs, maxtimes);
	tdtime[3] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_tdvector(tdb);
	free_tdmatrix(tda);
	free_mpfvector(mpfb);
	free_mpfvector(mpfans);
	free_mpfrsmatrix(mpfa_sp);
	free_mpfmatrix(mpfa);

	/* end */
	free_tdvector(tdans);
	free_tdvector(tdx);

// QD
QD:
	/* initialize & get problem */
	printf("QD Reading %s...\n", VEC_ANS_MM_FILE);fflush(stdout);
	mpfans = init_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE);
	mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfa = init_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE);
	mpfb = init_mpfvector_readMMcoordinate(VEC_B_MM_FILE);
	dim = mpfans->dim;
	qdx = init_qdvector(dim);
	qdb = init_qdvector(dim);
	qdans = init_qdvector(dim);
    qda_sp = init_set_qdrsmatrix_mpfrsmatrix(mpfa_sp);
	qda = init_qdmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_qdmatrix_mpfmat(qda, mpfa);

	subst_qdvector_mpfvec(qdans, mpfans);
	subst_qdvector_mpfvec(qdb, mpfans);
	//print_qdvector(tdans);

	qdrel[0] = 1.0e-13; qdrel[1] = 0.0; qdrel[2] = 0.0; qdrel[3] = 0.0;
	qdabs[0] = 1.0e-99; qdabs[1] = 0.0; qdabs[2] = 0.0; qdabs[3] = 0.0;

	/* run QDBiCG_sp */
	printf("\n-- dim = %ld, QDBiCG_sp\n", dim);
	start = get_secv();
	itimes_qd_sp[0] = QDBiCG_sp(qdx, qda_sp, qdb, qdrel, qdabs, maxtimes);
	qdtime_sp[0] = get_secv() - start;

	//free_tdvector(ddb);
	//free_tdrsmatrix(dda_sp);

	/* run TDBiCG_sp */
	printf("\n-- dim = %ld, QDBiCG\n", dim);
	start = get_secv();
	itimes_qd[0]    = QDBiCG(qdx, qda, qdb, qdrel, qdabs, maxtimes);
	qdtime[0] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run TDCGS_sp */
	printf("\n-- dim = %ld, QDCGS_sp\n", dim);
	start = get_secv();
	itimes_qd_sp[1] = QDCGS_sp(qdx, qda_sp, qdb, qdrel, qdabs, maxtimes);
	qdtime_sp[1] = get_secv() - start;

	/* run TDCGS */
	printf("\n-- dim = %ld, QDCGS\n", dim);
	start = get_secv();
	itimes_qd[1]    = QDCGS(qdx, qda, qdb, qdrel, qdabs, maxtimes);
	qdtime[1] = get_secv() - start;

	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	//free_tdmatrix(dda);
	//free_tdvector(ddb);

	/* run TDBiCGSTAB_sp */
	printf("\n-- dim = %ld, QDBiCGSTAB_sp\n", dim);
	start = get_secv();
	itimes_qd_sp[2] = QDBiCGSTAB_sp(qdx, qda_sp, qdb, qdrel, qdabs, maxtimes);
	qdtime_sp[2] = get_secv() - start;

	/* run TDBiCGSTAB */
	printf("\n-- dim = %ld, QDBiCGSTAB\n", dim);
	start = get_secv();
	itimes_qd[2] = QDBiCGSTAB(qdx, qda, qdb, qdrel, qdabs, maxtimes);
	qdtime[2] = get_secv() - start;
	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	/* run TDGPBiCG_sp */
	printf("\n-- dim = %ld, QDGPBiCG_sp\n", dim);
	start = get_secv();
	itimes_qd_sp[3] = QDGPBiCG_sp(qdx, qda_sp, qdb, qdrel, qdabs, maxtimes);
	qdtime_sp[3] = get_secv() - start;

	/* run DDGPBiCG */
	printf("\n-- dim = %ld, TDGPBiCG\n", dim);
	start = get_secv();
	itimes_qd[3] = QDGPBiCG(qdx, qda, qdb, qdrel, qdabs, maxtimes);
	qdtime[3] = get_secv() - start;
	/* print */
//	for(i = 0; i < dim; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));

	// free problem
	free_qdvector(qdb);
	free_qdmatrix(qda);
	free_mpfvector(mpfb);
	free_mpfvector(mpfans);
	free_mpfrsmatrix(mpfa_sp);
	free_mpfmatrix(mpfa);

	/* end */
	free_qdvector(qdans);
	free_qdvector(qdx);

MPFR:
#ifdef USE_GMP
/* MPF */
	set_bnc_default_prec(PREC);
	/* initialize */
	mpf_init(reps);
	mpf_init2(reps2, PREC * 2);
	mpf_init2(reps3, PREC * 4);
	mpf_init2(reps4, PREC * 8);
	mpf_init2(reps5, PREC * 16);
	mpf_init(aeps);
	mpf_init2(aeps2, PREC * 2);
	mpf_init2(aeps3, PREC * 4);
	mpf_init2(aeps4, PREC * 8);
	mpf_init2(aeps5, PREC * 16);

//	print_mpfmatrix(mpfa);
//	print_mpfmatrix(mpfa2);
//	print_mpfmatrix(mpfa3);

	mpf_set_d(reps, 1.0e-20);
	mpf_set_d(reps2, 1.0e-20);
	mpf_set_d(reps3, 1.0e-20);
	mpf_set_d(reps4, 1.0e-20);
	mpf_set_d(reps5, 1.0e-20);
	mpf_set_d(aeps, 1.0e-50);
	mpf_set_d(aeps2, 1.0e-50);
	mpf_set_d(aeps3, 1.0e-50);
	mpf_set_d(aeps4, 1.0e-50);
	mpf_set_d(aeps5, 1.0e-50);

//	goto MPFGPBiCG;
MPFBiCG:

#ifndef SPARSEONLY
	/* run MPFBiCG */
	/* load */
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans);
//	mpfa_sp = get_mpfproblem_sp(mpfa, mpfb, mpfans);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);

	printf("\n--- dim = %ld, prec = %ld, MPFBiCG\n", dim, (unsigned long)PREC);
	start = get_secv();
	itimes_mpf[0][0]  = MPFBiCG(mpfx, mpfa, mpfb, reps, aeps, maxtimes);
	mpftime[0][0] = get_secv() - start;
	free_mpfproblem(mpfa, mpfb, mpfans);

	printf("\n--- dim = %ld, prec = %ld, MPFBiCG\n", dim, (unsigned long)PREC * 2);
	mpfa2 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf[0][1] = MPFBiCG(mpfx2, mpfa2, mpfb2, reps2, aeps2, maxtimes);
	mpftime[0][1] = get_secv() - start;
	free_mpfproblem(mpfa2, mpfb2, mpfans2);

	printf("\n--- dim = %ld, prec = %ld, MPFBiCG\n", dim, (unsigned long)PREC * 4);
	mpfa3 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);

	start = get_secv();
	itimes_mpf[0][2] = MPFBiCG(mpfx3, mpfa3, mpfb3, reps3, aeps3, maxtimes);
	mpftime[0][2] = get_secv() - start;
	free_mpfproblem(mpfa3, mpfb3, mpfans3);

	printf("\n--- dim = %ld, prec = %ld, MPFBiCG\n", dim, (unsigned long)PREC * 8);
	mpfa4 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);

	start = get_secv();
	itimes_mpf[0][3] = MPFBiCG(mpfx4, mpfa4, mpfb4, reps4, aeps4, maxtimes);
	mpftime[0][3] = get_secv() - start;
	free_mpfproblem(mpfa4, mpfb4, mpfans4);

	printf("\n--- dim = %ld, prec = %ld, MPFBiCG\n", dim, (unsigned long)PREC * 16);
	mpfa5 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);

	start = get_secv();
	itimes_mpf[0][4] = MPFBiCG(mpfx5, mpfa5, mpfb5, reps5, aeps5, maxtimes);
	mpftime[0][4] = get_secv() - start;
	free_mpfproblem(mpfa5, mpfb5, mpfans5);

	/* print */
//	print_mpfvector(mpfx);
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
#endif // SPARSEONLY

MPFBiCG_sp:

	/* run MPFBiCG */
//	get_mpfproblem(mpfa, mpfb, mpfans);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans, PREC);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	printf("\n-- dim = %ld, prec = %ld, MPFBiCG_sp\n", dim, (unsigned long)PREC);
	mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);

#ifdef SPARSEONLY
	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);
#endif // SPARSEONLY

	start = get_secv();
	itimes_mpf_sp[0][0]  = MPFBiCG_sp(mpfx, mpfa_sp, mpfb, reps, aeps, maxtimes);
	mpftime_sp[0][0] = get_secv() - start;
	free_mpfproblem_sp(mpfa_sp, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCG_sp\n", dim, (unsigned long)PREC * 2);
	mpfa2_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf_sp[0][1] = MPFBiCG_sp(mpfx2, mpfa2_sp, mpfb2, reps2, aeps2, maxtimes);
	mpftime_sp[0][1] = get_secv() - start;
	free_mpfproblem_sp(mpfa2_sp, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCG_sp\n", dim, (unsigned long)PREC * 4);
	mpfa3_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);
//	get_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3, PREC * 4);
	start = get_secv();
	itimes_mpf_sp[0][2] = MPFBiCG_sp(mpfx3, mpfa3_sp, mpfb3, reps3, aeps3, maxtimes);
	mpftime_sp[0][2] = get_secv() - start;
	free_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCG_sp\n", dim, (unsigned long)PREC * 8);
	mpfa4_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);
	//get_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4, PREC * 8);
	start = get_secv();
	itimes_mpf_sp[0][3] = MPFBiCG_sp(mpfx4, mpfa4_sp, mpfb4, reps4, aeps4, maxtimes);
	mpftime_sp[0][3] = get_secv() - start;
	free_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCG_sp\n", dim, (unsigned long)PREC * 16);
	mpfa5_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);
	//get_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5, PREC * 16);
	start = get_secv();
	itimes_mpf_sp[0][4] = MPFBiCG_sp(mpfx5, mpfa5_sp, mpfb5, reps5, aeps5, maxtimes);
	mpftime_sp[0][4] = get_secv() - start;
	free_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5);

	/* print */
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/

MPFCGS:

#ifndef SPARSEONLY
	/* run MPFCGS */
	/* load */
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans);
//	mpfa_sp = get_mpfproblem_sp(mpfa, mpfb, mpfans);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS\n", dim, (unsigned long)PREC);
	start = get_secv();
	itimes_mpf[1][0]  = MPFCGS(mpfx, mpfa, mpfb, reps, aeps, maxtimes);
	mpftime[1][0] = get_secv() - start;
	free_mpfproblem(mpfa, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS\n", dim, (unsigned long)PREC * 2);
	mpfa2 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf[1][1] = MPFCGS(mpfx2, mpfa2, mpfb2, reps2, aeps2, maxtimes);
	mpftime[1][1] = get_secv() - start;
	free_mpfproblem(mpfa2, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS\n", dim, (unsigned long)PREC * 4);
	mpfa3 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);

	start = get_secv();
	itimes_mpf[1][2] = MPFCGS(mpfx3, mpfa3, mpfb3, reps3, aeps3, maxtimes);
	mpftime[1][2] = get_secv() - start;
	free_mpfproblem(mpfa3, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS\n", dim, (unsigned long)PREC * 8);
	mpfa4 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);

	start = get_secv();
	itimes_mpf[1][3] = MPFCGS(mpfx4, mpfa4, mpfb4, reps4, aeps4, maxtimes);
	mpftime[1][3] = get_secv() - start;
	free_mpfproblem(mpfa4, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS\n", dim, (unsigned long)PREC * 16);
	mpfa5 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);

	start = get_secv();
	itimes_mpf[1][4] = MPFCGS(mpfx5, mpfa5, mpfb5, reps5, aeps5, maxtimes);
	mpftime[1][4] = get_secv() - start;
	free_mpfproblem(mpfa5, mpfb5, mpfans5);

	/* print */
//	print_mpfvector(mpfx);
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
#endif // SPARSEONLY

MPFCGS_sp:

	/* run MPFCGS */
//	get_mpfproblem(mpfa, mpfb, mpfans);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans, PREC);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	printf("\n-- dim = %ld, prec = %ld, MPFCGS_sp\n", dim, (unsigned long)PREC);
	mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);

#ifdef SPARSEONLY
	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);
#endif

	start = get_secv();
	itimes_mpf_sp[1][0]  = MPFCGS_sp(mpfx, mpfa_sp, mpfb, reps, aeps, maxtimes);
	mpftime_sp[1][0] = get_secv() - start;
	free_mpfproblem_sp(mpfa_sp, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS_sp\n", dim, (unsigned long)PREC * 2);
	mpfa2_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf_sp[1][1] = MPFCGS_sp(mpfx2, mpfa2_sp, mpfb2, reps2, aeps2, maxtimes);
	mpftime_sp[1][1] = get_secv() - start;
	free_mpfproblem_sp(mpfa2_sp, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS_sp\n", dim, (unsigned long)PREC * 4);
	mpfa3_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);
//	get_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3, PREC * 4);
	start = get_secv();
	itimes_mpf_sp[1][2] = MPFCGS_sp(mpfx3, mpfa3_sp, mpfb3, reps3, aeps3, maxtimes);
	mpftime_sp[1][2] = get_secv() - start;
	free_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS_sp\n", dim, (unsigned long)PREC * 8);
	mpfa4_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);
	//get_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4, PREC * 8);
	start = get_secv();
	itimes_mpf_sp[1][3] = MPFCGS_sp(mpfx4, mpfa4_sp, mpfb4, reps4, aeps4, maxtimes);
	mpftime_sp[1][3] = get_secv() - start;
	free_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFCGS_sp\n", dim, (unsigned long)PREC * 16);
	mpfa5_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);
	//get_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5, PREC * 16);
	start = get_secv();
	itimes_mpf_sp[1][4] = MPFCGS_sp(mpfx5, mpfa5_sp, mpfb5, reps5, aeps5, maxtimes);
	mpftime_sp[1][4] = get_secv() - start;
	free_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5);

	/* print */
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/

MPFBiCGSTAB:

#ifndef SPARSEONLY
	/* run MPFBiCGSTAB */
	/* load */
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans);
//	mpfa_sp = get_mpfproblem_sp(mpfa, mpfb, mpfans);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB\n", dim, (unsigned long)PREC);
	start = get_secv();
	itimes_mpf[2][0]  = MPFBiCGSTAB(mpfx, mpfa, mpfb, reps, aeps, maxtimes);
	mpftime[2][0] = get_secv() - start;
	free_mpfproblem(mpfa, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB\n", dim, (unsigned long)PREC * 2);
	mpfa2 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf[2][1] = MPFBiCGSTAB(mpfx2, mpfa2, mpfb2, reps2, aeps2, maxtimes);
	mpftime[2][1] = get_secv() - start;
	free_mpfproblem(mpfa2, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB\n", dim, (unsigned long)PREC * 4);
	mpfa3 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);

	start = get_secv();
	itimes_mpf[2][2] = MPFBiCGSTAB(mpfx3, mpfa3, mpfb3, reps3, aeps3, maxtimes);
	mpftime[2][2] = get_secv() - start;
	free_mpfproblem(mpfa3, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB\n", dim, (unsigned long)PREC * 8);
	mpfa4 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);

	start = get_secv();
	itimes_mpf[2][3] = MPFBiCGSTAB(mpfx4, mpfa4, mpfb4, reps4, aeps4, maxtimes);
	mpftime[2][3] = get_secv() - start;
	free_mpfproblem(mpfa4, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB\n", dim, (unsigned long)PREC * 16);
	mpfa5 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);

	start = get_secv();
	itimes_mpf[2][4] = MPFBiCGSTAB(mpfx5, mpfa5, mpfb5, reps5, aeps5, maxtimes);
	mpftime[2][4] = get_secv() - start;
	free_mpfproblem(mpfa5, mpfb5, mpfans5);

	/* print */
//	print_mpfvector(mpfx);
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
#endif // SPARSEONLY

MPFBiCGSTAB_sp:

	/* run MPFBiCGSTAB */
//	get_mpfproblem(mpfa, mpfb, mpfans);
//	get_mpfproblem_sp(mpfa_sp, mpfb, mpfans, PREC);
//	print_mpfrsmatrix(mpfa_sp);
//	return;

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB_sp\n", dim, (unsigned long)PREC);
	mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);

#ifdef SPARSEONLY
	mpfx  = init2_mpfvector(dim, PREC);
	mpfx2 = init2_mpfvector(dim, PREC * 2);
	mpfx3 = init2_mpfvector(dim, PREC * 4);
	mpfx4 = init2_mpfvector(dim, PREC * 8);
	mpfx5 = init2_mpfvector(dim, PREC * 16);
#endif // SPARSEONLY

	start = get_secv();
	itimes_mpf_sp[2][0]  = MPFBiCGSTAB_sp(mpfx, mpfa_sp, mpfb, reps, aeps, maxtimes);
	mpftime_sp[2][0] = get_secv() - start;
	free_mpfproblem_sp(mpfa_sp, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB_sp\n", dim, (unsigned long)PREC * 2);
	mpfa2_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf_sp[2][1] = MPFBiCGSTAB_sp(mpfx2, mpfa2_sp, mpfb2, reps2, aeps2, maxtimes);
	mpftime_sp[2][1] = get_secv() - start;
	free_mpfproblem_sp(mpfa2_sp, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB_sp\n", dim, (unsigned long)PREC * 4);
	mpfa3_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);
//	get_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3, PREC * 4);
	start = get_secv();
	itimes_mpf_sp[2][2] = MPFBiCGSTAB_sp(mpfx3, mpfa3_sp, mpfb3, reps3, aeps3, maxtimes);
	mpftime_sp[2][2] = get_secv() - start;
	free_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB_sp\n", dim, (unsigned long)PREC * 8);
	mpfa4_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);
	//get_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4, PREC * 8);
	start = get_secv();
	itimes_mpf_sp[2][3] = MPFBiCGSTAB_sp(mpfx4, mpfa4_sp, mpfb4, reps4, aeps4, maxtimes);
	mpftime_sp[2][3] = get_secv() - start;
	free_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFBiCGSTAB_sp\n", dim, (unsigned long)PREC * 16);
	mpfa5_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);
	//get_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5, PREC * 16);
	start = get_secv();
	itimes_mpf_sp[2][4] = MPFBiCGSTAB_sp(mpfx5, mpfa5_sp, mpfb5, reps5, aeps5, maxtimes);
	mpftime_sp[2][4] = get_secv() - start;
	free_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5);

	/* print */
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/

MPFGPBiCG:
#ifndef SPARSEONLY
	/* run MPFGPBiCG */
	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG\n", dim, (unsigned long)PREC);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);

	start = get_secv();
	itimes_mpf[3][0]  = MPFGPBiCG(mpfx, mpfa, mpfb, reps, aeps, maxtimes);
	mpftime[3][0] = get_secv() - start;
	free_mpfproblem(mpfa, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG\n", dim, (unsigned long)PREC * 2);
	mpfa2 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf[3][1] = MPFGPBiCG(mpfx2, mpfa2, mpfb2, reps2, aeps2, maxtimes);
	mpftime[3][1] = get_secv() - start;
	free_mpfproblem(mpfa2, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG\n", dim, (unsigned long)PREC * 4);
	mpfa3 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);

	start = get_secv();
	itimes_mpf[3][2] = MPFGPBiCG(mpfx3, mpfa3, mpfb3, reps3, aeps3, maxtimes);
	mpftime[3][2] = get_secv() - start;
	free_mpfproblem(mpfa3, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG\n", dim, (unsigned long)PREC * 8);
	mpfa4 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);

	start = get_secv();
	itimes_mpf[3][3] = MPFGPBiCG(mpfx4, mpfa4, mpfb4, reps4, aeps4, maxtimes);
	mpftime[3][3] = get_secv() - start;
	free_mpfproblem(mpfa4, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG\n", dim, (unsigned long)PREC * 16);
	mpfa5 = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);

	start = get_secv();
	itimes_mpf[3][4] = MPFGPBiCG(mpfx5, mpfa5, mpfb5, reps5, aeps5, maxtimes);
	mpftime[3][4] = get_secv() - start;
	free_mpfproblem(mpfa5, mpfb5, mpfans5);

	/* print */
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
#endif // SPARSEONLY
MPFGPBiCG_sp:

	/* run MPFGPBiCG */
	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG_sp\n", dim, (unsigned long)PREC);
	mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	mpfans = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC);
	mpfb = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC);

	start = get_secv();
	itimes_mpf_sp[3][0]  = MPFGPBiCG_sp(mpfx, mpfa_sp, mpfb, reps, aeps, maxtimes);
	mpftime_sp[3][0] = get_secv() - start;
	free_mpfproblem_sp(mpfa_sp, mpfb, mpfans);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG_sp\n", dim, (unsigned long)PREC * 2);
	mpfa2_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 2);
	mpfans2 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 2);
	mpfb2 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 2);

	start = get_secv();
	itimes_mpf_sp[3][1] = MPFGPBiCG_sp(mpfx2, mpfa2_sp, mpfb2, reps2, aeps2, maxtimes);
	mpftime_sp[3][1] = get_secv() - start;
	free_mpfproblem_sp(mpfa2_sp, mpfb2, mpfans2);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG_sp\n", dim, (unsigned long)PREC * 4);
	mpfa3_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 4);
	mpfans3 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 4);
	mpfb3 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 4);

	start = get_secv();
	itimes_mpf_sp[3][2] = MPFGPBiCG_sp(mpfx3, mpfa3_sp, mpfb3, reps3, aeps3, maxtimes);
	mpftime_sp[3][2] = get_secv() - start;
	free_mpfproblem_sp(mpfa3_sp, mpfb3, mpfans3);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG_sp\n", dim, (unsigned long)PREC * 8);
	mpfa4_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 8);
	mpfans4 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 8);
	mpfb4 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 8);

	start = get_secv();
	itimes_mpf_sp[3][3] = MPFGPBiCG_sp(mpfx4, mpfa4_sp, mpfb4, reps4, aeps4, maxtimes);
	mpftime_sp[3][3] = get_secv() - start;
	free_mpfproblem_sp(mpfa4_sp, mpfb4, mpfans4);

	printf("\n-- dim = %ld, prec = %ld, MPFGPBiCG_sp\n", dim, (unsigned long)PREC * 16);
	mpfa5_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC * 16);
	mpfans5 = init2_mpfvector_readMMcoordinate(VEC_ANS_MM_FILE, PREC * 16);
	mpfb5 = init2_mpfvector_readMMcoordinate(VEC_B_MM_FILE, PREC * 16);

	start = get_secv();
	itimes_mpf_sp[3][4] = MPFGPBiCG_sp(mpfx5, mpfa5_sp, mpfb5, reps5, aeps5, maxtimes);
	mpftime_sp[3][4] = get_secv() - start;
	free_mpfproblem_sp(mpfa5_sp, mpfb5, mpfans5);

	/* print */
/*	for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx2, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx3, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx4, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfx5, i));
		printf(" ");
		mpf_out_str(stdout, 10, (unsigned int)floor(log10(2.0) * PREC), get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
	/* end */
	mpf_clear(reps); mpf_clear(aeps);
	mpf_clear(reps2); mpf_clear(aeps2);
	mpf_clear(reps3); mpf_clear(aeps3);
	free_mpfvector(mpfx);
	free_mpfvector(mpfx2);
	free_mpfvector(mpfx3);
#endif // USE_GMP

	/* print itimes */
	printf("Iterative Times\n");
	printf("double(BiCG)       : %ld(%f)\n", itimes_d[0]   , dtime[0]   );
	printf("double(BiCG_sp)    : %ld(%f)\n", itimes_d_sp[0], dtime_sp[0]);
	printf("     Speedup Ratio : %f\n", dtime[0] / dtime_sp[0]);
	printf("double(CGS)        : %ld(%f)\n", itimes_d[1]   , dtime[1]   );
	printf("double(CGS_sp)     : %ld(%f)\n", itimes_d_sp[1], dtime_sp[1]);
	printf("     Speedup Ratio : %f\n", dtime[1] / dtime_sp[1]);
	printf("double(BiCGSTAB)   : %ld(%f)\n", itimes_d[2]   , dtime[2]   );
	printf("double(BiCGSTAB_sp): %ld(%f)\n", itimes_d_sp[2], dtime_sp[2]);
	printf("     Speedup Ratio : %f\n", dtime[2] / dtime_sp[2]);
	printf("double(GPBiCG)     : %ld(%f)\n", itimes_d[3]   , dtime[3]   );
	printf("double(GPBiCG_sp)  : %ld(%f)\n", itimes_d_sp[3], dtime_sp[3]);
	printf("     Speedup Ratio : %f\n", dtime[3] / dtime_sp[3]);

// DD
	printf("DD    (BiCG)       : %ld(%f)\n", itimes_dd[0]   , ddtime[0]   );
	printf("DD    (BiCG_sp)    : %ld(%f)\n", itimes_dd_sp[0], ddtime_sp[0]);
	printf("     Speedup Ratio : %f\n", ddtime[0] / ddtime_sp[0]);
	printf("DD    (CGS)        : %ld(%f)\n", itimes_dd[1]   , ddtime[1]   );
	printf("DD    (CGS_sp)     : %ld(%f)\n", itimes_dd_sp[1], ddtime_sp[1]);
	printf("     Speedup Ratio : %f\n", ddtime[1] / ddtime_sp[1]);
	printf("DD    (BiCGSTAB)   : %ld(%f)\n", itimes_dd[2]   , ddtime[2]   );
	printf("DD    (BiCGSTAB_sp): %ld(%f)\n", itimes_dd_sp[2], ddtime_sp[2]);
	printf("     Speedup Ratio : %f\n", ddtime[2] / ddtime_sp[2]);
	printf("DD    (GPBiCG)     : %ld(%f)\n", itimes_dd[3]   , ddtime[3]   );
	printf("DD    (GPBiCG_sp)  : %ld(%f)\n", itimes_dd_sp[3], ddtime_sp[3]);
	printf("     Speedup Ratio : %f\n", ddtime[3] / ddtime_sp[3]);

// TD
	printf("TD    (BiCG)       : %ld(%f)\n", itimes_td[0]   , tdtime[0]   );
	printf("TD    (BiCG_sp)    : %ld(%f)\n", itimes_td_sp[0], tdtime_sp[0]);
	printf("T    Speedup Ratio : %f\n", tdtime[0] / tdtime_sp[0]);
	printf("TD    (CGS)        : %ld(%f)\n", itimes_td[1]   , tdtime[1]   );
	printf("TD    (CGS_sp)     : %ld(%f)\n", itimes_td_sp[1], tdtime_sp[1]);
	printf("T    Speedup Ratio : %f\n", tdtime[1] / tdtime_sp[1]);
	printf("TD    (BiCGSTAB)   : %ld(%f)\n", itimes_td[2]   , tdtime[2]   );
	printf("TD    (BiCGSTAB_sp): %ld(%f)\n", itimes_td_sp[2], tdtime_sp[2]);
	printf("T    Speedup Ratio : %f\n", tdtime[2] / tdtime_sp[2]);
	printf("TD    (GPBiCG)     : %ld(%f)\n", itimes_td[3]   , tdtime[3]   );
	printf("TD    (GPBiCG_sp)  : %ld(%f)\n", itimes_td_sp[3], tdtime_sp[3]);
	printf("T    Speedup Ratio : %f\n", tdtime[3] / tdtime_sp[3]);

// QD
	printf("QD    (BiCG)       : %ld(%f)\n", itimes_qd[0]   , qdtime[0]   );
	printf("QD    (BiCG_sp)    : %ld(%f)\n", itimes_qd_sp[0], qdtime_sp[0]);
	printf("Q    Speedup Ratio : %f\n", qdtime[0] / qdtime_sp[0]);
	printf("QD    (CGS)        : %ld(%f)\n", itimes_qd[1]   , qdtime[1]   );
	printf("QD    (CGS_sp)     : %ld(%f)\n", itimes_qd_sp[1], qdtime_sp[1]);
	printf("Q    Speedup Ratio : %f\n", qdtime[1] / qdtime_sp[1]);
	printf("QD    (BiCGSTAB)   : %ld(%f)\n", itimes_qd[2]   , qdtime[2]   );
	printf("QD    (BiCGSTAB_sp): %ld(%f)\n", itimes_qd_sp[2], qdtime_sp[2]);
	printf("Q    Speedup Ratio : %f\n", qdtime[2] / qdtime_sp[2]);
	printf("QD    (GPBiCG)     : %ld(%f)\n", itimes_qd[3]   , qdtime[3]   );
	printf("QD    (GPBiCG_sp)  : %ld(%f)\n", itimes_qd_sp[3], qdtime_sp[3]);
	printf("Q    Speedup Ratio : %f\n", qdtime[3] / qdtime_sp[3]);

#ifdef USE_GMP
#ifdef USE_MPFR
#ifndef SPARSEONLY
	printf("mpfr_t(%ld,BiCG)       : %ld(%f)\n", (unsigned long)PREC, itimes_mpf   [0][0], mpftime   [0][0]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[0][0], mpftime_sp[0][0]);
	printf("       Speedup Ratio  : %f\n", mpftime[0][0] / mpftime_sp[0][0]);
	printf("mpfr_t(%ld,CGS)        : %ld(%f)\n", (unsigned long)PREC, itimes_mpf   [1][0], mpftime   [1][0]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[1][0], mpftime_sp[1][0]);
	printf("       Speedup Ratio  : %f\n", mpftime[1][0] / mpftime_sp[1][0]);
	printf("mpfr_t(%ld,BiCGSTAB)   : %ld(%f)\n", (unsigned long)PREC, itimes_mpf   [2][0], mpftime   [2][0]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[2][0], mpftime_sp[2][0]);
	printf("       Speedup Ratio  : %f\n", mpftime[2][0] / mpftime_sp[2][0]);
	printf("mpfr_t(%ld,GPBiCG)     : %ld(%f)\n", (unsigned long)PREC, itimes_mpf   [3][0], mpftime   [3][0]);
	printf("mpfr_t(%ld,GPBICG_sp)  : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[3][0], mpftime_sp[3][0]);
	printf("       Speedup Ratio  : %f\n", mpftime[3][0] / mpftime_sp[3][0]);
	printf("mpfr_t(%ld,BiCG)       : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf   [0][1], mpftime   [0][1]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf_sp[0][1], mpftime_sp[0][1]);
	printf("       Speedup Ratio  : %f\n", mpftime[0][1] / mpftime_sp[0][1]);
	printf("mpfr_t(%ld,CGS)        : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf   [1][1], mpftime   [1][1]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf_sp[1][1], mpftime_sp[1][1]);
	printf("       Speedup Ratio  : %f\n", mpftime[1][1] / mpftime_sp[1][1]);
	printf("mpfr_t(%ld,BiCGSTAB)   : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf   [2][1], mpftime   [2][1]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf_sp[2][1], mpftime_sp[2][1]);
	printf("       Speedup Ratio  : %f\n", mpftime[2][1] / mpftime_sp[2][1]);
	printf("mpfr_t(%ld,GPBiCG)     : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf   [3][1], mpftime   [3][1]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC * 2, itimes_mpf_sp[3][1], mpftime_sp[3][1]);
	printf("       Speedup Ratio  : %f\n", mpftime[3][1] / mpftime_sp[3][1]);
	printf("mpfr_t(%ld,BiCG)       : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf   [0][2], mpftime   [0][2]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf_sp[0][2], mpftime_sp[0][2]);
	printf("       Speedup Ratio  : %f\n", mpftime[0][2] / mpftime_sp[0][2]);
	printf("mpfr_t(%ld,CGS)        : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf   [1][2], mpftime   [1][2]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf_sp[1][2], mpftime_sp[1][2]);
	printf("       Speedup Ratio  : %f\n", mpftime[1][2] / mpftime_sp[1][2]);
	printf("mpfr_t(%ld,BiCGSTAB)   : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf   [2][2], mpftime   [2][2]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf_sp[2][2], mpftime_sp[2][2]);
	printf("       Speedup Ratio  : %f\n", mpftime[2][2] / mpftime_sp[2][2]);
	printf("mpfr_t(%ld,GPBiCG)     : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf   [3][2], mpftime   [3][2]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC * 4, itimes_mpf_sp[3][2], mpftime_sp[3][2]);
	printf("       Speedup Ratio  : %f\n", mpftime[3][2] / mpftime_sp[3][2]);
	printf("mpfr_t(%ld,BiCG)       : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf   [0][3], mpftime   [0][3]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf_sp[0][3], mpftime_sp[0][3]);
	printf("       Speedup Ratio  : %f\n", mpftime[0][3] / mpftime_sp[0][3]);
	printf("mpfr_t(%ld,CGS)        : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf   [1][3], mpftime   [1][3]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf_sp[1][3], mpftime_sp[1][3]);
	printf("       Speedup Ratio  : %f\n", mpftime[1][3] / mpftime_sp[1][3]);
	printf("mpfr_t(%ld,BiCGSTAB)   : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf   [2][3], mpftime   [2][3]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf_sp[2][3], mpftime_sp[2][3]);
	printf("       Speedup Ratio  : %f\n", mpftime[2][3] / mpftime_sp[2][3]);
	printf("mpfr_t(%ld,GPBiCG)     : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf   [3][3], mpftime   [3][3]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC * 8, itimes_mpf_sp[3][3], mpftime_sp[3][3]);
	printf("       Speedup Ratio  : %f\n", mpftime[3][3] / mpftime_sp[3][3]);
	printf("mpfr_t(%ld,BiCG)       : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf   [0][4], mpftime   [0][4]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf_sp[0][4], mpftime_sp[0][4]);
	printf("       Speedup Ratio  : %f\n", mpftime[0][4] / mpftime_sp[0][4]);
	printf("mpfr_t(%ld,CGS)        : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf   [1][4], mpftime   [1][4]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf_sp[1][4], mpftime_sp[1][4]);
	printf("       Speedup Ratio  : %f\n", mpftime[1][4] / mpftime_sp[1][4]);
	printf("mpfr_t(%ld,BiCGSTAB)   : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf   [2][4], mpftime   [2][4]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf_sp[2][4], mpftime_sp[2][4]);
	printf("       Speedup Ratio  : %f\n", mpftime[2][4] / mpftime_sp[2][4]);
	printf("mpfr_t(%ld,GPBiCG)     : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf   [3][4], mpftime   [3][4]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC * 16, itimes_mpf_sp[3][4], mpftime_sp[3][4]);
	printf("       Speedup Ratio  : %f\n", mpftime[3][4] / mpftime_sp[3][4]);
#else // SPARSEONLU
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[0][0], mpftime_sp[0][0]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[1][0], mpftime_sp[1][0]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[2][0], mpftime_sp[2][0]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC, itimes_mpf_sp[3][0], mpftime_sp[3][0]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC*2, itimes_mpf_sp[0][1], mpftime_sp[0][1]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC*2, itimes_mpf_sp[1][1], mpftime_sp[1][1]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC*2, itimes_mpf_sp[2][1], mpftime_sp[2][1]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC*2, itimes_mpf_sp[3][1], mpftime_sp[3][1]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC*4, itimes_mpf_sp[0][2], mpftime_sp[0][2]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC*4, itimes_mpf_sp[1][2], mpftime_sp[1][2]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC*4, itimes_mpf_sp[2][2], mpftime_sp[2][2]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC*4, itimes_mpf_sp[3][2], mpftime_sp[3][2]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC*8, itimes_mpf_sp[0][3], mpftime_sp[0][3]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC*8, itimes_mpf_sp[1][3], mpftime_sp[1][3]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC*8, itimes_mpf_sp[2][3], mpftime_sp[2][3]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC*8, itimes_mpf_sp[3][3], mpftime_sp[3][3]);
	printf("mpfr_t(%ld,BiCG_sp)    : %ld(%f)\n", (unsigned long)PREC*16, itimes_mpf_sp[0][4], mpftime_sp[0][4]);
	printf("mpfr_t(%ld,CGS_sp)     : %ld(%f)\n", (unsigned long)PREC*16, itimes_mpf_sp[1][4], mpftime_sp[1][4]);
	printf("mpfr_t(%ld,BiCGSTAB_sp): %ld(%f)\n", (unsigned long)PREC*16, itimes_mpf_sp[2][4], mpftime_sp[2][4]);
	printf("mpfr_t(%ld,GPBiCG_sp)  : %ld(%f)\n", (unsigned long)PREC*16, itimes_mpf_sp[3][4], mpftime_sp[3][4]);
#endif // SPARSEONLY
#else // USE_MPFR
	printf("mpf_t(128): %ld(%f)\n", itimes_mpf, mpftime[0]);
	printf("mpf_t(256): %ld(%f)\n", itimes_mpf2, mpftime[1]);
	printf("mpf_t(512): %ld(%f)\n", itimes_mpf3, mpftime[2]);
#endif // USE_MPFR
#endif // USE_GMP
}

