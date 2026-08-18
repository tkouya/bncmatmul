/********************************************************************************/
/* test_spmm.c:                                                                 */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncmatmul.h"
#include "get_secv.h" // get_secv
//#include "bnc.h"
//#include "bncsparse.h"
//#include "bncmm.h"

/* Downloaded file from UF Sparse Matrix Collection */
#define MATRIX_MM_FILE "cage3/cage3.mtx"

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
// read MatrixMarket format (coodinate type only)
DRSMatrix test_init_drsmatrix_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num, new_nzero_total_num, nzero_diagonal_num;
    int row_index, col_index;
    int symmetric_flag;
    long int *nzero_col_dim;
    long int i, j, total_index;
	mmcoordinate *buf, *new_buf;
	double val;
	DRSMatrix dsmat;

	if(fname == NULL)
	{
		fprintf(stderr, "(init_drsmatrix_readMMcoordinate) fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL)
		{
			fprintf(stderr, "(init_drsmatrix_readMMcoordinate) cannot read %s!!\n", fname);
            return NULL;
		}
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "(init_drsmatrix_readMMcoordinate) Could not process Matrix Market banner.\n");
        return NULL;
    }
	// print information about MM file
    mm_write_banner(stdout, matcode);

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
	if(mm_is_complex(matcode))
	{
		fprintf(stderr, "(init_drsmatrix_readMMcoordinate) Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "(init_drsmatrix_readMMcoordinate) Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }

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
		fprintf(stderr, "(init_drsmatrix_readMMcoordinate) BNCmatmul supports only square sparse matrices!(%d, %d)\n", row_index, col_index);
		return NULL;
	}

	// print information about MM file
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);

	// initialize buffers
	nzero_col_dim = (long int *)calloc(sizeof(long int), row_dim);
	//row_index = (long int *)calloc(sizeof(long int), nzero_total_num);
	//col_index = (long int *)calloc(sizeof(long int), nzero_total_num);
	//val = (double *)calloc(sizeof(double), nzero_total_num);
	//buf = (mmcoordinate *)calloc(sizeof(mmcoordinate), nzero_total_num);
	buf = (mmcoordinate *)malloc(sizeof(mmcoordinate) * nzero_total_num);

	if((nzero_col_dim == NULL) || (buf == NULL))
	{
		fprintf(stderr, "(init_drsmatrix_readMMcoordinate) Error: cannot allocate buf or nzero_col_dim!\n");
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	nzero_diagonal_num = 0;

	// fix! : 2012-07-13
	for(i = 0; i < row_dim; i++)
		nzero_col_dim[i] = 0;

    for (i = 0; i < nzero_total_num; i++)
    {
        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
        row_index--;
        col_index--;
//		dsmat->nzero_index[row_index][dsmat->nzero_col_dim[row_index]] = col_index;
		//printf("%d %d %d->%d\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
		nzero_col_dim[row_index]++;

		// set
		buf[i].row_index = row_index;
		buf[i].col_index = col_index;
		buf[i].val = val;

		// count diagonal elements
		if(row_index == col_index)
			nzero_diagonal_num++;

//		if(i % 100 == 0)
//			fprintf(stderr, "reading %d -th row...\n", row_index);
	}

    if (f !=stdin) fclose(f);

	// if symmetric matrix...
	if(symmetric_flag == 1)
	{
		new_nzero_total_num = (nzero_total_num - nzero_diagonal_num) * 2 + nzero_diagonal_num;
		if(new_nzero_total_num > nzero_total_num)
		{
			fprintf(stderr, "Reallocation...%d -> %d\n", nzero_total_num, new_nzero_total_num);
			new_buf = (mmcoordinate *)realloc(buf, sizeof(mmcoordinate) * new_nzero_total_num);
			if(new_buf == NULL)
			{
				fprintf(stderr, "(init_drsmatrix_readMMcoordinate) Error: cannot allocate buf or nzero_col_dim!\n");
				free(buf);
				return NULL;
			}

			total_index = nzero_total_num;

			// fix! : 2012-07-13
			for(i = 0; i < row_dim; i++)
				nzero_col_dim[i] = 0;

			for(i = 0; i < nzero_total_num; i++)
			{
				// set
				//printf(" set %d ->", i);

				// fix! : 2012-07-13
				nzero_col_dim[buf[i].row_index]++;

				new_buf[i].row_index = buf[i].row_index;
				new_buf[i].col_index = buf[i].col_index;
				new_buf[i].val = buf[i].val;

				// copy to opposite triangular area except diagonal element
				if(buf[i].row_index != buf[i].col_index)
				{
					// fix! : 2012-07-13
					nzero_col_dim[buf[i].col_index]++;

					new_buf[total_index].row_index = buf[i].col_index;
					new_buf[total_index].col_index = buf[i].row_index;
					new_buf[total_index].val = buf[i].val;
					total_index++;
				}
			}
			buf = new_buf;
			nzero_total_num = new_nzero_total_num;
		}
	}

	// sorting
	qsort(buf, nzero_total_num, sizeof(mmcoordinate), compare_mmcoordinate);
//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %10.3g %d\n", buf[i].row_index, buf[i].col_index, buf[i].val, nzero_col_dim[buf[i].row_index]);

	// initialize drsmatrix
	dsmat = init_drsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(dsmat == NULL)
	{
		fprintf(stderr, "(init_drsmatrix_readMMcoordinate) ERROR: cannot allocate matrix area(dim= %d x %d\n",row_dim, col_dim);
		return NULL;
	}

	// copy buffers to drsmatrix
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < nzero_col_dim[i]; j++)
		{
			dsmat->element[total_index] = buf[total_index].val;
			dsmat->nzero_index[i][j] = buf[total_index].col_index;
			total_index++;
		}
	}

	// free buf
	free(buf);

	return dsmat;
}


int main(void)
{
	long int dim, total_index;
	long int maxtimes = MAXTIMES;
	DMatrix da;
	DRSMatrix da_sp;
    DVector dvec;
    DDMatrix dda;
    DDRSMatrix dda_sp;
    DDVector ddvec;
    TDMatrix tda;
    TDRSMatrix tda_sp;
    TDVector tdvec;
    QDMatrix qda;
    QDRSMatrix qda_sp;
    QDVector qdvec;
    double ddtmp[DDSIZE], tdtmp[TDSIZE], qdtmp[QDSIZE];
	double start, dtime[5], dtime_sp[5];
	long int itimes_d[5], itimes_d_sp[5];
	long int i, j;

#ifdef USE_GMP
	MPFMatrix mpfa;
	MPFMatrix mpfa2;
	MPFMatrix mpfa3;
	MPFMatrix mpfa4;
	MPFMatrix mpfa5;

	MPFRSMatrix mpfa_sp;
	MPFRSMatrix mpfa2_sp;
	MPFRSMatrix mpfa3_sp;
	MPFRSMatrix mpfa4_sp;
	MPFRSMatrix mpfa5_sp;
#endif // USE_GMP

//goto MPFR;

/* Double */
	/* initialize & get problem */

    printf("Reading %s...\n", MATRIX_MM_FILE);
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    print_drsmatrix(da_sp);
	free_drsmatrix(da_sp);

	/* run DBiCG_sp */
    printf("Reading %s...\n", MATRIX_MM_FILE);
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
    print_dmatrix(da);

	free_dmatrix(da);

MPFR:
#ifdef USE_GMP
/* MPF */
	set_bnc_default_prec(PREC);
	/* load */
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    print_mpfrsmatrix(mpfa_sp);
	free_mpfrsmatrix(mpfa_sp);

    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
    print_mpfmatrix(mpfa);
    free_mpfmatrix(mpfa);

    // MPFRSMatrix -> DDRSMatrix
    mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    printf("row_dim, nzero_total_num = %ld, %ld\n", mpfa_sp->row_dim, mpfa_sp->nzero_total_num);
/*   dda_sp = init_ddrsmatrix(mpfa_sp->row_dim, mpfa_sp->nzero_col_dim, mpfa_sp->nzero_total_num);
    printf("init_ddrsmatrix!\n");

    // MPFR -> DD
    total_index = 0;
	for(i = 0; i < dda_sp->row_dim; i++)
	{
		for(j = 0; j < dda_sp->nzero_col_dim[i]; j++)
		{
            mpf_get_dd(ddtmp, mpfa_sp->element[total_index]);
            dda_sp->element[0][total_index] = ddtmp[0];
            dda_sp->element[1][total_index] = ddtmp[1];
            dda_sp->nzero_index[i][j] = mpfa_sp->nzero_index[i][j];
			total_index++;
		}
	}
*/
    dda_sp = init_set_ddrsmatrix_mpfrsmatrix(mpfa_sp);
    print_ddrsmatrix(dda_sp);

    free_ddrsmatrix(dda_sp);
    free_mpfrsmatrix(mpfa_sp);

#endif // USE_GMP
}

