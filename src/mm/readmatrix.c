/********************************************************************************/
/*                                                                              */
/* readmatrix.c : Reading & Writing MatrixMarket Format Files                   */
/* Copyright (c) 2011 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* 2011-08-26 Version 0.0: make readmatrix.c                                    */
/* 2012-07-13 Version 0.1: sparse symmetric matrix is readable                  */
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
/* 
*   Matrix Market I/O example program extended(by T.Kouya)
*
*   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file.
*   and copies it to stdout.  This porgram does nothing useful, but
*   illustrates common usage of the Matrix Matrix I/O routines.
*   (See http://math.nist.gov/MatrixMarket for details.)
*
*   Usage:  a.out [filename] > output
*
*       
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

#include "bncmm.h"

/* double */

// read MatrixMarket format (coodinate or array type only) as vector if possible
DVector init_dvector_readMMcoordinate(const char *fname)
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
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
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
		fprintf(stderr, "Symmetrix matrix is not supported in current BNCpack!\n");
		return NULL;
	}

    /* find out size of sparse matrix .... */

	if(mm_is_coordinate(matcode))
	{
	    if ((ret_code = mm_read_mtx_crd_size(f, &row_dim, &col_dim, &nzero_total_num)) !=0)
	        return NULL;

	    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);
	}
	else if(mm_is_array(matcode))
	{
		if(ret_code = mm_read_mtx_array_size(f, &row_dim, &col_dim) != 0)
			return NULL;
		mm_write_mtx_array_size(stdout, row_dim, col_dim);
	}
	else
	{
		fprintf(stderr, "%s is not supported format\n", fname);
		return NULL;
	}

	// check size of vector
	if(col_dim >= 2)
	{
		fprintf(stderr, "%s is not a vector(rod_dim = %d, col_dim = %d)\n", fname, row_dim, col_dim);
		return NULL;
	}

	// initialize dmatrix
	dvec = init_dvector(row_dim);

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate vector area(dim= %d x %d\n", row_dim, col_dim);
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
		fprintf(stderr, "%s is not supported format\n", fname);
		free_dvector(dvec);
		return NULL;
	}

    if (f !=stdin) fclose(f);

	return dvec;
}

// read MatrixMarket format (coodinate type only)
DMatrix init_dmatrix_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	double val;
	DMatrix dmat;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }

	// symmetric matrix
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

	// print information about MM file
    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);

	// initialize dmatrix
	dmat = init_dmatrix(row_dim, col_dim);

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(dim= %d x %d\n", row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nzero_total_num; i++)
    {
        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
        row_index--;
        col_index--;

		// set
		set_dmatrix_ij(dmat, row_index, col_index, val);

		if(symmetric_flag == 1)
			set_dmatrix_ij(dmat, col_index, row_index, val);

//		if(i % 100 == 0)
//			fprintf(stderr, "reading %d -th row...\n", row_index);
	}

    if (f !=stdin) fclose(f);

	return dmat;
}

// compare mmcoordinate
int compare_mmcoordinate(const void *a, const void *b)
{
	mmcoordinate *mm_a, *mm_b;

	mm_a = (mmcoordinate *)a;
	mm_b = (mmcoordinate *)b;

	if(mm_a->row_index < mm_b->row_index)
	{
		return -1;
	}
	else if(mm_a->row_index == mm_b->row_index)
	{
		if(mm_a->col_index < mm_b->col_index)
			return -1;
		else if(mm_a->col_index == mm_b->col_index)
			return 0;
		else
			return 1;
	}
	else
		return 1;
}

// read MatrixMarket format (coodinate type only)
DRSMatrix init_drsmatrix_readMMcoordinate(const char *fname)
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
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "Sorry, this application does not support ");
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
		fprintf(stderr, "BNCpack supports only square sparse matrices!(%d, %d)\n", row_index, col_index);
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
		fprintf(stderr, "Error: cannot allocate buf or nzero_col_dim!\n");
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
				fprintf(stderr, "Error: cannot allocate buf or nzero_col_dim!\n");
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
		fprintf(stderr, "ERROR: cannot allocate matrix area(dim= %d x %d\n",row_dim, col_dim);
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

/* MPFR */

#ifdef USE_GMP

// read MatrixMarket format (coodinate type only) as vector if possible
MPFVector _init2_mpfvector_readMMcoordinate(const char *fname, unsigned long prec)
{
//	unsigned long prec;
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	char val_str[MAX_VAL_STR];
	MPFVector dvec;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
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
		fprintf(stderr, "Symmetrix matrix is not supported in current BNCpack!\n");
		return NULL;
	}

    /* find out size of sparse matrix .... */

	if(mm_is_coordinate(matcode))
	{
	    if ((ret_code = mm_read_mtx_crd_size(f, &row_dim, &col_dim, &nzero_total_num)) !=0)
	        return NULL;

	    mm_write_mtx_crd_size(stdout, row_dim, col_dim, nzero_total_num);
	}
	else if(mm_is_array(matcode))
	{
		if(ret_code = mm_read_mtx_array_size(f, &row_dim, &col_dim) != 0)
			return NULL;
		mm_write_mtx_array_size(stdout, row_dim, col_dim);
	}
	else
	{
		fprintf(stderr, "%s is not supported format\n", fname);
		return NULL;
	}

	// check size of vector
	if(col_dim >= 2)
	{
		fprintf(stderr, "%s is not a vector(rod_dim = %d, col_dim = %d)\n", fname, row_dim, col_dim);
		return NULL;
	}

	// initialize dmatrix
	dvec = init2_mpfvector(row_dim, prec);

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate vector area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

 	if(mm_is_coordinate(matcode))
 	{
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
	        row_index--;
	        col_index--;

			// set
			set_mpfvector_i_str(dvec, row_index, val_str, 10);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
	else if(mm_is_array(matcode))
	{
		nzero_total_num = row_dim * col_dim;
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        fscanf(f, "%s\n", val_str);

			// set
			set_mpfvector_i_str(dvec, i, val_str, 10);

		}
	}
	else
	{
		fprintf(stderr, "%s is not supported format\n", fname);
		free_mpfvector(dvec);
		return NULL;
	}

    if (f !=stdin) fclose(f);

	return dvec;
}

// read MatrixMarket format (coodinate type only) as vector if possible
MPFVector init2_mpfvector_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_mpfvector_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only) as vector if possible
MPFVector init_mpfvector_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_mpfvector_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
MPFMatrix _init2_mpfmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int row_index, col_index;
    int symmetric_flag;
    long int i, j, total_index;
	char val_str[MAX_VAL_STR];
	MPFMatrix dmat;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return NULL;
    }

	// symmetric matrix
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

	// initialize drsmatrix
	dmat = init2_mpfmatrix(row_dim, col_dim, prec);

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nzero_total_num; i++)
    {
        fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
        row_index--;
        col_index--;

		// set
		set_mpfmatrix_ij_str(dmat, row_index, col_index, val_str, 10);

		// set
		if(symmetric_flag == 1)
			set_mpfmatrix_ij_str(dmat, col_index, row_index, val_str, 10);

//		if(i % 100 == 0)
//			fprintf(stderr, "reading %d -th row...\n", row_index);
	}

    if (f !=stdin) fclose(f);

	return dmat;
}

// read MatrixMarket format (coodinate type only)
MPFMatrix init2_mpfmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_mpfmatrix_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
MPFMatrix init_mpfmatrix_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_mpfmatrix_readMMcoordinate(fname, prec);
}

// compare mmcoordinate
int compare_mmcoordinate_str(const void *a, const void *b)
{
	mmcoordinate_str *mm_a, *mm_b;

	mm_a = (mmcoordinate_str *)a;
	mm_b = (mmcoordinate_str *)b;

	if(mm_a->row_index < mm_b->row_index)
	{
		return -1;
	}
	else if(mm_a->row_index == mm_b->row_index)
	{
		if(mm_a->col_index < mm_b->col_index)
			return -1;
		else if(mm_a->col_index == mm_b->col_index)
			return 0;
		else
			return 1;
	}
	else
		return 1;
}

// read MatrixMarket format (coodinate type only)
MPFRSMatrix _init2_mpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec)
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
	char val_str[MAX_VAL_STR];
	MPFRSMatrix dsmat;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return NULL;
	}
    else    
    { 
        if ((f = fopen(fname, "r")) == NULL) 
            return NULL;
    }

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
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}

    if (!mm_is_coordinate(matcode))
    {
        fprintf(stderr, "Sorry, this application does not support ");
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
		fprintf(stderr, "BNCpack supports only square sparse matrices!(%d, %d)\n", row_index, col_index);
		return NULL;
	}

	// initialize buffers
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
			if(new_buf == NULL)
			{
				fprintf(stderr, "Error: cannot allocate buf or nzero_col_dim!\n");
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

	// sorting
//	qsort(buf, nzero_total_num, sizeof(mmcoordinate_str), compare_mmcoordinate_str);
	qsort(new_buf, new_nzero_total_num, sizeof(mmcoordinate_str), compare_mmcoordinate_str);
//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %s %d\n", buf[i].row_index, buf[i].col_index, buf[i].val_str, nzero_col_dim[buf[i].row_index]);


	// initialize drsmatrix
	dsmat = init2_mpfrsmatrix(row_dim, nzero_col_dim, new_nzero_total_num, prec);
	if(dsmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
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

	// free buf
	if(buf != NULL)
		free(buf);
	if(new_buf != NULL)
		free(new_buf);

	return dsmat;
}
// read MatrixMarket format (coodinate type only)
MPFRSMatrix init2_mpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_mpfrsmatrix_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
MPFRSMatrix init_mpfrsmatrix_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_mpfrsmatrix_readMMcoordinate(fname, prec);
}
#endif

