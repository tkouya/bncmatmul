/* read and write complex matrix and vector */
#include "bncmm.h"
#include "bncmm_c.h"

/* double */

// read MatrixMarket format (coodinate or array type only) as vector if possible
CDVector init_cdvector_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	double val, val_im;
	CDVector dvec;

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
/*	if(mm_is_complex(matcode))
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
		fprintf(stderr, "%s is not a vector(row_dim = %d, col_dim = %d)\n", fname, row_dim, col_dim);
		return NULL;
	}

	// initialize dmatrix
	dvec = init_cdvector(row_dim);

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
		// Real Number
		if(mm_is_real(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
		        row_index--;
		        col_index--;

				// set
				set_cdvector_i(dvec, row_index, val + 0.0 * I);

		//		if(i % 100 == 0)
		//			fprintf(stderr, "reading %d -th row...\n", row_index);
			}
		}
		// Complex Number
		else if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val, &val_im);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
		        row_index--;
		        col_index--;

				// set
				set_cdvector_i(dvec, row_index, val + val_im * I);

		//		if(i % 100 == 0)
		//			fprintf(stderr, "reading %d -th row...\n", row_index);
			}
		}
	}
	else if(mm_is_array(matcode))
	{
		nzero_total_num = row_dim * col_dim;

		// Real Number
		if(mm_is_real(matcode))
		{
	 	   for (i = 0; i < nzero_total_num; i++)
	 	   {

	 	       fscanf(f, "%lg\n", &val);

				// set
				set_cdvector_i(dvec, i, val + 0.0 * I);
			}
		}
		// Complex Number
		else if(mm_is_complex(matcode))
		{
	 	   for (i = 0; i < nzero_total_num; i++)
	 	   {

	 	       fscanf(f, "%lg %lg\n", &val, &val_im);

				// set
				set_cdvector_i(dvec, i, val + val_im * I);
			}
		}
	}
	else
	{
		fprintf(stderr, "%s is not supported format\n", fname);
		free_cdvector(dvec);
		return NULL;
	}

    if (f !=stdin) fclose(f);

	return dvec;
}

// read MatrixMarket format (coodinate type only)
CDMatrix init_cdmatrix_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	double val, val_im;
	CDMatrix dmat;

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
/*	if(mm_is_complex(matcode))
	{
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}
*/
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
	dmat = init_cdmatrix(row_dim, col_dim);

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(dim= %d x %d\n", row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	// Real Matrix
	if(mm_is_real(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
	        row_index--;
	        col_index--;
	
			// set
			set_cdmatrix_ij(dmat, row_index, col_index, val + 0.0 * I);
	
			if(symmetric_flag == 1)
				set_cdmatrix_ij(dmat, col_index, row_index, val + 0.0 * I);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
	// Comlex Matrix
	else if(mm_is_complex(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val, &val_im);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
	        row_index--;
	        col_index--;
	
			// set
			set_cdmatrix_ij(dmat, row_index, col_index, val + val_im * I);
	
			if(symmetric_flag == 1)
				set_cdmatrix_ij(dmat, col_index, row_index, val + val_im * I);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
    if (f !=stdin) fclose(f);

	return dmat;
}

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cdvector(const char *fname, CDVector dvec)
{
    MM_typecode matcode;
    FILE *f;
    int i;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: vector is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dvec->dim, 1, dvec->dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dvec->dim, 1, dvec->dim);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%d %d %25.17g %25.17g\n", i + 1, 1, creal(get_cdvector_i(dvec, i)), cimag(get_cdvector_i(dvec, i)));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer vector as MatrixMarket format (array type) 
int writeMMarray_cdvector(const char *fname, CDVector dvec)
{
    MM_typecode matcode;
    FILE *f;
    int i;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: vector is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_array_size(stdout, dvec->dim, 1);

    mm_write_banner(f, matcode); 
    mm_write_mtx_array_size(f, dvec->dim, 1);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%25.17g %25.17g\n", creal(get_cdvector_i(dvec, i)), cimag(get_cdvector_i(dvec, i)));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cdmatrix(const char *fname, CDMatrix dmat)
{
    MM_typecode matcode;
    FILE *f;
    int i, j;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: matrix is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    for (i = 0; i < dmat->col_dim; i++)
	{
		for(j = 0; j < dmat->row_dim; j++)
			fprintf(f, "%d %d %25.17g %25.17g\n", j + 1, i + 1, creal(get_cdmatrix_ij(dmat, j, i)), cimag(get_cdmatrix_ij(dmat, j, i)));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

/* MPF */
#ifdef USE_GMP
// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector _init2_cmpfvector_readMMcoordinate(const char *fname, unsigned long prec)
{
//	unsigned long prec;
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int i, row_index, col_index;
    int symmetric_flag;
	char val_str[MAX_VAL_STR], val_im_str[MAX_VAL_STR];
	CMPFVector dvec;

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
/*	if(mm_is_complex(matcode))
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
		fprintf(stderr, "%s is not a vector(row_dim = %d, col_dim = %d)\n", fname, row_dim, col_dim);
		return NULL;
	}

	// initialize dmatrix
	dvec = init2_cmpfvector(row_dim, prec);

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
 		if(mm_is_real(matcode))
 		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
		        row_index--;
		        col_index--;

				// set
				set_cmpfvector_i_str(dvec, row_index, val_str, "0", 10);
	
		//		if(i % 100 == 0)
		//			fprintf(stderr, "reading %d -th row...\n", row_index);
			}
		}
		else if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%d %d %s %s\n", &row_index, &col_index, val_str, val_im_str);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
		        row_index--;
		        col_index--;

				// set
				set_cmpfvector_i_str(dvec, row_index, val_str, val_im_str, 10);
	
		//		if(i % 100 == 0)
		//			fprintf(stderr, "reading %d -th row...%s %s\n", row_index, val_str, val_im_str);
			}
		}
	}
	else if(mm_is_array(matcode))
	{
		nzero_total_num = row_dim * col_dim;
		if(mm_is_real(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%s\n", val_str);

				// set
				set_cmpfvector_i_str(dvec, i, val_str, "0", 10);
			}
		}
		if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        fscanf(f, "%s %s\n", val_str, val_im_str);

				// set
				set_cmpfvector_i_str(dvec, i, val_str, val_im_str, 10);
			}
		}
	}
	else
	{
		fprintf(stderr, "%s is not supported format\n", fname);
		free_cmpfvector(dvec);
		return NULL;
	}

    if (f !=stdin) fclose(f);

	return dvec;
}

// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector init2_cmpfvector_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_cmpfvector_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector init_cmpfvector_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_cmpfvector_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
CMPFMatrix _init2_cmpfmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num;
    int row_index, col_index;
    int symmetric_flag;
    long int i, j, total_index;
	char val_str[MAX_VAL_STR], val_im_str[MAX_VAL_STR];
	CMPFMatrix dmat;

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
/*	if(mm_is_complex(matcode))
	{
		fprintf(stderr, "Complex number is not supported in current BNCpack!\n");
		return NULL;
	}
*/
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
	dmat = init2_cmpfmatrix(row_dim, col_dim, prec);

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
		return NULL;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	if(mm_is_real(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
		{
			fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
	//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
			row_index--;
			col_index--;

			// set
			set_cmpfmatrix_ij_str(dmat, row_index, col_index, val_str, "0", 10);

			// set
			if(symmetric_flag == 1)
				set_cmpfmatrix_ij_str(dmat, col_index, row_index, val_str, "0", 10);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
	else if(mm_is_complex(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
		{
 	       fscanf(f, "%d %d %s %s\n", &row_index, &col_index, val_str, val_im_str);
	//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
	        row_index--;
	        col_index--;

			// set
			set_cmpfmatrix_ij_str(dmat, row_index, col_index, val_str, val_im_str, 10);

		// set
			if(symmetric_flag == 1)
				set_cmpfmatrix_ij_str(dmat, col_index, row_index, val_str, val_im_str, 10);

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
	}
    if (f !=stdin) fclose(f);

	return dmat;
}

// read MatrixMarket format (coodinate type only)
CMPFMatrix init2_cmpfmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_cmpfmatrix_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
CMPFMatrix init_cmpfmatrix_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_cmpfmatrix_readMMcoordinate(fname, prec);
}

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cmpfvector(const char *fname, CMPFVector dvec)
{
    MM_typecode matcode;
    FILE *f;
    int i;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: vector is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dvec->dim, 1, dvec->dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dvec->dim, 1, dvec->dim);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%d %d ", i + 1, 1);
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, "\n");
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer vector as MatrixMarket format (array type) 
int writeMMarray_cmpfvector(const char *fname, CMPFVector dvec)
{
    MM_typecode matcode;
    FILE *f;
    int i;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dvec == NULL)
	{
		fprintf(stderr, "ERROR: vector is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_array(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_array_size(stdout, dvec->dim, 1);

    mm_write_banner(f, matcode); 
    mm_write_mtx_array_size(f, dvec->dim, 1);

    for (i = 0; i < dvec->dim; i++)
	{
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, "\n");
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cmpfmatrix(const char *fname, CMPFMatrix dmat)
{
    MM_typecode matcode;
    FILE *f;
    int i, j;

	if(fname == NULL)
	{
		fprintf(stderr, "fname is not set\n");
		return ERROR;
	}
    else    
    { 
        if ((f = fopen(fname, "w")) == NULL) 
            return ERROR;
    }

	if(dmat == NULL)
	{
		fprintf(stderr, "ERROR: matrix is not set\n");
		return ERROR;
	}

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_complex(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    for (i = 0; i < dmat->col_dim; i++)
	{
		for(j = 0; j < dmat->row_dim; j++)
		{
			fprintf(f, "%d %d ", j + 1, i + 1);
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfmatrix_ij(dmat, j, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfmatrix_ij(dmat, j, i)));
			fprintf(f, "\n");
		}
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}
#endif

