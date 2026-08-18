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
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
		if((ret_code = mm_read_mtx_array_size(f, &row_dim, &col_dim)) != 0)
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
		        ret_fscanf = fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
				if(ret_fscanf == 3)
				{
			//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
					row_index--;
					col_index--;

					// set
					set_cdvector_i(dvec, row_index, val + 0.0 * I);

			//		if(i % 100 == 0)
			//			fprintf(stderr, "reading %d -th row...\n", row_index);
				}
			}
		}
		// Complex Number
		else if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        ret_fscanf = fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val, &val_im);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
				if(ret_fscanf == 4)
				{
					row_index--;
					col_index--;

					// set
					set_cdvector_i(dvec, row_index, val + val_im * I);

			//		if(i % 100 == 0)
			//			fprintf(stderr, "reading %d -th row...\n", row_index);
				}
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

				ret_fscanf = fscanf(f, "%lg\n", &val);
				if(ret_fscanf == 1)
				{
					// set
					set_cdvector_i(dvec, i, val + 0.0 * I);
				}
			}
		}
		// Complex Number
		else if(mm_is_complex(matcode))
		{
	 	   for (i = 0; i < nzero_total_num; i++)
	 	   {

				ret_fscanf = fscanf(f, "%lg %lg\n", &val, &val_im);
				if(ret_fscanf == 2)
				{
					// set
					set_cdvector_i(dvec, i, val + val_im * I);
				}
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
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
	        ret_fscanf = fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
			if(ret_fscanf == 3)
			{
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
	}
	// Comlex Matrix
	else if(mm_is_complex(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
	    {
	        ret_fscanf = fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val, &val_im);
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
			if(ret_fscanf == 4)
			{
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


// compare mmcoordinate
int compare_mmcoordinate_c(const void *a, const void *b)
{
	mmcoordinate_c *mm_a, *mm_b;

	mm_a = (mmcoordinate_c *)a;
	mm_b = (mmcoordinate_c *)b;

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
#if 0
CDRSMatrix init_cdrsmatrix_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num, new_nzero_total_num, nzero_diagonal_num;
    int row_index, col_index;
    int symmetric_flag;
    long int *nzero_col_dim;
    long int i, j, total_index, buf_total_index;
	mmcoordinate_c *buf, *new_buf;
	double val_re, val_im;
	CDRSMatrix cdsmat;

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
	/* if(mm_is_complex(matcode))
	{
		fprintf(stderr, "Real number is not supported!\n");
		return NULL;
	}
	*/

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
	buf = (mmcoordinate_c *)calloc(nzero_total_num, sizeof(mmcoordinate_c));

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
 		if(mm_is_real(matcode))
 		{
	        //fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
        	fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val_re);
			val_im = 0.0; // The imaginary part is zero.
		}
		else if(mm_is_complex(matcode))
		{
	        //fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
        	fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val_re, &val_im);
		}

//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
        row_index--;
        col_index--;
//		dsmat->nzero_index[row_index][dsmat->nzero_col_dim[row_index]] = col_index;
		//printf("%d %d %d->%d\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
		nzero_col_dim[row_index]++;

		// set
		buf[i].row_index = row_index;
		buf[i].col_index = col_index;
		buf[i].val = val_re + val_im * I;
	
		// count diagonal elements
		if(row_index == col_index)
			nzero_diagonal_num++;

//		if(i % 100 == 0)
//			fprintf(stderr, "reading %d -th row...\n", row_index);
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
			//new_buf = (mmcoordinate_c *)realloc(buf, sizeof(mmcoordinate_c) * new_nzero_total_num);
			new_buf = (mmcoordinate_c_str *)malloc(sizeof(mmcoordinate_c_str) * new_nzero_total_num);
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

			//printf("embedding...\n");
			for(i = 0; i < nzero_total_num; i++)
			{
				// set
				//printf(" set %d ->", i);fflush(stdout);

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
			//buf = new_buf;
			nzero_total_num = new_nzero_total_num;
		}
	}

	// sorting
	//fprintf(stderr, "QSORTING...\n");fflush(stdout);
	//qsort(buf, nzero_total_num, sizeof(mmcoordinate_c), compare_mmcoordinate_c);
	qsort(new_buf, nzero_total_num, sizeof(mmcoordinate_c), compare_mmcoordinate_c);
//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %10.3g %d\n", buf[i].row_index, buf[i].col_index, buf[i].val, nzero_col_dim[buf[i].row_index]);

	// initialize drsmatrix
	//printf("init_cdrsmatrix...\n");
	cdsmat = init_cdrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(cdsmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(dim= %d x %d\n",row_dim, col_dim);
		return NULL;
	}

	// copy buffers to drsmatrix
	buf_total_index = 0;
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < cdsmat->re->nzero_col_dim[i]; j++)
		{

			cdsmat->re->element[total_index] = creal(new_buf[buf_total_index].val);
			cdsmat->im->element[total_index] = cimag(new_buf[buf_total_index].val);
			cdsmat->re->nzero_index[i][j] = new_buf[buf_total_index].col_index;
			cdsmat->im->nzero_index[i][j] = new_buf[buf_total_index].col_index;

			buf_total_index++;
			total_index++;
		}
		// Fix! 2024-08-05(Mon) T.Kouya
		for(j = cdsmat->re->nzero_col_dim[i]; j < cdsmat->re->real_nzero_col_dim[i]; j++)
		{
			cdsmat->re->element[total_index] = 0.0;
			cdsmat->im->element[total_index] = 0.0;
			total_index++;
		}
	}

	// free buf
	free(buf);
	if(new_buf != NULL) free(new_buf);

	return cdsmat;
}
#endif // 0
CDRSMatrix init_cdrsmatrix_readMMcoordinate(const char *fname)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num, new_nzero_total_num, nzero_diagonal_num;
    int row_index, col_index;
    int symmetric_flag;
    long int *nzero_col_dim;
    long int i, j, total_index, buf_total_index;
	mmcoordinate_c *buf, *new_buf;
	double val_re, val_im;
	CDRSMatrix cdsmat;
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
	/* if(mm_is_complex(matcode))
	{
		fprintf(stderr, "Real number is not supported!\n");
		return NULL;
	}
	*/

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
	buf = (mmcoordinate_c *)calloc(nzero_total_num, sizeof(mmcoordinate_c));

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
 		if(mm_is_real(matcode))
 		{
	        //fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
        	ret_fscanf = fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val_re);
			val_im = 0.0; // The imaginary part is zero.
			if(ret_fscanf != 3) ret_fscanf = -1; // error!
		}
		else if(mm_is_complex(matcode))
		{
	        //fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
        	ret_fscanf = fscanf(f, "%d %d %lg %lg\n", &row_index, &col_index, &val_re, &val_im);
			if(ret_fscanf != 4) ret_fscanf = -1; // error!
		}

		if(ret_fscanf != -1)
		{
	//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
			row_index--;
			col_index--;
	//		dsmat->nzero_index[row_index][dsmat->nzero_col_dim[row_index]] = col_index;
			//printf("%d %d %d->%d\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
			nzero_col_dim[row_index]++;

			// set
			buf[i].row_index = row_index;
			buf[i].col_index = col_index;
			buf[i].val = val_re + val_im * I;
			//printf("buf[%ld] = %25.17e + %25.17e * I\n", i, creal(buf[i].val), cimag(buf[i].val));
		
			// count diagonal elements
			if(row_index == col_index)
				nzero_diagonal_num++;

	//		if(i % 100 == 0)
	//			fprintf(stderr, "reading %d -th row...\n", row_index);
		}
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
			//new_buf = (mmcoordinate_c *)realloc(buf, sizeof(mmcoordinate_c) * new_nzero_total_num);
			new_buf = (mmcoordinate_c *)malloc(sizeof(mmcoordinate_c) * new_nzero_total_num);
			if(new_buf == NULL)
			{
				fprintf(stderr, "Error: cannot allocate buf or nzero_col_dim!\n");
				free(buf);
				return NULL;
			}
			//free(buf); // free old buf

			total_index = nzero_total_num;

			// fix! : 2012-07-13
			for(i = 0; i < row_dim; i++)
				nzero_col_dim[i] = 0;

			printf("embedding...\n");
			for(i = 0; i < nzero_total_num; i++)
			{
				// set
				//printf(" set %d ->", i);fflush(stdout);

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
			//buf = new_buf;
			nzero_total_num = new_nzero_total_num;
		}
	}

	// sorting
	//fprintf(stderr, "QSORTING...\n");fflush(stdout);
	//qsort(buf, nzero_total_num, sizeof(mmcoordinate_c), compare_mmcoordinate_c);
	//qsort(new_buf, nzero_total_num, sizeof(mmcoordinate_c), test_compare_mmcoordinate_c);
	qsort(new_buf, nzero_total_num, sizeof(mmcoordinate_c), compare_mmcoordinate_c);
	//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %10.3g %d\n", buf[i].row_index, buf[i].col_index, buf[i].val, nzero_col_dim[buf[i].row_index]);

	// initialize drsmatrix
	//printf("init_cdrsmatrix...\n");
	cdsmat = init_cdrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(cdsmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(dim= %d x %d\n",row_dim, col_dim);
		return NULL;
	}

	// copy buffers to drsmatrix
	buf_total_index = 0;
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < cdsmat->re->nzero_col_dim[i]; j++)
		{
			//cdsmat->re->element[total_index] = buf[buf_total_index].val;
			//cdsmat->re->nzero_index[i][j] = buf[buf_total_index].col_index;
			cdsmat->re->element[total_index] = creal(new_buf[buf_total_index].val);
			cdsmat->im->element[total_index] = cimag(new_buf[buf_total_index].val);
			cdsmat->re->nzero_index[i][j] = new_buf[buf_total_index].col_index;
			cdsmat->im->nzero_index[i][j] = new_buf[buf_total_index].col_index; // fix!

			buf_total_index++;
			total_index++;
		}
		// Fix! 2024-08-05(Mon) T.Kouya
		for(j = cdsmat->re->nzero_col_dim[i]; j < cdsmat->re->real_nzero_col_dim[i]; j++)
		{
			cdsmat->re->element[total_index] = 0.0;
			cdsmat->im->element[total_index] = 0.0;
			total_index++;
		}
	}

	// free buf
	//free(buf);
	if(symmetric_flag == 1) free(new_buf);

	return cdsmat;
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
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
		if((ret_code = mm_read_mtx_array_size(f, &row_dim, &col_dim)) != 0)
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
		        ret_fscanf = fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
				if(ret_fscanf == 3)
				{
					row_index--;
					col_index--;

					// set
					set_cmpfvector_i_str(dvec, row_index, val_str, "0", 10);
		
			//		if(i % 100 == 0)
			//			fprintf(stderr, "reading %d -th row...\n", row_index);
				}
			}
		}
		else if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        ret_fscanf = fscanf(f, "%d %d %s %s\n", &row_index, &col_index, val_str, val_im_str);
		//        gmp_fscanf(f, "%d %d %Fg\n", &I[i], &J[i], &val[i]);
				if(ret_fscanf == 4)
				{
					row_index--;
					col_index--;

					// set
					set_cmpfvector_i_str(dvec, row_index, val_str, val_im_str, 10);
		
			//		if(i % 100 == 0)
			//			fprintf(stderr, "reading %d -th row...%s %s\n", row_index, val_str, val_im_str);
				}
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
		        ret_fscanf = fscanf(f, "%s\n", val_str);
				if(ret_fscanf == 1)
				{
					// set
					set_cmpfvector_i_str(dvec, i, val_str, "0", 10);
				}
			}
		}
		if(mm_is_complex(matcode))
		{
		    for (i = 0; i < nzero_total_num; i++)
		    {
		        ret_fscanf = fscanf(f, "%s %s\n", val_str, val_im_str);
				if(ret_fscanf == 1)
				{
					// set
					set_cmpfvector_i_str(dvec, i, val_str, val_im_str, 10);
				}
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
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
			ret_fscanf = fscanf(f, "%d %d %s\n", &row_index, &col_index, val_str);
	//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
			if(ret_fscanf == 3)
			{
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
	}
	else if(mm_is_complex(matcode))
	{
	    for (i = 0; i < nzero_total_num; i++)
		{
 	       ret_fscanf = fscanf(f, "%d %d %s %s\n", &row_index, &col_index, val_str, val_im_str);
	//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
	//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
			if(ret_fscanf == 4)
			{
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
#ifdef USE_MPFCMPLX
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfvector_i(dvec, i)));
#else // USE_MPFCMPLX
			mpf_out_str(f, 10, 0, mpc_realref(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, mpc_imagref(get_cmpfvector_i(dvec, i)));
#endif // USE_MPFCMPLX
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
#ifdef USE_MPFCMPLX
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfvector_i(dvec, i)));
#else // USE_MPFCMPLX
			mpf_out_str(f, 10, 0, mpc_realref(get_cmpfvector_i(dvec, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, mpc_imagref(get_cmpfvector_i(dvec, i)));
#endif // USE_MPFCMPLX
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
#ifdef USE_MPFCMPLX
			mpf_out_str(f, 10, 0, getp_real_mpfcmplx(get_cmpfmatrix_ij(dmat, j, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, getp_image_mpfcmplx(get_cmpfmatrix_ij(dmat, j, i)));
#else // USE_MPFCMPLX
			mpf_out_str(f, 10, 0, mpc_realref(get_cmpfmatrix_ij(dmat, j, i)));
			fprintf(f, " ");
			mpf_out_str(f, 10, 0, mpc_imagref(get_cmpfmatrix_ij(dmat, j, i)));
#endif // USE_MPFCMPLX
			fprintf(f, "\n");
		}
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}


// compare mmcoordinate
int compare_mmcoordinate_c_str(const void *a, const void *b)
{
	mmcoordinate_c_str *mm_a, *mm_b;

	mm_a = (mmcoordinate_c_str *)a;
	mm_b = (mmcoordinate_c_str *)b;

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
CMPFRSMatrix _init2_cmpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	int ret_code;
    MM_typecode matcode;
    FILE *f;
    int row_dim, col_dim, nzero_total_num, new_nzero_total_num, nzero_diagonal_num;
    int row_index, col_index;
    int symmetric_flag;
    long int *nzero_col_dim;
    long int i, j, total_index, buf_total_index;
	int flag_new_buf = 0; // 
	mmcoordinate_c_str *buf, *new_buf;
	char val_re_str[MAX_VAL_STR], val_im_str[MAX_VAL_STR];
	CMPFRSMatrix cdsmat;
	mpf_t tmp[2];
	int ret_fscanf; // 2025-07-09 (Wed) T.Kouya

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
	/*if(mm_is_complex(matcode))
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
	buf = (mmcoordinate_c_str *)calloc(nzero_total_num, sizeof(mmcoordinate_c_str));

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
 		if(mm_is_real(matcode))
 		{
	        ret_fscanf = fscanf(f, "%d %d %s\n", &row_index, &col_index, val_re_str);
			//val_str_im = "0.0"; // The imaginary part is zero.
			strcpy(val_im_str, "0.0");
			if(ret_fscanf != 3) ret_fscanf = -1; // error
		}
		else if(mm_is_complex(matcode))
		{
	        //fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
        	ret_fscanf = fscanf(f, "%d %d %s %s\n", &row_index, &col_index, val_re_str, val_im_str);
			if(ret_fscanf != 4) ret_fscanf = -1; // error
		}
		//        fscanf(f, "%d %d %lg\n", &row_index, &col_index, &val);
//        gmp_fscanf(f, "%d %d %Fg\n", &row_index, &col_index, val);
		if(ret_fscanf != -1)
		{
			row_index--;
			col_index--;
	//		dsmat->nzero_index[row_index][dsmat->nzero_col_dim[row_index]] = col_index;
			//printf("%d %d %d->%d\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
			nzero_col_dim[row_index]++;

			// set
			buf[i].row_index = row_index;
			buf[i].col_index = col_index;
			strcpy(buf[i].val_re_str, val_re_str);
			strcpy(buf[i].val_im_str, val_im_str);
	//		printf("val_str -> %s-->%s--\n", val_str, buf[i].val_str);

			// count nonzero diagonal elements
			if(row_index == col_index)
				nzero_diagonal_num++;

			/* if(i % 100 == 0)
				fprintf(stderr, "reading %d -th row...\n", row_index);
			*/
		}
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
			flag_new_buf = 1; // newly malloc
			new_buf = (mmcoordinate_c_str *)malloc(sizeof(mmcoordinate_c_str) * new_nzero_total_num);
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
				strcpy(new_buf[i].val_re_str, buf[i].val_re_str);
				strcpy(new_buf[i].val_im_str, buf[i].val_im_str);
			//	printf("val_str -> %s-->%s--\n", buf[i].val_str, new_buf[i].val_str);

				// copy to opposite triangular area except diagonal element
				if(buf[i].row_index != buf[i].col_index)
				{
			//		printf("set %d -> ", total_index);
					// fix! : 2012-07-13
					nzero_col_dim[buf[i].col_index]++;

					new_buf[total_index].row_index = buf[i].col_index;
					new_buf[total_index].col_index = buf[i].row_index;
					strcpy(new_buf[total_index].val_re_str, buf[i].val_re_str);
					strcpy(new_buf[total_index].val_im_str, buf[i].val_im_str);
					total_index++;
				}
			}
			//buf = new_buf;
			nzero_total_num = new_nzero_total_num;
		}
	}

	// sorting
//	qsort(buf, nzero_total_num, sizeof(mmcoordinate_str), compare_mmcoordinate_str);
	qsort(new_buf, new_nzero_total_num, sizeof(mmcoordinate_c_str), compare_mmcoordinate_c_str);
//	for(i = 0; i < nzero_total_num; i++)
//		printf("%d %d %s %d\n", buf[i].row_index, buf[i].col_index, buf[i].val_str, nzero_col_dim[buf[i].row_index]);


	// initialize drsmatrix
	cdsmat = init2_cmpfrsmatrix(row_dim, nzero_col_dim, new_nzero_total_num, prec);
	if(cdsmat == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate matrix area(prec=%ld, dim= %d x %d\n", prec, row_dim, col_dim);
		return NULL;
	}

	mpf_init2(tmp[0], prec);
	mpf_init2(tmp[1], prec);

	// copy buffers to drsmatrix
	buf_total_index = 0;
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < cdsmat->nzero_col_dim[i]; j++)
		{
			//mpf_set_str(cdsmat->element[total_index], new_buf[total_index].val_str, 10);
			mpf_set_str(tmp[0], new_buf[total_index].val_re_str, 10);			
			mpf_set_str(tmp[1], new_buf[total_index].val_im_str, 10);
			mpc_set_fr_fr(cdsmat->element[total_index], tmp[0], tmp[1], MPC_RNDNN);
			cdsmat->nzero_index[i][j] = new_buf[total_index].col_index;
			total_index++;
		}
	}

	// free buf
	if(buf != NULL)
		free(buf);

	//if(new_buf != NULL)
	if(flag_new_buf == 1)
		free(new_buf);
	
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);

	return cdsmat;
}
// read MatrixMarket format (coodinate type only)
CMPFRSMatrix init2_cmpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec)
{
	return _init2_cmpfrsmatrix_readMMcoordinate(fname, prec);
}

// read MatrixMarket format (coodinate type only)
CMPFRSMatrix init_cmpfrsmatrix_readMMcoordinate(const char *fname)
{
	unsigned long prec;

	prec = get_bnc_default_prec();
	return _init2_cmpfrsmatrix_readMMcoordinate(fname, prec);
}

#endif // USE_GMP

