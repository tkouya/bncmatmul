/********************************************************************************/
/*                                                                              */
/* writematrix.c : Reading & Writing MatrixMarket Format Files                  */
/* Copyright (c) 2011 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* 2011-08-29 Version 0.0: make writematrix.c                                   */
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

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_dvector(const char *fname, DVector dvec)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dvec->dim, 1, dvec->dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dvec->dim, 1, dvec->dim);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%d %d %25.17g\n", i + 1, 1, get_dvector_i(dvec, i));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer vector as MatrixMarket format (array type) 
int writeMMarray_dvector(const char *fname, DVector dvec)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_array_size(stdout, dvec->dim, 1);

    mm_write_banner(f, matcode); 
    mm_write_mtx_array_size(f, dvec->dim, 1);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%25.17g\n", get_dvector_i(dvec, i));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_dmatrix(const char *fname, DMatrix dmat)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    for (i = 0; i < dmat->col_dim; i++)
	{
		for(j = 0; j < dmat->row_dim; j++)
			fprintf(f, "%d %d %25.17g\n", j + 1, i + 1, get_dmatrix_ij(dmat, j, i));
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// compare mmcoordinate to sort as columnwise
int compare_mmcoordinate_col(const void *a, const void *b)
{
	mmcoordinate *mm_a, *mm_b;

	mm_a = (mmcoordinate *)a;
	mm_b = (mmcoordinate *)b;

	if(mm_a->col_index < mm_b->col_index)
	{
		return -1;
	}
	else if(mm_a->col_index == mm_b->col_index)
	{
		if(mm_a->row_index < mm_b->row_index)
			return -1;
		else if(mm_a->row_index == mm_b->row_index)
			return 0;
		else
			return 1;
	}
	else
		return 1;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_drsmatrix(const char *fname, DRSMatrix dmat)
{
    MM_typecode matcode;
    FILE *f;
    int i, j, total_index;
    mmcoordinate *buf;

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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->nzero_total_num);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->nzero_total_num);

	// initialize buffers
	buf = (mmcoordinate *)calloc(sizeof(mmcoordinate), dmat->nzero_total_num);

	if(buf == NULL)
		return ERROR;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	total_index = 0;
    for (i = 0; i < dmat->row_dim; i++)
    {
    	for(j = 0; j < dmat->nzero_col_dim[i]; j++)
    	{
			// set
			buf[total_index].row_index = i + 1;
			buf[total_index].col_index = dmat->nzero_index[i][j] + 1;
			buf[total_index].val = dmat->element[total_index];
			total_index++;
		}
	}

	// sorting
	qsort(buf, dmat->nzero_total_num, sizeof(mmcoordinate), compare_mmcoordinate_col);
    for (i = 0; i < dmat->nzero_total_num; i++)
	{
		fprintf(f, "%d %d %25.17g\n", buf[i].row_index, buf[i].col_index, buf[i].val);
	}

    if (f !=stdin) fclose(f);

	free(buf);

	return SUCCESS;
}

/* MPF */
#ifdef USE_GMP

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfvector(const char *fname, MPFVector dvec)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dvec->dim, 1, dvec->dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dvec->dim, 1, dvec->dim);

    for (i = 0; i < dvec->dim; i++)
	{
			fprintf(f, "%d %d ", i + 1, 1);
			mpf_out_str(f, 10, 0, get_mpfvector_i(dvec, i));
			fprintf(f, "\n");
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer vector as MatrixMarket format (array type) 
int writeMMarray_mpfvector(const char *fname, MPFVector dvec)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_array_size(stdout, dvec->dim, 1);

    mm_write_banner(f, matcode); 
    mm_write_mtx_array_size(f, dvec->dim, 1);

    for (i = 0; i < dvec->dim; i++)
	{
			mpf_out_str(f, 10, 0, get_mpfvector_i(dvec, i));
			fprintf(f, "\n");
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfmatrix(const char *fname, MPFMatrix dmat)
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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->row_dim * dmat->col_dim);

    for (i = 0; i < dmat->col_dim; i++)
	{
		for(j = 0; j < dmat->row_dim; j++)
		{
			fprintf(f, "%d %d ", j + 1, i + 1);
			mpf_out_str(f, 10, 0, get_mpfmatrix_ij(dmat, j, i));
			fprintf(f, "\n");
		}
	}

    if (f !=stdin) fclose(f);

	return SUCCESS;
}

// compare mmcoordinate to sort as columnwise
int compare_mmcoordinate_mpf_col(const void *a, const void *b)
{
	mmcoordinate_mpf *mm_a, *mm_b;

	mm_a = (mmcoordinate_mpf *)a;
	mm_b = (mmcoordinate_mpf *)b;

	if(mm_a->col_index < mm_b->col_index)
	{
		return -1;
	}
	else if(mm_a->col_index == mm_b->col_index)
	{
		if(mm_a->row_index < mm_b->row_index)
			return -1;
		else if(mm_a->row_index == mm_b->row_index)
			return 0;
		else
			return 1;
	}
	else
		return 1;
}

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_mpfrsmatrix(const char *fname, MPFRSMatrix dmat)
{
    MM_typecode matcode;
    FILE *f;
    int i, j, total_index;
    mmcoordinate_mpf *buf;

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
    mm_set_real(&matcode);

    mm_write_banner(stdout, matcode); 
    mm_write_mtx_crd_size(stdout, dmat->row_dim, dmat->col_dim, dmat->nzero_total_num);

    mm_write_banner(f, matcode); 
    mm_write_mtx_crd_size(f, dmat->row_dim, dmat->col_dim, dmat->nzero_total_num);

	// initialize buffers
	buf = (mmcoordinate_mpf *)calloc(sizeof(mmcoordinate_mpf), dmat->nzero_total_num);

	if(buf == NULL)
		return ERROR;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	total_index = 0;
    for (i = 0; i < dmat->row_dim; i++)
    {
    	for(j = 0; j < dmat->nzero_col_dim[i]; j++)
    	{
			// set
			buf[total_index].row_index = i + 1;
			buf[total_index].col_index = dmat->nzero_index[i][j] + 1;
			buf[total_index].val = dmat->element[total_index];
			total_index++;
		}
	}

	// sorting
	qsort(buf, dmat->nzero_total_num, sizeof(mmcoordinate_mpf), compare_mmcoordinate_mpf_col);
    for (i = 0; i < dmat->nzero_total_num; i++)
	{
		fprintf(f, "%d %d ", buf[i].row_index, buf[i].col_index);
		mpf_out_str(f, 10, 0, buf[i].val);
		fprintf(f, "\n");
	}

    if (f !=stdin) fclose(f);

	free(buf);

	return SUCCESS;
}

#endif // USE_GMP

