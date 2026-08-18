#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncmatmul.h"
#include "mkl_spblas.h" // Intel inspector-executor sparse lib

// ret := (sparse_matrix_t)mat
sparse_status_t convert_drsmatrix_mkl_csrmat(sparse_matrix_t *ret, MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DRSMatrix mat)
{
	long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    int row_dim = (int)mat->row_dim;
    sparse_status_t ret_mkl;

	// convert our CSR to intel math kernel csr format
	//i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	//j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));
	ret_i_csr_start = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_i_csr_end = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_j_csr = (MKL_INT *)calloc(mat->real_nzero_total_num, sizeof(MKL_INT));

    total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		ret_i_csr_start[i] = (MKL_INT)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
    		//ret_i_csr_end[i] = (MKL_INT)total_index - 1;
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
		ret_i_csr_end[i] = (MKL_INT)total_index;
        //ret_i_csr_end[i] = ret_i_csr_start[i] + mat->real_nzero_col_dim[i];
    }
	//i_mat_csr_end[row_dim] = mat->real_nzero_total_num;
	//mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);
    ret_mkl = mkl_sparse_d_create_csr(
        ret,
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        ret_i_csr_start,
        ret_i_csr_end,
        ret_j_csr,
        mat->element
    );

    //mkl_sparse_destroy(mat);

    return ret_mkl;
}

// init and set dvector
DVector init_set_dvector(DVector org)
{
    DVector dv;

    dv = init_dvector(org->dim);
    subst_dvector(dv, org);

    return dv;
}

int main(int argc, char *argv[])
{
    DRSMatrix drs_mat;
    DVector dx, dv, org_dx;
    sparse_matrix_t mkl_drsmat;
    MKL_INT *i_csr_start, *i_csr_end, *j_csr;
    struct matrix_descr descr;

    if(argc <= 1)
    {
        printf("Usage: %s [sparse matrix filename]\n", argv[0]);
        return 0;
    }

    drs_mat = init_drsmatrix_readMMcoordinate(argv[1]);

    //printf("drs_mat = \n");
    //print_drsmatrix(drs_mat);
    printf("||drs_mat||_F = %25.17e\n", normf_drsmatrix(drs_mat));

    dx = init_dvector(drs_mat->col_dim);
    dv = init_dvector(drs_mat->col_dim);
    for(int i = 0; i < dx->dim; i++)
        set_dvector_i(dx, i, (double)(i + 1));

    org_dx = init_set_dvector(dx);

    //printf("dv = \n");
    //print_dvector(dv);
    printf("||dx||_2      = %25.17e\n", norm2_dvector(dx));
    //printf("||dx||_2      = %25.17e\n", norm2_dvector(org_dx));

    // DRSmatrix -> sparse_matrix_t
    printf("ret_mkl = %d\n", convert_drsmatrix_mkl_csrmat(&mkl_drsmat, i_csr_start, i_csr_end, j_csr, drs_mat));

    // b := A * x by IMKL
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    //descr.mode = ?
    //descr.diag = ?
    mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        (double)1.0,
        mkl_drsmat,
        descr,
        dx->element,
        (double)0.0,
        dv->element
    );
    //printf("A * dx = \n");
    //print_dvector(dx);
    printf("||A * dx||_2  = %25.17e\n", norm2_dvector(dv));

    // b := A * x by BNCmatmul
    mul_drsmatrix_dvec(dv, drs_mat, org_dx);
    printf("||A * dx||_2  = %25.17e\n", norm2_dvector(dv));

    free_drsmatrix(drs_mat);
    free_dvector(dx); free_dvector(org_dx); free_dvector(dv);
    free(i_csr_start); free(i_csr_end); free(j_csr);

    mkl_sparse_destroy(drs_mat);

    return 0;
}