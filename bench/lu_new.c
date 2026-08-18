// LU decomposition with partial pivoting
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// A * B
int DMatMul(double mat_a[], long int row_dim, long int mid_dim, double mat_b[], long int col_dim)
{
    
}

// LU decomposition for row-major matrix
int DLUdecompPM(double mat[], long int row_dim, long int col_dim, long int ch[])
{
	long int i, j, k, itmp, maxii, index;
	double dtmp, dmaxii;

    for(i = 0; i < row_dim; i++)
        ch[i] = i;

	for(i = 0; i < row_dim; i++)
	{
        index = i * col_dim + i;
		dmaxii = fabs(mat[index]);
		maxii = i;
		for(j = (i + 1); j < row_dim; j++)
		{
            index = j * col_dim + i; 
			dtmp = fabs(mat[index]);
			if(dtmp > dmaxii)
			{
				maxii = j;
				dmaxii = dtmp;
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! DLUdecompPM!\n", i);
			return -1;
		}

		if(maxii != i)
        {
            itmp = ch[maxii];
            ch[maxii] = ch[i];
            ch[i] = itmp;

            // exchange ch[i] and i
            for(j = 0; j < col_dim; j++)
            {
                index = i * col_dim + j;
                dmaxii = mat[index];
                mat[index] = mat[ch[i] * col_dim + j];
                mat[ch[i] * col_dim + j] = dmaxii;
            }
        }

		for(j = (i + 1); j < row_dim; j++)
			mat[j * col_dim + i] /= mat[i * col_dim + i];

		for(j = (i + 1); j < row_dim; j++)
		{
			for(k = (i + 1); k < col_dim; k++)
				mat[j * col_dim + k] -= mat[j * col_dim + i] * mat[i * col_dim + k];
		}
	}

	return 0;
}

int SolveDLSPM(double answer[], double lu[], long int row_dim, long int col_dim, long int ch[], double b[])
{
	long int i, j;
	double dtmp;

	for(i = 0; i < row_dim; i++)
		answer[i] = b[ch[i]];

	
/* Forward */
	for(i = 0; i < row_dim; i++)
	{
		if(lu[i * col_dim + i] == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < row_dim; j++)
			answer[j] -= lu[j * col_dim + i] * answer[i];
	}

/* Backword */
	for(i = (row_dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < col_dim; j++)
			answer[i] -= lu[i * col_dim + j] * answer[j];
		answer[i] /= lu[i * col_dim + i];
	}

	return 0;
}

int main(int argc, char *argv[])
{
    long int dim = 3;
    long int i, j, *pivot;
    double *mat_a, *vec_b, *vec_x, *vec_true_x;

    mat_a = (double *)calloc(dim * dim, sizeof(double));
    vec_b = (double *)calloc(dim, sizeof(double));
    vec_x = (double *)calloc(dim, sizeof(double));
    vec_true_x = (double *)calloc(dim, sizeof(double));
    pivot = (long int *)calloc(dim, sizeof(long int));

    for(i = 0; i < dim; i++)
    {
        vec_true_x[i] = (double)(i + 1);
        for(j = 0; j < dim; j++)
            mat_a[i * dim + j] = 1.0 / (double)(i + j + 1);
    }
    for(i = 0; i < dim; i++)
    {
        vec_b[i] = 0.0;
        for(j = 0; j < dim; j++)
            vec_b[i] += mat_a[i * dim + j] * vec_true_x[j];
    }

    DLUdecompPM(mat_a, dim, dim, pivot);
    SolveDLSPM(vec_x, mat_a, dim, dim, pivot, vec_b);

    for(i = 0; i < dim; i++)
        printf("%5ld %5ld %25.17e %25.17e\n", i, pivot[i], vec_true_x[i], vec_x[i]);

    free(mat_a);
    free(vec_b);
    free(vec_x);
    free(vec_true_x);
    free(pivot);

    return 0;
}