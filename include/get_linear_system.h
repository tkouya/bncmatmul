#ifndef __GET_LINEAR_SYSTEM_H__
	#define __GET_LINEAR_SYSTEM_H__

#define max(i,j) (((i) > (j)) ? (i) : (j))
#define min(i,j) (((i) < (j)) ? (i) : (j))

/* set a vector for numerical validation */
/* type =  0 ... test_v = [1 2 ... n]^T                            */
/* type =  1 ... test_v = [ ... (-1)^irand * irand ...]^T          */
/* type >= 2 ... test_v = [ ... (-1)^irand * irand/randmax ... ]^T */
void set_test_mpfvector(MPFVector test_v, int type);

#define _BNC_GENMAT_REAL_DENSE 0
#define _BNC_GENMAT_REAL_TRIDIAG 1

/* generate random regular matrix */
void generate_regular_mpfmatrix(mpf_t cond, MPFMatrix mat, MPFMatrix inv_mat, int matrix_type, int seed);

/* Generate Real Dense Matrix with Eigenvalues as eig[] */
/* A := X^(-1) * Diag(eig[]) * X */
/* matrix_type = 0: Unsymmetrix Dense Matrix */
/* mat: A, trans_mat X */
void generate_mpfmatrix(MPFMatrix mat, MPFMatrix trans_mat, MPFVector eig, int matrix_type, int seed);

/* check equarity of two vectors */
/* return 0 if vec_a == mat_b */
/* return -1 if vec_a_i < vec_b_i for all i, j */
/* return +1 if vec_a_i > vec_b_i for all i, j */
/* return +2 otherwise */
int compare_mpfvector(MPFVector vec_a, MPFVector vec_b);

/* check equarity of two matrices */
/* return 0 if mat_a == mat_b */
/* return -1 if mat_a_ij < mat_b_ij for all i, j */
/* return +1 if mat_a_ij > mat_b_ij for all i, j */
/* return +2 otherwise */
int compare_mpfmatrix(MPFMatrix mat_a, MPFMatrix mat_b);

/* get nearly correctly rounded coefficient matrix A */
int get_correct_rounded_mpfmatrix(MPFMatrix mat, void(* get_coefmat_mpfmatrix)(MPFMatrix mat));

/* get nearly correctly rounded coefficient matrix A and constant vector b */
/* A * x = b ... mat as A, vec as b, ans_vec as x */
int get_correct_rounded_mpfmatrix_mpfvec(MPFMatrix mat, MPFVector vec, MPFVector ans_vec, void(* get_coefmat_mpfmatrix)(MPFMatrix mat));


#endif // __GET_LINEAR_SYSTEM_H__