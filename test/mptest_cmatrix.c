// mptest_cmatrix.c : complex linear equation
#include "matmul_strassen.h"

// produce mp test linear matrix
// D = diag[10^0, ..., 10^(log10_min_d)]
// V = random
// A := V * D * V^(-1)
// x := [1, 2, ..., n]
// b := A * b
void get_mptest_linear_eq_c(CMPFMatrix mat_a, CMPFVector vec_b, CMPFVector true_x, CMPFMatrix mp_mat_true_eigenvalues, unsigned long prec, long int dimension, long int log10_max, long int log10_min, int random_seed)
{
    long int i, j;
    mpf_t tmp, tmp1, base10, log10_step, in_log10_min, in_log10_max;
    mpc_t ctmp, ctmp1;
    CMPFMatrix rand_v, inv_rand_v;

    printf("prec in bits = %ld\n", prec);

    mpf_init2(log10_step, prec);
    mpf_init2(base10, prec); mpf_set_ui(base10, 10UL);
    mpf_init2(in_log10_max, prec); mpf_set_si(in_log10_max, log10_max);
    mpf_init2(in_log10_min, prec); mpf_set_si(in_log10_min, log10_min);
    mpf_init2(tmp, prec);
    mpf_init2(tmp1, prec);
    mpc_init2(ctmp, prec);
    mpc_init2(ctmp1, prec);

    // 対角行列
    set0_cmpfmatrix(mp_mat_true_eigenvalues);
    //log10_max_d, log10_min_d = 1.0, -25.0
    //log10_step_d = (log10_max_d - log10_min_d) / (dimension)
    mpf_sub(log10_step, in_log10_max, in_log10_min);
    mpf_div_ui(log10_step, log10_step, (unsigned long)dimension);
    //log10_d = np.arange(log10_min_d, log10_max_d, log10_step_d)
    //mat_true_eigenvalues = np.diag([10**i for i in log10_d])
    for(i = 0; i < dimension; i++)
    {
        mpf_mul_ui(tmp1, log10_step, i);
        //mpf_add(tmp1, tmp1, in_log10_min);
        mpf_sub(tmp1, in_log10_max, tmp1);
        mpfr_pow(tmp, base10, tmp1, get_bnc_default_rounding_mode());
        //printf("%d, ", i); mpf_out_str(stdout, 10, 3, tmp); printf("="); mpf_out_str(stdout, 10, 3, base10); printf("^"); mpf_out_str(stdout, 10, 3, tmp1); printf("\n");
        mpc_set_fr_fr(ctmp, tmp, tmp, get_bnc_default_rounding_mode());
        set_cmpfmatrix_ij(mp_mat_true_eigenvalues, i, i, ctmp);
    }
    printf("D(%ld, %ld) =\n", dimension, dimension);
    //print_cmpfmatrix(mp_mat_true_eigenvalues);

    // 乱数行列 in [0, 1]
    rand_v = init2_cmpfmatrix(dimension, dimension, prec);
    //np.random.seed(random_seed)
    //mat_t = sc.random.rand(dimension, dimension)
    mpf_srand(random_seed);
    for(i = 0; i < dimension; i++)
    {
        for(j = 0; j < dimension; j++)
        {
            //mpf_urand(tmp); mpf_urand(tmp1);
            mpf_nrand(tmp); mpf_nrand(tmp1);
            mpc_set_fr_fr(ctmp, tmp, tmp1, get_bnc_default_rounding_mode());
            set_cmpfmatrix_ij(rand_v, i, j, ctmp);
        }
    }
    printf("V(%ld, %ld) =\n", dimension, dimension);
    //mat_t_inv = sclinalg.inv(mat_t)
/*
    inv_rand_v = init2_cmpfmatrix(dimension, dimension, prec);
    subst_cmpfmatrix(inv_rand_v, rand_v);
    inv_cmpfmatrix(inv_rand_v);
    //print('T = \n', mat_t)
    printf("V(%d, %d)^(-1) =\n", dimension, dimension);
    //print_mpfmatrix(rand_v);

    //mp_mat_a = mp_mat_t * mp_mat_true_eigenvalues * mp_mat_t_inv
    mul_cmpfmatrix(mat_a, rand_v, mp_mat_true_eigenvalues);
    mul_cmpfmatrix(mat_a, inv_rand_v, mat_a);
    ///print('mp.A = \n', mp_mat_a)
    printf("A(%d, %d) =\n", dimension, dimension);
    //print_mpfmatrix(mat_a);
*/
    printf("A(%ld, %ld) = V\n", dimension, dimension);
    subst_cmpfmatrix(mat_a, rand_v);
    // A * x = b
    //mp_vec_true_x = mpmath.matrix([mpmath.mpmathify(i) for i in range(dimension)])
    for(i = 0; i < dimension; i++)
    {
        mpc_set_ui_ui(ctmp, (unsigned long)(i + 1), (unsigned long)(i + 1), get_bnc_default_rounding_mode());
        set_cmpfvector_i(true_x, i, ctmp);
    }
    printf("x(%ld) = \n", dimension);
    //print_mpfvector(true_x);

    //mp_vec_b = mp_mat_a * mp_vec_true_x
    mul_cmpfmatrix_cmpfvec(vec_b, mat_a, true_x);
    printf("b(%ld) = \n", dimension);
    //print_mpfvector(vec_b);
    

    //# A, b, x, eig
    //return mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_true_eigenvalues

    free_cmpfmatrix(rand_v);

    mpf_clear(log10_step);
    mpf_clear(base10);
    mpf_clear(in_log10_max);
    mpf_clear(in_log10_min);
    mpf_clear(tmp);
    mpf_clear(tmp1);
    mpc_clear(ctmp);
    mpc_clear(ctmp1);
}

int main(int argc, char *argv[])
{
    long int dim;
    unsigned long prec;
	char fname_A[256], fname_true_x[256], fname_vec_b[256];
    CMPFMatrix A, diag;
    CMPFVector true_x, vec_b;

//	dim = 128;
	if(argc <= 1)
	{
		fprintf(stderr, "Usage: %s [dim] [prec_b]\n", argv[0]);
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

	prec = 128;
	if(argc >= 3)
	{
		prec = (unsigned long)atol(argv[2]);
	//	if(prec < 128)
	//		prec = 128;
	}

    set_bnc_default_prec(prec);

    A = init2_cmpfmatrix(dim, dim, prec);
    diag = init2_cmpfmatrix(dim, dim, prec);
    true_x = init2_cmpfvector(dim, prec);
    vec_b = init2_cmpfvector(dim, prec);

    //get_mptest_linear_eq_c(A, vec_b, true_x, diag, prec, dim, 1, 25, 20230407);
    //get_mptest_linear_eq_c(A, vec_b, true_x, diag, prec, dim, 1, -12, 20230407);
    //get_mptest_linear_eq_c(A, vec_b, true_x, diag, prec, dim, 1, -5, 20230407);
    get_mptest_linear_eq_c(A, vec_b, true_x, diag, prec, dim, 1, -1, 20230407);

   	sprintf(fname_A, "../python/cmat_a_%ld_%ld_b%lu_c.txt", dim, dim, prec);
	sprintf(fname_true_x, "../python/cvec_true_x_%ld_b%lu_c.txt", dim, prec);
	sprintf(fname_vec_b, "../python/cvec_b_%ld_b%lu_c.txt", dim, prec);

    write_test_linear_eq_c(A, true_x, vec_b, dim, fname_A, fname_true_x, fname_vec_b);

    free_cmpfmatrix(A);
    free_cmpfmatrix(diag);
    free_cmpfvector(true_x);
    free_cmpfvector(vec_b);

    return 0;
}