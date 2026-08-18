#ifndef __LU_BENCH__
	#define __LU_BENCH__

#ifdef USE_GMP

//#include "serial_lu_bench.c"

// (1) L11 * U11 = A11
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  min_dim        |
// |                 |
// |                 |
// |                 |
// +-----------+-----+
//
int MPFLUdecomp_square(MPFMatrix a, long int start_index, long int min_dim);

// (2) Solve L21 * U11 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int MPFLUdecomp_l21(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim);

// (3) Solve L11 * U12 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int MPFLUdecomp_u12(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim);

// (4) D22 := L21 * U12
// (5) A22 := A22 - D22
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |  A22   |
// |  |     |        |
// +--+-----+--------+
//
#define STRASSEN_MIN_DIM 32
int MPFLUdecomp_a22(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim);

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
int MPFLUdecomp_strassen(MPFMatrix a, long int min_dim);

// n!
void mpf_factorial(mpf_t ret, long int n);
// Pascal Matrix
void pascal_mpfmatrix(MPFMatrix ret, long int dim);
// I - randmatrix
void im_rand_mpfmatrix(MPFMatrix ret, unsigned long seed);
void get_mpfproblem(MPFMatrix a, MPFVector b, MPFVector ans);

#ifdef _OPENMP
int MPFLUdecomp_square_omp(MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_l21_omp(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_u12_omp(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_a22_omp(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim);
int MPFLUdecomp_strassen_omp(MPFMatrix a, long int min_dim);
int MPFLUdecomp_omp(MPFMatrix a);
#endif // _OPENMP

#endif // USE_GMP

#endif // __LU_BENCH__
