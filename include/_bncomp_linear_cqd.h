 // -------------------
// bncomp_linear_cqd.c
// -------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_cqdvector(CQDVector c, CQDVector a);

/* c = a + b */
void _bncomp_add_cqdvector(CQDVector c, CQDVector a, CQDVector b);

/* c = a - b */
void _bncomp_sub_cqdvector(CQDVector c, CQDVector a, CQDVector b);

/* c = val * a */
void _bncomp_cmul_cqdvector_4m(CQDVector c, cqdfloat *val, CQDVector a);

/* c = val * a */
void _bncomp_cmul_cqdvector_3m(CQDVector c, cqdfloat *val, CQDVector a);

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b);

/* a^T * b */
void _bncomp_dotp_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_cqdmatrix(CQDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);

/* c := a - b */
void _bncomp_sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);

/* c := sc * a */
void _bncomp_cmul_cqdmatrix(CQDMatrix c, cqdfloat *sc, CQDMatrix a);

/* c := a */
void _bncomp_subst_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := I */
void _bncomp_setI_cqdmatrix(CQDMatrix c);

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_cqdmatrix(CQDMatrix mat);

/* v := a * vb */
void _bncomp_mul_cqdmatrix_cqdvec_4m(CQDVector v, CQDMatrix a, CQDVector vb);

/* v := a * vb */
void _bncomp_mul_cqdmatrix_cqdvec_3m(CQDVector v, CQDMatrix a, CQDVector vb);

#ifndef USE_3M
#define _bncomp_mul_cqdmatrix_cqdvec _bncomp_mul_cqdmatrix_cqdvec_4m
#else // USE_3M
#define _bncomp_mul_cqdmatrix_cqdvec _bncomp_mul_cqdmatrix_cqdvec_3m
#endif // USE_3M

/* v := a^T * vb */
void _bncomp_mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb);

/* v := conj(a)^T * vb */
void _bncomp_mul_cqdmatrixs_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

#ifdef USE_4M
#define _bncomp_mul_cqdmatrix_oz _bncomp_mul_cqdmatrix_oz_4m
#else // USE_4M
#define _bncomp_mul_cqdmatrix_oz _bncomp_mul_cqdmatrix_oz_3m
#endif // USE_4M

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix a, CQDMatrix b);
// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix a, CQDMatrix b);
