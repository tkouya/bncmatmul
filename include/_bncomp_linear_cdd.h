// -------------------
// bncomp_linear_cdd.c
// -------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_cddvector(CDDVector c, CDDVector a);

/* c = a + b */
void _bncomp_add_cddvector(CDDVector c, CDDVector a, CDDVector b);

/* c = a - b */
void _bncomp_sub_cddvector(CDDVector c, CDDVector a, CDDVector b);

/* c = val * a */
void _bncomp_cmul_cddvector_4m(CDDVector c, cddfloat *val, CDDVector a);

/* c = val * a */
void _bncomp_cmul_cddvector_3m(CDDVector c, cddfloat *val, CDDVector a);

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_cddvector(cddfloat *ret, CDDVector a, CDDVector b);

/* a^T * b */
void _bncomp_dotp_cddvector(cddfloat *ret, CDDVector a, CDDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_cddmatrix(CDDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);

/* c := a - b */
void _bncomp_sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);

/* c := sc * a */
void _bncomp_cmul_cddmatrix(CDDMatrix c, cddfloat *sc, CDDMatrix a);

/* c := a */
void _bncomp_subst_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := I */
void _bncomp_setI_cddmatrix(CDDMatrix c);

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_cddmatrix(CDDMatrix mat);

/* v := a * vb */
void _bncomp_mul_cddmatrix_cddvec_4m(CDDVector v, CDDMatrix a, CDDVector vb);

/* v := a * vb */
void _bncomp_mul_cddmatrix_cddvec_3m(CDDVector v, CDDMatrix a, CDDVector vb);

#ifndef USE_3M
#define _bncomp_mul_cddmatrix_cddvec _bncomp_mul_cddmatrix_cddvec_4m
#else // USE_3M
#define _bncomp_mul_cddmatrix_cddvec _bncomp_mul_cddmatrix_cddvec_3m
#endif // USE_3M

/* v := a^T * vb */
void _bncomp_mul_cddmatrixt_cddvec(CDDVector v, CDDMatrix a, CDDVector vb);

/* v := conj(a)^T * vb */
void _bncomp_mul_cddmatrixs_cddvec(CDDVector v, CDDMatrix a, CDDVector vb);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_3m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_4m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

#ifdef USE_4M
#define _bncomp_mul_cddmatrix_oz _bncomp_mul_cddmatrix_oz_4m
#else // USE_4M
#define _bncomp_mul_cddmatrix_oz _bncomp_mul_cddmatrix_oz_3m
#endif // USE_4M

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix a, CDDMatrix b);
// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix a, CDDMatrix b);
