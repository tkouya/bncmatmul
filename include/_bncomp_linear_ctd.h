// -------------------
// bncomp_linear_ctd.c
// -------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_ctdvector(CTDVector c, CTDVector a);

/* c = a + b */
void _bncomp_add_ctdvector(CTDVector c, CTDVector a, CTDVector b);

/* c = a - b */
void _bncomp_sub_ctdvector(CTDVector c, CTDVector a, CTDVector b);

/* c = val * a */
void _bncomp_cmul_ctdvector_4m(CTDVector c, ctdfloat *val, CTDVector a);

/* c = val * a */
void _bncomp_cmul_ctdvector_3m(CTDVector c, ctdfloat *val, CTDVector a);

/* (a, b) */
/* (a, b) = conj(a)^T * b */
void _bncomp_ip_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b);

/* a^T * b */
void _bncomp_dotp_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_ctdmatrix(CTDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);

/* c := a - b */
void _bncomp_sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);

/* c := sc * a */
void _bncomp_cmul_ctdmatrix(CTDMatrix c, ctdfloat *sc, CTDMatrix a);

/* c := a */
void _bncomp_subst_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := I */
void _bncomp_setI_ctdmatrix(CTDMatrix c);

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_ctdmatrix(CTDMatrix mat);

/* v := a * vb */
void _bncomp_mul_ctdmatrix_ctdvec_4m(CTDVector v, CTDMatrix a, CTDVector vb);

/* v := a * vb */
void _bncomp_mul_ctdmatrix_ctdvec_3m(CTDVector v, CTDMatrix a, CTDVector vb);

#ifndef USE_3M
#define _bncomp_mul_ctdmatrix_ctdvec _bncomp_mul_ctdmatrix_ctdvec_4m
#else // USE_3M
#define _bncomp_mul_ctdmatrix_ctdvec _bncomp_mul_ctdmatrix_ctdvec_3m
#endif // USE_3M

/* v := a^T * vb */
void _bncomp_mul_ctdmatrixt_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb);

/* v := conj(a)^T * vb */
void _bncomp_mul_ctdmatrixs_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_3m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_4m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

#ifdef USE_4M
#define _bncomp_mul_ctdmatrix_oz _bncomp_mul_ctdmatrix_oz_4m
#else // USE_4M
#define _bncomp_mul_ctdmatrix_oz _bncomp_mul_ctdmatrix_oz_3m
#endif // USE_4M

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_ctdmatrix_3m(CTDMatrix ret, CTDMatrix a, CTDMatrix b);
// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_ctdmatrix_4m(CTDMatrix ret, CTDMatrix a, CTDMatrix b);
