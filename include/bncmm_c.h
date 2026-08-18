/* read and write complex matrix and vector */
//#include "bncmm.h"

#ifndef __BNCMM_C_H

#define __BNCMM_C_H

/* double */
// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	double _Complex val;
} mmcoordinate_c;

// read MatrixMarket format (coodinate or array type only) as vector if possible
CDVector init_cdvector_readMMcoordinate(const char *fname);

// read MatrixMarket format (coodinate type only)
CDMatrix init_cdmatrix_readMMcoordinate(const char *fname);

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cdvector(const char *fname, CDVector dvec);

// writer vector as MatrixMarket format (array type) 
int writeMMarray_cdvector(const char *fname, CDVector dvec);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cdmatrix(const char *fname, CDMatrix dmat);

// 2024-09-03(Tue) T.Kouya
// read MatrixMarket format (coodinate type only)
CDRSMatrix init_cdrsmatrix_readMMcoordinate(const char *fname);

/* MPF */
#ifdef USE_GMP

// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	char val_re_str[MAX_VAL_STR];
	char val_im_str[MAX_VAL_STR];
} mmcoordinate_c_str;

// MatrixMarket(Coordinate) struct
typedef struct {
	int row_index;
	int col_index;
	mpc_ptr val;
} mmcoordinate_mpc;

// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector _init2_cmpfvector_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector init2_cmpfvector_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only) as vector if possible
CMPFVector init_cmpfvector_readMMcoordinate(const char *fname);

// read MatrixMarket format (coodinate type only)
CMPFMatrix _init2_cmpfmatrix_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only)
CMPFMatrix init2_cmpfmatrix_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only)
CMPFMatrix init_cmpfmatrix_readMMcoordinate(const char *fname);

// writer vector as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cmpfvector(const char *fname, CMPFVector dvec);

// writer vector as MatrixMarket format (array type) 
int writeMMarray_cmpfvector(const char *fname, CMPFVector dvec);

// writer matrix as MatrixMarket format (coodinate type) 
int writeMMcoordinate_cmpfmatrix(const char *fname, CMPFMatrix dmat);

// compare mmcoordinate
int compare_mmcoordinate_c_str(const void *a, const void *b);

// read MatrixMarket format (coodinate type only)
CMPFRSMatrix _init2_cmpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only)
CMPFRSMatrix init2_cmpfrsmatrix_readMMcoordinate(const char *fname, unsigned long prec);

// read MatrixMarket format (coodinate type only)
CMPFRSMatrix init_cmpfrsmatrix_readMMcoordinate(const char *fname);

#endif // USE_GMP

#endif /* End of __BNCMM_C_H */
