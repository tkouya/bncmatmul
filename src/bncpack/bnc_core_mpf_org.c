#include <stdio.h>
#include <stdlib.h> // abs(int)
#include <math.h> // ceil
#include <string.h> // memcpy

#include "get_secv.h"

#include "gmp.h"
#include "mpfr.h"

// size of one element of mpf_t including mantissa
static inline int size_of_mpf_t(unsigned long int prec)
{
	size_t one_mpf_t_size = 0;
	mpf_t a;

	// a = sqrt(2)
	mpf_init2(a, prec);
	mpf_set_ui(a, 2UL); mpf_sqrt(a, a);

	// -------------------------------------- GMP 6.1.2
	// typedef struct
	// {
	//   int _mp_prec; /* Max precision, in number of `mp_limb_t's.
	// 				   Set by mpf_init and modified by
	// 				   mpf_set_prec.  The area pointed to by the
	// 				   _mp_d field contains `prec' + 1 limbs.  */
	//   int _mp_size; /* abs(_mp_size) is the number of limbs the
	// 				   last field points to.  If _mp_size is
	// 				   negative this is a negative number.  */
	//   mp_exp_t _mp_exp;		/* Exponent, in the base of `mp_limb_t'.  */
	//   mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
	// } __mpf_struct;
	// 
	// /* typedef __mpf_struct MP_FLOAT; */
	// typedef __mpf_struct mpf_t[1];
	// -------------------------------------- GMP 6.1.2

	printf("mpf_prec = %ld -> %d limb(s) * %ld\n", prec, abs(a->_mp_size), sizeof(mp_limb_t));
	one_mpf_t_size = \
		sizeof(a) \
		+ sizeof(a->_mp_prec) \
		+ sizeof(a->_mp_size) \
		+ sizeof(a->_mp_exp) \
		+ sizeof(a->_mp_d) \
		+ abs(a->_mp_size) * sizeof(mp_limb_t);

	mpf_clear(a);

	return one_mpf_t_size;
}

// mpfr-impl.h

#define MPFR_BYTES_PER_MP_LIMB (GMP_NUMB_BITS/CHAR_BIT)

#define MPFR_SIGN_POS (1)
#define MPFR_SIGN_NEG (-1)

#define MPFR_IS_NEG(x) (MPFR_SIGN(x) < 0)
#define MPFR_IS_POS(x) (MPFR_SIGN(x) > 0)

#define MPFR_SET_POS(x) (MPFR_SIGN(x) = MPFR_SIGN_POS)
#define MPFR_SET_NEG(x) (MPFR_SIGN(x) = MPFR_SIGN_NEG)

/* Size of mpfr_exp_t in limbs */
#define MPFR_EXP_LIMB_SIZE \
  ((sizeof (mpfr_exp_t) - 1) / MPFR_BYTES_PER_MP_LIMB + 1)

/* Invalid exponent value (to track bugs...) */
#define MPFR_EXP_INVALID \
 ((mpfr_exp_t) 1 << (GMP_NUMB_BITS*sizeof(mpfr_exp_t)/sizeof(mp_limb_t)-2))

#define MPFR_PREC2LIMBS(p) (((p) - 1) / GMP_NUMB_BITS + 1)
#define MPFR_PREC(x)      ((x)->_mpfr_prec)
#define MPFR_EXP(x)       ((x)->_mpfr_exp)
#define MPFR_MANT(x)      ((x)->_mpfr_d)
#define MPFR_GET_PREC(x)  (MPFR_PREC (x))
#define MPFR_LAST_LIMB(x) ((MPFR_GET_PREC (x) - 1) / GMP_NUMB_BITS)
#define MPFR_LIMB_SIZE(x) (MPFR_LAST_LIMB (x) + 1)

typedef union { mp_size_t s; mp_limb_t l; } mpfr_size_limb_t;

#define MPFR_GET_ALLOC_SIZE(x) \
  (((mp_size_t *) (mpfr_size_limb_t *) MPFR_MANT(x))[-1] + 0)

#define MPFR_SET_ALLOC_SIZE(x, n) \
  (((mp_size_t *) (mpfr_size_limb_t *) MPFR_MANT(x))[-1] = (n))

#define MPFR_MALLOC_SIZE(s) \
  (sizeof(mpfr_size_limb_t) + MPFR_BYTES_PER_MP_LIMB * (size_t) (s))

#define MPFR_SET_MANT_PTR(x,p) \
  (MPFR_MANT(x) = (mp_limb_t *) ((mpfr_size_limb_t *) (p) + 1))

#define MPFR_GET_REAL_PTR(x) \
  ((mp_limb_t *) ((mpfr_size_limb_t *) MPFR_MANT(x) - 1))

//#define _BNC_NUM_LIMB_MPFR_D(a) ((int)ceil((a)->_mpfr_prec / mp_bits_per_limb))
#define _BNC_NUM_LIMB_MPFR_D(a) MPFR_PREC2LIMBS(MPFR_PREC(a))

// size of one element of mpfr_t including mantissa
static inline size_t size_of_mpfr_t(unsigned long int prec)
{
	size_t one_mpfr_t_size;

	one_mpfr_t_size = sizeof(mpfr_t) + MPFR_MALLOC_SIZE(MPFR_PREC2LIMBS(prec));

	return one_mpfr_t_size;
}

// gmp-impl.h
/* __GMPF_BITS_TO_PREC applies a minimum 53 bits, rounds upwards to a whole
   limb and adds an extra limb.  __GMPF_PREC_TO_BITS drops that extra limb,
   hence giving back the user's size in bits rounded up.  Notice that
   converting prec->bits->prec gives an unchanged value.  */
#define __GMPF_BITS_TO_PREC(n)						\
  ((mp_size_t) ((__GMP_MAX (53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define __GMPF_PREC_TO_BITS(n) \
  ((mp_bitcnt_t) (n) * GMP_NUMB_BITS - GMP_NUMB_BITS)

// alloc mpf_t array and init
//int init_mpf_t_block(unsigned long num)
static inline mpf_ptr bnc_custom_init2_mpf_t(mp_bitcnt_t prec_in_bits)
{
	mpf_ptr r;

	r = (mpf_ptr)malloc(size_of_mpf_t(prec_in_bits));

	r->_mp_size = 0;
	r->_mp_exp = 0;
	r->_mp_prec = __GMPF_BITS_TO_PREC (prec_in_bits);
//	r->_mp_d = __GMP_ALLOCATE_FUNC_LIMBS (prec + 1);
//	r->_mp_d = malloc(prec + 1);
	r->_mp_d = (mp_limb_t *)(
		(unsigned char *)r \
		+ sizeof(r->_mp_prec) \
		+ sizeof(r->_mp_size) \
		+ sizeof(r->_mp_exp) \
		+ sizeof(r->_mp_d) \
	);

	return r;
}


// alloc mpf_t array and init
//int init_mpf_t_block(unsigned long num)
static inline mpfr_ptr bnc_custom_init2_mpfr_t(mpfr_prec_t prec_in_bits)
{
	unsigned char *r_ptr;
	mp_size_t r_size;
	mpfr_ptr r;

	r_size = MPFR_PREC2LIMBS(prec_in_bits);
	//r = (mpfr_ptr)malloc(sizeof(mpfr_t) + MPFR_MALLOC_SIZE(r_size));
	r = (mpfr_ptr)malloc(size_of_mpfr_t(prec_in_bits));

//  MPFR_PREC(x) = p;                /* Set prec */
//  MPFR_EXP (x) = MPFR_EXP_INVALID; /* make sure that the exp field has a */
                                     /* valid value in the C point of view */
//  MPFR_SET_POS(x);                 /* Set a sign */
//  MPFR_SET_MANT_PTR(x, tmp);       /* Set Mantissa ptr */
//  MPFR_SET_ALLOC_SIZE(x, xsize);   /* Fix alloc size of Mantissa */
//  MPFR_SET_NAN(x);                 /* initializes to NaN */

	r->_mpfr_prec = prec_in_bits;
	r->_mpfr_exp = MPFR_EXP_INVALID;
	r->_mpfr_sign = 1;

	MPFR_SET_MANT_PTR(r, r + 1);

//	printf("b mpfr_d's alloc   size = %ld\n", r_size);
	MPFR_SET_ALLOC_SIZE(r, r_size);
//	printf("b mpfr_d's alloced size = %ld\n", MPFR_GET_ALLOC_SIZE(r));

//	MPFR_SET_NAN(r);                 /* initializes to NaN */
	mpfr_set_nan(r);

	return r;
}

#define MPFR_PTR_BNC_CUSTOM_INDEX(rtop, index, size_mpfr) ((mpfr_ptr)((unsigned char *)(rtop) + (index) * (size_mpfr)))

//int init_mpf_t_block(unsigned long num)
static inline mpfr_ptr bnc_custom_init2_mpfr_t_array(int dimension, mpfr_prec_t prec_in_bits)
{
    int i;
    size_t size_mpfr;
//	unsigned char *rtop;
	mpfr_ptr rtop;
    mpfr_ptr r;

    size_mpfr = size_of_mpfr_t(prec_in_bits);

    rtop = (mpfr_ptr)malloc((size_t)(dimension) * size_mpfr);

	if(rtop == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate mpfr array with %s!\n", __func__);
		return NULL;
	}

    for(i = 0; i < dimension; i++)
    {
		//r = (mpfr_ptr)((unsigned char *)rtop + i * size_mpfr);
		r = MPFR_PTR_BNC_CUSTOM_INDEX((mpfr_ptr)rtop, i, size_mpfr);
		
		//  MPFR_PREC(x) = p;                /* Set prec */
		//  MPFR_EXP (x) = MPFR_EXP_INVALID; /* make sure that the exp field has a */
		/* valid value in the C point of view */
		//  MPFR_SET_POS(x);                 /* Set a sign */
		//  MPFR_SET_MANT_PTR(x, tmp);       /* Set Mantissa ptr */
		//  MPFR_SET_ALLOC_SIZE(x, xsize);   /* Fix alloc size of Mantissa */
		//  MPFR_SET_NAN(x);                 /* initializes to NaN */
		
		r->_mpfr_prec = prec_in_bits;
		r->_mpfr_exp = MPFR_EXP_INVALID;
		r->_mpfr_sign = 1;
		
		MPFR_SET_MANT_PTR(r, r + 1);
		
		//    printf("b mpfr_d's alloc size = %ld\n", MPFR_PREC2LIMBS(prec_in_bits));
		MPFR_SET_ALLOC_SIZE(r, MPFR_PREC2LIMBS(prec_in_bits));
		//    printf("a mpfr_d's alloc size = %ld\n", MPFR_GET_ALLOC_SIZE(r));
		
		//    r->_mpfr_d = (mp_limb_t *)malloc(24);
		//    printf("r->_mpfr_d = %d\n", (unsigned int *)(r->_mpfr_d));
		
		//    MPFR_SET_NAN(r);                 /* initializes to NaN */
		mpfr_set_nan(r);
    }

    return (mpfr_ptr)rtop;
}

static inline mpfr_ptr bnc_custom_get_array_i(mpfr_ptr array_top, int index, mpfr_prec_t prec_in_bits)
{
//    int i;
    size_t size_mpfr = size_of_mpfr_t(prec_in_bits);

    return MPFR_PTR_BNC_CUSTOM_INDEX(array_top, index, size_mpfr);
}

static inline void bnc_custom_set_array_i(mpfr_ptr array_top, int index, mpfr_prec_t prec_in_bits, mpfr_t val, mpfr_rnd_t rmode)
{
//    int i;
    size_t size_mpfr = size_of_mpfr_t(prec_in_bits);
    
    mpfr_set(MPFR_PTR_BNC_CUSTOM_INDEX(array_top, index, size_mpfr), val, rmode);
}

static inline void bnc_custom_set_array(mpfr_ptr array_top, int dimension, mpfr_prec_t prec_in_bits, mpfr_t val, mpfr_rnd_t rmode)
{
    size_t size_mpfr = size_of_mpfr_t(prec_in_bits);
    
    while(--dimension >= 0)
    {
		printf("dimension = %d\n", dimension);
        mpfr_set(MPFR_PTR_BNC_CUSTOM_INDEX(array_top, dimension, size_mpfr), val, rmode);
    }

}

static inline void bnc_custom_set_array_ui(mpfr_ptr array_top, int dimension, mpfr_prec_t prec_in_bits, unsigned long val, mpfr_rnd_t rmode)
{
    size_t size_mpfr = size_of_mpfr_t(prec_in_bits);
    
    while(--dimension >= 0)
        mpfr_set_ui(MPFR_PTR_BNC_CUSTOM_INDEX(array_top, dimension, size_mpfr), val, rmode);

}

static inline void bnc_custom_print_full_array(mpfr_ptr array_top, int dimension, mpfr_prec_t prec_in_bits)
{
    int i;
    size_t size_mpfr = size_of_mpfr_t(prec_in_bits);
    
    for(i = 0; i < dimension; i++)
        mpfr_printf("%5d, %RNe\n", i, MPFR_PTR_BNC_CUSTOM_INDEX(array_top, i, size_mpfr));
    
}

// ---------------------------------------------------
// ------------ MPFRVector and MPFRMatrix ------------
// ------------   2019-08-18 by T.Kouya   ------------
// ---------------------------------------------------

// MPFRVector in BNCpack
typedef struct {
	unsigned long prec;
	mpfr_ptr element;
	long int dim;
	size_t size_of_element; // size of one element 
} mpfrvector;

// MPFRVector is pointer for mpfrvector 
typedef mpfrvector *MPFRVector;

// initialize
MPFRVector init_mpfrvector(long int dim)
{
    MPFRVector ret = NULL;

    ret = (MPFRVector)malloc(sizeof(mpfrvector));

//    ret->prec = prec;
    ret->prec = mpfr_get_default_prec();
    ret->dim = dim;

    ret->element = bnc_custom_init2_mpfr_t_array(dim, ret->prec);

    if(ret->element == NULL)
    {
        free(ret);
        return NULL;
    }

    // set zeros
    bnc_custom_set_array_ui(ret->element, dim, ret->prec, 0UL, MPFR_RNDN);

	// set size
	ret->size_of_element = size_of_mpfr_t(ret->prec);

    return ret;
}

// initialize as prec bits
MPFRVector init2_mpfrvector(long int dim, unsigned long prec)
{
    MPFRVector ret = NULL;

    ret = (MPFRVector)malloc(sizeof(mpfrvector));

    ret->prec = prec;
    ret->dim = dim;

    ret->element = bnc_custom_init2_mpfr_t_array(dim, prec);

    if(ret->element == NULL)
    {
        free(ret);
        return NULL;
    }

    // set zeros
    bnc_custom_set_array_ui(ret->element, dim, prec, 0UL, MPFR_RNDN);

	// set size
	ret->size_of_element = size_of_mpfr_t(ret->prec);

    return ret;
}

// free
void free_mpfrvector(MPFRVector vec)
{
    free(vec->element);
    free(vec);
}

static inline mpfr_ptr get_mpfrvector_i(MPFRVector vec, long int index)
{
//	return MPFR_PTR_BNC_CUSTOM_INDEX(vec->element, index, size_of_mpfr_t(vec->prec));
	return MPFR_PTR_BNC_CUSTOM_INDEX(vec->element, index, vec->size_of_element);
}

static inline void set_mpfrvector_i(MPFRVector vec, long int index, mpfr_t val, mpfr_rnd_t rmode)
{
//    mpfr_set(MPFR_PTR_BNC_CUSTOM_INDEX(vec, index, size_of_mpfr_t(vec->prec)), val, rmode);
	mpfr_set(MPFR_PTR_BNC_CUSTOM_INDEX(vec, index, vec->size_of_element), val, rmode);
}

static inline void set_mpfrvector_i_ui(MPFRVector vec, long int index, unsigned long val, mpfr_rnd_t rmode)
{
//    mpfr_set_ui(MPFR_PTR_BNC_CUSTOM_INDEX(vec, index, size_of_mpfr_t(vec->prec)), val, rmode);
	mpfr_set_ui(MPFR_PTR_BNC_CUSTOM_INDEX(vec, index, vec->size_of_element), val, rmode);
}

// MPFRMatrix in BNCpack
typedef struct {
	unsigned long prec;
	mpfr_ptr element;
	long int row_dim, col_dim;
	void *element_block; // mantissa block
	size_t size_of_element; // size of one element 
} mpfrmatrix;

// MPFRMatrix is pointer for mpfrmatrix
typedef mpfrmatrix *MPFRMatrix;

// Row major & zero index

// initialize
MPFRMatrix init_mpfrmatrix(long int row_dim, long int col_dim)
{
    MPFRMatrix ret = NULL;

    ret = (MPFRMatrix)malloc(sizeof(mpfrmatrix));

//    ret->prec = prec;
    ret->prec = mpfr_get_default_prec();
    ret->row_dim = row_dim;
    ret->col_dim = col_dim;

    ret->element = bnc_custom_init2_mpfr_t_array(row_dim * col_dim, ret->prec);

    if(ret->element == NULL)
    {
        free(ret);
        return NULL;
    }

    // set zeros
    bnc_custom_set_array_ui(ret->element, row_dim * col_dim, ret->prec, 0UL, MPFR_RNDN);

	// set size
	ret->size_of_element = size_of_mpfr_t(ret->prec);

    return ret;
}

// initialize as prec bits
MPFRMatrix init2_mpfrmatrix(long int row_dim, long int col_dim, unsigned long prec)
{
    MPFRMatrix ret = NULL;

    ret = (MPFRMatrix)malloc(sizeof(mpfrmatrix));

    ret->prec = prec;
    ret->row_dim = row_dim;
    ret->col_dim = col_dim;

    ret->element = bnc_custom_init2_mpfr_t_array(row_dim * col_dim, prec);

    if(ret->element == NULL)
    {
        free(ret);
        return NULL;
    }

    // set zeros
    bnc_custom_set_array_ui(ret->element, row_dim * col_dim, prec, 0UL, MPFR_RNDN);

	// set size
	ret->size_of_element = size_of_mpfr_t(ret->prec);

    return ret;
}

// free
void free_mpfrmatrix(MPFRMatrix mat)
{
    free(mat->element);
    free(mat);
}

// Row major
//#define MPFR_PTR_ROW_MAJOR_IJ(mat, row_index, col_index, size_mpfr) MPFR_PTR_BNC_CUSTOM_INDEX((mat)->element, (row_index) * ((mat)->col_dim) + (col_index), (size_mpfr))
#define MPFR_PTR_ROW_MAJOR_IJ(mat, row_index, col_index) MPFR_PTR_BNC_CUSTOM_INDEX((mat)->element, (row_index) * ((mat)->col_dim) + (col_index), ((mat)->size_of_element))
// Column major
//#define MPFR_PTR_COL_MAJOR_IJ(mat, row_index, col_index, size_mpfr) MPFR_PTR_BNC_CUSTOM_INDEX((mat)->element, (row_index) + ((mat)->row_dim) * (col_index), (size_mpfr))
#define MPFR_PTR_COL_MAJOR_IJ(mat, row_index, col_index) MPFR_PTR_BNC_CUSTOM_INDEX((mat)->element, (row_index) + ((mat)->row_dim) * (col_index), ((mat)->size_of_element))

static inline mpfr_ptr get_mpfrmatrix_ij(MPFRMatrix mat, long int i, long int j)
{
#ifdef USE_BNC_COL_MAJOR
//    return MPFR_PTR_COL_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec));
    return MPFR_PTR_COL_MAJOR_IJ(mat, i, j);
#else  // USE_BNC_COL_MAJOR
//    return MPFR_PTR_ROW_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec));
    return MPFR_PTR_ROW_MAJOR_IJ(mat, i, j);
#endif // USE_BNC_COL_MAJOR
}

static inline void set_mpfrmatrix_ij(MPFRMatrix mat, long int i, long int j, mpfr_t val, mpfr_rnd_t rmode)
{
#ifdef USE_BNC_COL_MAJOR
//    mpfr_set(MPFR_PTR_COL_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec)), val, rmode);
    mpfr_set(MPFR_PTR_COL_MAJOR_IJ(mat, i, j), val, rmode);
#else  // USE_BNC_COL_MAJOR
//    mpfr_set(MPFR_PTR_ROW_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec)), val, rmode);
    mpfr_set(MPFR_PTR_ROW_MAJOR_IJ(mat, i, j), val, rmode);
#endif // USE_BNC_COL_MAJOR
}

static inline void set_mpfrmatrix_ij_ui(MPFRMatrix mat, long int i, long int j, unsigned long val, mpfr_rnd_t rmode)
{
#ifdef USE_BNC_COL_MAJOR
//   mpfr_set_ui(MPFR_PTR_COL_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec)), val, rmode);
   mpfr_set_ui(MPFR_PTR_COL_MAJOR_IJ(mat, i, j), val, rmode);
#else  // USE_BNC_COL_MAJOR
//   mpfr_set_ui(MPFR_PTR_ROW_MAJOR_IJ(mat, i, j, size_of_mpfr_t(mat->prec)), val, rmode);
   mpfr_set_ui(MPFR_PTR_ROW_MAJOR_IJ(mat, i, j), val, rmode);
#endif // USE_BNC_COL_MAJOR
}

// Matrix multiplication
// ret := mat_a * mat_b
void mul_mpfrmatrix_simple(MPFRMatrix ret, MPFRMatrix mat_a, MPFRMatrix mat_b, mpfr_rnd_t rmode)
{
    long int i, j, k;
    long row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = mat_a->col_dim;
    mpfr_t tmp, tmp2;

    // mat_a->col_dim != mat_b->row_dim ?
    if(mid_dim != mat_b->row_dim)
    {
        fprintf(stderr, "ERROR: mid_dim is illegal(mat_a's col %ld != mat_b's row %ld) in mul_mpfrmatrix_simple!\n", mid_dim, mat_b->row_dim);
        return;
    }
    // ret->row_dim != mat_a->row_dim ?
    if(mid_dim != mat_b->row_dim)
    {
        fprintf(stderr, "ERROR: row_dim is illegal(ret's row %ld != mat_a's row %ld) in mul_mpfrmatrix_simple!\n", row_dim, mat_a->row_dim);
        return;
    }
    // ret->col_dim != mat_b->col_dim ?
    if(mid_dim != mat_b->row_dim)
    {
        fprintf(stderr, "ERROR: col_dim is illegal(ret's col %ld != mat_b's col %ld) in mul_mpfrmatrix_simple!\n", col_dim, mat_b->col_dim);
        return;
    }

	mpfr_init2(tmp, ret->prec);
	mpfr_init2(tmp2, ret->prec);

    // main triple loop
    for(i = 0; i < row_dim; i++)
    {
        for(j = 0; j < col_dim; j++)
        {
//#if 0
			//set_mpfrmatrix_ij_ui(ret, i, j, 0UL, rmode);
			mpfr_set_ui(tmp, 0UL, rmode);
            for(k = 0; k < mid_dim; k++)
            {
                mpfr_fma(
                         //get_mpfrmatrix_ij(ret, i, j),
                         tmp,
                         get_mpfrmatrix_ij(mat_a, i, k),
                         get_mpfrmatrix_ij(mat_b, k, j),
                         //get_mpfrmatrix_ij(ret, i, j),
                         tmp,
                         rmode
                );
            }
            set_mpfrmatrix_ij(ret, i, j, tmp, rmode);
//#endif // 0
#if 0
			mpfr_set_ui(tmp2, 0UL, rmode);
			for(k = 0; k < mid_dim; k++)
			{
//				mpfr_mul(tmp , get_mpfrmatrix_ij(mat_a, i, k), get_mpfrmatrix_ij(mat_b, k, j), rmode);
				mpfr_mul(tmp , 
					//MPFR_PTR_ROW_MAJOR_IJ(mat_a, i, k, size_of_mpfr_t(mat_a->prec)),
					//MPFR_PTR_ROW_MAJOR_IJ(mat_b, k, j, size_of_mpfr_t(mat_b->prec)),
					MPFR_PTR_ROW_MAJOR_IJ(mat_a, i, k),
					MPFR_PTR_ROW_MAJOR_IJ(mat_b, k, j),
					rmode
				);
				mpfr_add(tmp2, tmp2, tmp, rmode);
			}
            set_mpfrmatrix_ij(ret, i, j, tmp2, rmode);
#endif // 0
        }
    }

	mpfr_clear(tmp);
	mpfr_clear(tmp2);
}


#ifdef _OPENMP

#include "bncomp.h"

/* c = a * b */
void _bncomp_mul_mpfrmatrix(MPFRMatrix c, MPFRMatrix a, MPFRMatrix b, mpfr_rnd_t rmode)
{
	long int i, j, k, row_dim, col_dim, mid_dim;
	int thread_index, thread_num;
	mpfr_t tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	unsigned int prec;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_mpfrmatrix\n");
		return;
	}

	prec = c->prec;

	row_dim = c->row_dim;
	col_dim = c->col_dim;
	mid_dim = a->col_dim;

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpfr_init2(tmp[thread_index], prec);
		mpfr_init2(tmp1[thread_index], prec);
	}

	#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			mpfr_set_ui(tmp[thread_index], 0UL, rmode);
			for(k = 0; k < mid_dim; k++)
			{
				mpfr_fma(tmp[thread_index], get_mpfrmatrix_ij(a, i, k), get_mpfrmatrix_ij(b, k, j), tmp[thread_index], rmode);
			}
			set_mpfrmatrix_ij(c, i, j, tmp[thread_index], rmode);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpfr_clear(tmp[thread_index]);
		mpfr_clear(tmp1[thread_index]);
	}
}
#endif // _OPENMP

void print_mpfrmatrix(MPFRMatrix mat)
{
    bnc_custom_print_full_array(mat->element, mat->row_dim * mat->col_dim, mat->prec);
}

int main(int argc, char *argv[])
{
	long int dim, i, j;
	unsigned int prec;
	mpf_ptr mpf_a;
	mpfr_t mpfr_a, mpfr_tmp[2];
	mpfr_ptr pmpfr_a, mpfr_element;
	double stime, etime;

	if(argc <= 1)
	{
		printf("Usage: %s [prec in bits]\n", argv[0]);
		return 0;
	}

	prec = atoi(argv[1]);

	dim = 10;
	if(argc >= 3)
	{
		dim = atoi(argv[2]);
		goto matmul;
	}

/*
//	printf(" mpf_t(%u bits) is %d bytes\n", prec, size_of_mpf_t(prec));

	// mpf_t in GMP
	mpf_a = bnc_custom_init2_mpf_t(prec);

	mpf_set_ui(mpf_a, 2UL); mpf_sqrt(mpf_a, mpf_a);
	gmp_printf("mpf_a = %50.45Fe\n", mpf_a);

	free(mpf_a);
*/
	// mpfr_t in MPFR
	printf("mpfr_t(%d bits) is %ld bytes\n", prec, size_of_mpfr_t(prec));
	mpfr_init2(mpfr_a, prec);
	pmpfr_a = bnc_custom_init2_mpfr_t(prec);
//	mpfr_init2(mpfr_a, prec);
	mpfr_printf("mpfr_a  = %RNe\n", mpfr_a);
	mpfr_printf("pmpfr_a = %RNe\n", pmpfr_a);

	mpfr_set_ui(pmpfr_a, 2UL, MPFR_RNDN);
	mpfr_sqrt(pmpfr_a, pmpfr_a, MPFR_RNDN);
	mpfr_sqrt_ui(mpfr_a, 2UL, MPFR_RNDN);
	mpfr_printf("mpfr_a  = %RNe\n", mpfr_a);
//	mpfr_out_str(stdout, 10, 0, pmpfr_a, MPFR_RNDN);
	mpfr_printf("pmpfr_a = %RNe\n", pmpfr_a);
	mpfr_sub(mpfr_a, mpfr_a, pmpfr_a, MPFR_RNDN);
	mpfr_printf("mpfr_a - pmpfr_a = %RNe\n", mpfr_a);

	//free((unsigned char *)pmpfr_a);
	//mpfr_clear(pmpfr_a);
	//return 0;

//	free(pmpfr_a);
//	mpfr_clear(mpfr_a);

    // mpfr_element
	dim = 10;

	mpfr_element = bnc_custom_init2_mpfr_t_array(dim, prec);

	bnc_custom_set_array(mpfr_element, dim, prec, pmpfr_a, MPFR_RNDN);
	
	bnc_custom_print_full_array(mpfr_element, dim, prec);
	
	free(mpfr_element);
	free((void *)pmpfr_a);

matmul:

	// matrix
	mpfr_init2(mpfr_tmp[0], prec);
	mpfr_init2(mpfr_tmp[1], prec);
	mpfr_init2(mpfr_a, prec);
	
	printf("%d bits MPFRMatrix(%ld, %ld):\n", prec, dim, dim);
	MPFRMatrix mpfr_mat[3];

	mpfr_mat[0] = init2_mpfrmatrix(dim, dim, prec);
	mpfr_mat[1] = init2_mpfrmatrix(dim, dim, prec);
	mpfr_mat[2] = init2_mpfrmatrix(dim, dim, prec);

	mpfr_sqrt_ui(mpfr_tmp[0], 3UL, MPFR_RNDN);
	mpfr_sqrt_ui(mpfr_tmp[1], 5UL, MPFR_RNDN);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			mpfr_mul_ui(mpfr_a, mpfr_tmp[0], (unsigned long)(i + j + 1), MPFR_RNDN);
//			set_mpfrmatrix_ij_ui(mpfr_mat[0], i, j, (unsigned long)(i + j + 1), MPFR_RNDN);
			set_mpfrmatrix_ij(mpfr_mat[0], i, j, mpfr_a, MPFR_RNDN);

			mpfr_mul_ui(mpfr_a, mpfr_tmp[1], (unsigned long)(dim - i), MPFR_RNDN);
//			set_mpfrmatrix_ij_ui(mpfr_mat[1], i, j, (unsigned long)(dim - i), MPFR_RNDN);
			set_mpfrmatrix_ij(mpfr_mat[1], i, j, mpfr_a, MPFR_RNDN);
		}
	}

	stime = get_real_secv();
#ifdef _OPENMP
	_bncomp_mul_mpfrmatrix(mpfr_mat[2], mpfr_mat[0], mpfr_mat[1], MPFR_RNDN);
#else // _OPENMP
	mul_mpfrmatrix_simple(mpfr_mat[2], mpfr_mat[0], mpfr_mat[1], MPFR_RNDN);
#endif // _OPENMP
	etime = get_real_secv() - stime;

	printf("%ld * %ld matrix %d bits mul = %f seconds\n", dim, dim, prec, etime);
//	print_mpfrmatrix(mpfr_mat[2]);

	free_mpfrmatrix(mpfr_mat[0]);
	free_mpfrmatrix(mpfr_mat[1]);
	free_mpfrmatrix(mpfr_mat[2]);

	mpfr_clear(mpfr_tmp[0]);
	mpfr_clear(mpfr_tmp[1]);
	mpfr_clear(mpfr_a);

	return 0;
}
