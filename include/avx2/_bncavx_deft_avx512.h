// ------------------------
// Error Freee Trans. with double
// AVX-512F
// ------------------------
#ifndef __BNCAVX_DEFT_AVX512_H
#define __BNCAVX_DEFT_AVX512_H

/************ Basic Functions ************ start **/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
#if defined(__AVX512F__)
static inline __m512d _bncavx512_dquick_two_sum(__m512d a, __m512d b, __m512d *err)
{
	__m512d s;

	//s = a + b;
    s = _mm512_add_pd(a, b);

	//*err = b - (s - a);
    *err = _mm512_sub_pd(b, _mm512_sub_pd(s, a));

	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline __m512d _bncavx512_dquick_two_diff(__m512d a, __m512d b, __m512d *err)
{
	__m512d s;

	//s = a - b;
    s = _mm512_sub_pd(a, b);

	//*err = (a - s) - b;
    *err = _mm512_sub_pd(_mm512_sub_pd(a, s), b);

	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline __m512d _bncavx512_dtwo_sum(__m512d a, __m512d b, __m512d *err)
{
	__m512d s, bb;

	//s = a + b;
    s = _mm512_add_pd(a, b);

	//bb = s - a;
    bb = _mm512_sub_pd(s, a);

	//*err = (a - (s - bb)) + (b - bb);
    *err = _mm512_add_pd(_mm512_sub_pd(a, _mm512_sub_pd(s, bb)), _mm512_sub_pd(b, bb));

	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline __m512d _bncavx512_dtwo_diff(__m512d a, __m512d b, __m512d *err)
{
	__m512d s, bb;

	//s = a - b;
    s = _mm512_sub_pd(a, b);

	//bb = s - a;
    bb = _mm512_sub_pd(s, a);

	//*err = (a - (s - bb)) - (b + bb);
    *err = _mm512_sub_pd(_mm512_sub_pd(a, _mm512_sub_pd(s, bb)), _mm512_add_pd(b, bb));

	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline __m512d _bncavx512_dtwo_prod(__m512d a, __m512d b, __m512d *err)
{
	__m512d p;

	//p = a * b;
    p = _mm512_mul_pd(a, b);

    // return a * b - c
    //#define QD_FMS(a, b, c) DFMA((a), (b), (-(c)))
	//*err = QD_FMS(a, b, p);
    *err = _mm512_fmsub_pd(a, b, p);

	return p;
}
#endif //defined(__AVX512F__)

#endif //ifndef __BNCAVX_DEFT_AVX512_H
