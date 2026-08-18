// ------------------------
// Error Freee Trans. with double
// ------------------------
#ifndef __BNCAVX_DEFT_H
#define __BNCAVX_DEFT_H

/************ Basic Functions ************ start **/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
#if defined(__AVX2__)
static inline __m256d _bncavx2_dquick_two_sum(__m256d a, __m256d b, __m256d *err)
{
	__m256d s;

	//s = a + b;
    s = _mm256_add_pd(a, b);

	//*err = b - (s - a);
    *err = _mm256_sub_pd(b, _mm256_sub_pd(s, a));

	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline __m256d _bncavx2_dquick_two_diff(__m256d a, __m256d b, __m256d *err)
{
	__m256d s;

	//s = a - b;
    s = _mm256_sub_pd(a, b);

	//*err = (a - s) - b;
    *err = _mm256_sub_pd(_mm256_sub_pd(a, s), b);

	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline __m256d _bncavx2_dtwo_sum(__m256d a, __m256d b, __m256d *err)
{
	__m256d s, bb;

	//s = a + b;
    s = _mm256_add_pd(a, b);

	//bb = s - a;
    bb = _mm256_sub_pd(s, a);

	//*err = (a - (s - bb)) + (b - bb);
    *err = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb)), _mm256_sub_pd(b, bb));

	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline __m256d _bncavx2_dtwo_diff(__m256d a, __m256d b, __m256d *err)
{
	__m256d s, bb;

	//s = a - b;
    s = _mm256_sub_pd(a, b);

	//bb = s - a;
    bb = _mm256_sub_pd(s, a);

	//*err = (a - (s - bb)) - (b + bb);
    *err = _mm256_sub_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb)), _mm256_add_pd(b, bb));

	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline __m256d _bncavx2_dtwo_prod(__m256d a, __m256d b, __m256d *err)
{
	__m256d p;

	//p = a * b;
    p = _mm256_mul_pd(a, b);

    // return a * b - c
    //#define QD_FMS(a, b, c) DFMA((a), (b), (-(c)))
	//*err = QD_FMS(a, b, p);
    *err = _mm256_fmsub_pd(a, b, p);

	return p;
}
#endif //defined(__AVX2__)

#endif //ifndef __BNCAVX_DEFT_H
