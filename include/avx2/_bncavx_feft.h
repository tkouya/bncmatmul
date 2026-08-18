#ifndef __BNCAVX_FEFT_H
#define __BNCAVX_FEFT_H

/************ Basic Functions ************ start **/
#if defined(__AVX2__)
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
static inline __m256 _bncavx2_fquick_two_sum(__m256 a, __m256 b, __m256 *err)
{
	__m256 s;

	//s = a + b;
    s = _mm256_add_ps(a, b);

	//*err = b - (s - a);
    *err = _mm256_sub_ps(b, _mm256_sub_ps(s, a));

	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline __m256 _bncavx2_fquick_two_diff(__m256 a, __m256 b, __m256 *err)
{
	__m256 s;

	//s = a - b;
    s = _mm256_sub_ps(a, b);

	//*err = (a - s) - b;
    *err = _mm256_sub_ps(_mm256_sub_ps(a, s), b);

	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline __m256 _bncavx2_ftwo_sum(__m256 a, __m256 b, __m256 *err)
{
	__m256 s, bb;

	//s = a + b;
    s = _mm256_add_ps(a, b);

	//bb = s - a;
    bb = _mm256_sub_ps(s, a);

	//*err = (a - (s - bb)) + (b - bb);
    *err = _mm256_add_ps(_mm256_sub_ps(a, _mm256_sub_ps(s, bb)), _mm256_sub_ps(b, bb));

	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline __m256 _bncavx2_ftwo_diff(__m256 a, __m256 b, __m256 *err)
{
	__m256 s, bb;

	//s = a - b;
    s = _mm256_sub_ps(a, b);

	//bb = s - a;
    bb = _mm256_sub_ps(s, a);

	//*err = (a - (s - bb)) - (b + bb);
    *err = _mm256_sub_ps(_mm256_sub_ps(a, _mm256_sub_ps(s, bb)), _mm256_add_ps(b, bb));

	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline __m256 _bncavx2_ftwo_prod(__m256 a, __m256 b, __m256 *err)
{
	__m256 p;

	//p = a * b;
    p = _mm256_mul_ps(a, b);

    // return a * b - c
    //#define QD_FMS(a, b, c) DFMA((a), (b), (-(c)))
	//*err = QD_FMS(a, b, p);
    *err = _mm256_fmsub_ps(a, b, p);

	return p;
}
#endif //defined(__AVX2__)

#endif //ifndef __BNCAVX_FEFT_H
