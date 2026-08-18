// ------------------------
// Error Free Transformations with float
// AVX-512F
// ------------------------
#ifndef __BNCAVX_FEFT_AVX512_H
#define __BNCAVX_FEFT_AVX512_H

/************ Basic Functions ************ start **/
/* Computes fl(a+b) and err(a+b).  Assumes |a| >= |b|. */
#if defined(__AVX512F__)
static inline __m512 _bncavx512_fquick_two_sum(__m512 a, __m512 b, __m512 *err)
{
	__m512 s;

	//s = a + b;
    s = _mm512_add_ps(a, b);

	//*err = b - (s - a);
    *err = _mm512_sub_ps(b, _mm512_sub_ps(s, a));

	return s;
}

/* Computes fl(a-b) and err(a-b).  Assumes |a| >= |b| */
static inline __m512 _bncavx512_fquick_two_diff(__m512 a, __m512 b, __m512 *err)
{
	__m512 s;

	//s = a - b;
    s = _mm512_sub_ps(a, b);

	//*err = (a - s) - b;
    *err = _mm512_sub_ps(_mm512_sub_ps(a, s), b);

	return s;
}

/* Computes fl(a+b) and err(a+b).  */
static inline __m512 _bncavx512_ftwo_sum(__m512 a, __m512 b, __m512 *err)
{
	__m512 s, bb;

	//s = a + b;
    s = _mm512_add_ps(a, b);

	//bb = s - a;
    bb = _mm512_sub_ps(s, a);

	//*err = (a - (s - bb)) + (b - bb);
    *err = _mm512_add_ps(_mm512_sub_ps(a, _mm512_sub_ps(s, bb)), _mm512_sub_ps(b, bb));

	return s;
}

/* Computes fl(a-b) and err(a-b).  */
static inline __m512 _bncavx512_ftwo_diff(__m512 a, __m512 b, __m512 *err)
{
	__m512 s, bb;

	//s = a - b;
    s = _mm512_sub_ps(a, b);

	//bb = s - a;
    bb = _mm512_sub_ps(s, a);

	//*err = (a - (s - bb)) - (b + bb);
    *err = _mm512_sub_ps(_mm512_sub_ps(a, _mm512_sub_ps(s, bb)), _mm512_add_ps(b, bb));

	return s;
}

/* Computes fl(a*b) and err(a*b). */
static inline __m512 _bncavx512_ftwo_prod(__m512 a, __m512 b, __m512 *err)
{
	__m512 p;

	//p = a * b;
    p = _mm512_mul_ps(a, b);

    // return a * b - c
    //#define QS_FMS(a, b, c) SFMA((a), (b), (-(c)))
	//*err = QS_FMS(a, b, p);
    *err = _mm512_fmsub_ps(a, b, p);

	return p;
}
#endif //defined(__AVX512F__)

#endif //ifndef __BNCAVX_FEFT_AVX512_H
