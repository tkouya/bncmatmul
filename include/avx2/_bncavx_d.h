#ifndef __BNCAVX_D_H
#define __BNCAVX_D_H

//https://stackoverflow.com/questions/32408665/fastest-way-to-compute-absolute-value-using-sse
#if defined(__AVX2__) // && !defined(__AVX512F__) // __AVX2__
// val4 := max(val4, 0 - val4)
static inline __m256d _bncavx2_fabs(__m256d val4)
{
	return _mm256_max_pd(val4, _mm256_sub_pd(_mm256_setzero_pd(), val4));
}

// return 0 - a
static inline __m256d _bncavx2_dneg(__m256d a)
{
    return _mm256_sub_pd(_mm256_setzero_pd(), a);
}
#endif // defined(__AVX2__)

#if defined(__AVX512F__) // __AVX512F__
// val8 := max(val8, 0 - val8)
static inline __m512d _bncavx512_fabs(__m512d val8)
{
	return _mm512_max_pd(val8, _mm512_sub_pd(_mm512_setzero_pd(), val8));
}

// return 0 - a
static inline __m512d _bncavx512_dneg(__m512d a)
{
    return _mm512_sub_pd(_mm512_setzero_pd(), a);
}
#endif // __AVX512F__

#if defined(__AVX2__) // __AVX2__

// ret := ret4[0] + ret4[1] + ret4[2] + ret4[3]
static double _bncavx2_dsum256d(__m256d ret4)
{
    static double ret4_i[4];

    // ret4_i := ret4
    _mm256_storeu_pd(ret4_i, ret4);

    return ret4_i[0] + ret4_i[1] + ret4_i[2] + ret4_i[3];
}

// AVX2 256bits: ret := a * b + c
static void _bncavx2_dfma(double ret[], double a[], double b[], double c[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, in_c;
    double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm256_loadu_pd(&a[index]);
        in_b = _mm256_loadu_pd(&b[index]);
        in_c = _mm256_loadu_pd(&c[index]);
*/
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);
        in_c = _mm256_load_pd(&c[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        _mm256_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);
            in_c = _mm256_set_pd(0.0, 0.0, c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_pd(0.0, c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
    }
//    printf("\n");
}
#endif // __AVX2__

#if defined(__AVX512F__) // __AVX512F__
// ret := ret4[0] + ret4[1] + ret4[2] + ret4[3]
static double _bncavx2_dsum512d(__m512d ret8)
{
    static double ret8_i[8];

    // ret4_i := ret4
    _mm512_storeu_pd(ret8_i, ret8);

    return ret8_i[0] + ret8_i[1] + ret8_i[2] + ret8_i[3] + ret8_i[4] + ret8_i[5] + ret8_i[6] + ret8_i[7];
}

// AVX-512 512bits: ret := a * b + c
static void _bncavx512_dfma(double ret[], double a[], double b[], double c[], int dim)
{
    int i, index, div, rem, unit = 8;
    __m512d in_ret, in_a, in_b, in_c;
    double in_ret_rem[8];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm512_loadu_pd(&a[index]);
        in_b = _mm512_loadu_pd(&b[index]);
        in_c = _mm512_loadu_pd(&c[index]);
*/
        in_a = _mm512_load_pd(&a[index]);
        in_b = _mm512_load_pd(&b[index]);
        in_c = _mm512_load_pd(&c[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

        // store in_ret to ret
//        _mm512_storeu_pd(&ret[index], in_ret);
        _mm512_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, b[index + 2], b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
        else if(rem == 4)
        {
            in_a = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, 0.0, 0.0, c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
        }
        else if(rem == 5)
        {
            in_a = _mm512_set_pd(0.0, 0.0, 0.0, a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, 0.0, b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, 0.0, c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
        }
        else if(rem == 6)
        {
            in_a = _mm512_set_pd(0.0, 0.0, a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, 0.0, b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, 0.0, c[index + 5], c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
            ret[index + 5] = in_ret_rem[5];
        }
        else if(rem == 7)
        {
            in_a = _mm512_set_pd(0.0, a[index + 6], a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm512_set_pd(0.0, b[index + 6], b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm512_set_pd(0.0, c[index + 6], c[index + 5], c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm512_fmadd_pd(in_a, in_b, in_c);

            // store in_ret to ret
            _mm512_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
            ret[index + 5] = in_ret_rem[5];
            ret[index + 6] = in_ret_rem[6];
        }
    }
//    printf("\n");
}
#endif // __AVX512F__

#if defined(__AVX2__)
// AVX 256bits: ret := a * b
static void _bncavx2_dmul(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b;
    double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm256_loadu_pd(&a[index]);
        in_b = _mm256_loadu_pd(&b[index]);
*/
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_mul_pd(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        _mm256_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
    }
//    printf("\n");
}
#endif //defined(__AVX2__)

#if defined(__AVX2__)
// AVX 256bits: ret := a / b
static void _bncavx2_ddiv(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b;
    double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm256_loadu_pd(&a[index]);
        in_b = _mm256_loadu_pd(&b[index]);
*/
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_div_pd(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        _mm256_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_div_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_div_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_div_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
    }
//    printf("\n");
}
#endif // defined(__AVX2__)

#if defined(__AVX2__)
// AVX 256bits: ret := a + b
static void _bncavx2_dadd(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b;
    double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm256_loadu_pd(&a[index]);
        in_b = _mm256_loadu_pd(&b[index]);
*/
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_add_pd(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        _mm256_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_add_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_add_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_add_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
    }
//    printf("\n");
}
#endif //defined(__AVX2__)

#if defined(__AVX2__)
// AVX 256bits: ret := a - b
static void _bncavx2_dsub(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b;
    double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
    //    index = i * unit;
        // load a, b, c to in_a, in_b, in_c
/*
        in_a = _mm256_loadu_pd(&a[index]);
        in_b = _mm256_loadu_pd(&b[index]);
*/
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_sub_pd(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        _mm256_store_pd(&ret[index], in_ret);
//        printf("%d ", index);
    }
   //printf("set in_a, in_b, in_c\n");

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_sub_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_sub_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_sub_pd(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_pd(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
    }
//    printf("\n");
}
#endif //defined(__AVX2__)

#if defined(__AVX2__)
// AVX2 256bits: ret := dotp(a, b)
static double _bncavx2_ddotp(double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b;
    double in_ret_rem[4] = {0.0, 0.0, 0.0, 0.0};

    // in_ret = {0, 0, 0, 0};
    in_ret = _mm256_setzero_pd();

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(i = 0; i < div; i++)
    {
        index = i * unit;
            // load a, b, c to in_a, in_b, in_c
    /*
            in_a = _mm256_loadu_pd(&a[index]);
            in_b = _mm256_loadu_pd(&b[index]);
    */
            in_a = _mm256_load_pd(&a[index]);
            in_b = _mm256_load_pd(&b[index]);
    
            // in_ret := in_a * in_b + in_ret
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_ret);

//            printf("1) %d ", index);
    }

    // rem != 0
    if(rem > 0)
    {
        index = div * unit;
//        printf("2) %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_ret);
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_ret);
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_pd(in_a, in_b, in_ret);

        }
    }

    // store in_ret to ret
    _mm256_storeu_pd(in_ret_rem, in_ret);
//    printf("in_ret_rem = %10.3e %10.3e %10.3e %10.3e\n", in_ret_rem[0], in_ret_rem[1], in_ret_rem[2], in_ret_rem[3]);

    return in_ret_rem[0] + in_ret_rem[1] + in_ret_rem[2] + in_ret_rem[3];
}
#endif //if defined(__AVX2__)

#endif //ifndef __BNCAVX_D_H
