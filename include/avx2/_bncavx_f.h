#ifndef __BNCAVX_F_H
#define __BNCAVX_F_H

//https://stackoverflow.com/questions/32408665/fastest-way-to-compute-absolute-value-using-sse
#if defined(__AVX2__) // && !defined(__AVX512F__) // __AVX2__
// val8 := max(val8, 0 - val8)
static inline __m256 _bncavx2_fabsf(__m256 val8)
{
	return _mm256_max_ps(val8, _mm256_sub_ps(_mm256_setzero_ps(), val8));
}

// return 0 - a
static inline __m256 _bncavx2_fneg(__m256 a)
{
    return _mm256_sub_ps(_mm256_setzero_ps(), a);
}
#endif // defined(__AVX2__)

#if defined(__AVX512F__) // __AVX512F__
// val16 := max(val16, 0 - val16)
static inline __m512 _bncavx512_fabsf(__m512 val16)
{
	return _mm512_max_ps(val16, _mm512_sub_ps(_mm512_setzero_ps(), val16));
}

// return 0 - a
static inline __m512 _bncavx512_fneg(__m512 a)
{
    return _mm512_sub_ps(_mm512_setzero_ps(), a);
}
#endif // __AVX512F__

#if defined(__AVX2__) // __AVX2__
// AVX2 256bits: ret := a * b + c
static void _bncavx2_ffma(float ret[], float a[], float b[], float c[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; // 8
    __m256 in_ret, in_a, in_b, in_c;
    float in_ret_rem[_BNC_S_WIDTH];

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
        in_a = _mm256_loadu_ps(&a[index]);
        in_b = _mm256_loadu_ps(&b[index]);
        in_c = _mm256_loadu_ps(&c[index]);
*/
        in_a = _mm256_load_ps(&a[index]);
        in_b = _mm256_load_ps(&b[index]);
        in_c = _mm256_load_ps(&c[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

        // store in_ret to ret
//        _mm256_storeu_ps(&ret[index], in_ret);
        _mm256_store_ps(&ret[index], in_ret);
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
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
        else if(rem == 4)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
        }
        else if(rem == 4)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
        }
        else if(rem == 5)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, 0.0f, c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
        }
        else if(rem == 6)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, 0.0f, c[index + 5], c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
            ret[index + 5] = in_ret_rem[5];
        }
        else if(rem == 7)
        {
            in_a = _mm256_set_ps(0.0f, a[index + 6], a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, b[index + 6], b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);
            in_c = _mm256_set_ps(0.0f, c[index + 6], c[index + 5], c[index + 4], c[index + 3], c[index + 2], c[index + 1], c[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_fmadd_ps(in_a, in_b, in_c);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
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
#endif // __AVX2__


#if defined(__AVX2__)
// AVX2 256bits: ret := a * b
static void _bncavx2_fmul(float ret[], float a[], float b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; // 8
    __m256 in_ret, in_a, in_b;
    float in_ret_rem[_BNC_S_WIDTH];

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
        in_a = _mm256_loadu_ps(&a[index]);
        in_b = _mm256_loadu_ps(&b[index]);
*/
        in_a = _mm256_load_ps(&a[index]);
        in_b = _mm256_load_ps(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        in_ret = _mm256_mul_ps(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_ps(&ret[index], in_ret);
        _mm256_store_ps(&ret[index], in_ret);
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
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index]);

            // in_ret := in_a * in_b
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index] = in_ret_rem[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index + 1], b[index]);

            // in_ret := in_a * in_b
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
        }
        else if(rem == 4)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, b[index + 3], b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
        }
        else if(rem == 4)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, b[index + 3], b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
        }
        else if(rem == 5)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, 0.0f, a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, 0.0f, b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
        }
        else if(rem == 6)
        {
            in_a = _mm256_set_ps(0.0f, 0.0f, a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, 0.0f, b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
            ret[index    ] = in_ret_rem[0];
            ret[index + 1] = in_ret_rem[1];
            ret[index + 2] = in_ret_rem[2];
            ret[index + 3] = in_ret_rem[3];
            ret[index + 4] = in_ret_rem[4];
            ret[index + 5] = in_ret_rem[5];
        }
        else if(rem == 7)
        {
            in_a = _mm256_set_ps(0.0f, a[index + 6], a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0f, b[index + 6], b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            // in_ret := in_a * in_b + in_c
            in_ret = _mm256_mul_ps(in_a, in_b);

            // store in_ret to ret
            _mm256_storeu_ps(in_ret_rem, in_ret);
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
#endif //if defined(__AVX2__)

#endif // ifndef __BNCAVX_F_H
