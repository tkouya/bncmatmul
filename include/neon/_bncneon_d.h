#ifndef __BNCNEON_D_H
#define __BNCNEON_D_H

#if defined(__ARM_NEON) // __ARM_NEON
// val2 := max(val2, 0 - val2)
static inline float64x2_t _bncneon_fabs(float64x2_t val2)
{
	return vabsq_f64(val2); 
    //returnn_mm256_max_pd(val4, _mm256_sub_pd(_mm256_setzero_pd(), val4));
}

// return 0 - a
static inline float64x2_t _bncneon_dneg(float64x2_t a)
{
    //return _mm256_sub_pd(_mm256_setzero_pd(), a);
    return vnegq_f64(a);
}

// ret := ret4[0] + ret4[1]
static double _bncneon_dsum128(float64x2_t ret2)
{
    //static double ret2_i[2];

    // ret4_i := ret4
    //_mm256_storeu_pd(ret4_i, ret4);

    //return ret4_i[0] + ret4_i[1] + ret4_i[2] + ret4_i[3];
    //return vgetq_lane_f64(ret2, 0) + vgetq_lane_f64(ret2, 1);
    return vaddvq_f64(ret2);
}

// NEON 128bits: ret := a * b + c
static void _bncneon_dfma(double ret[], double a[], double b[], double c[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b, in_c;
    //double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
        in_c = vld1q_f64(&c[index]); //_mm256_load_pd(&c[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vfmaq_f64(in_c, in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        //_mm256_store_pd(&ret[index], in_ret);
        vst1q_f64(&ret[index], in_ret);
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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);
            in_c = vsetq_lane_f64(c[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vfmaq_f64(in_c, in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
            ret[index] = vgetq_lane_f64(in_ret, 0);
        }
    }
//    printf("\n");
}

// AVX 256bits: ret := a * b
static void _bncneon_dmul(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b, in_c;
    //double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vmulq_f64(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        //_mm256_store_pd(&ret[index], in_ret);
        vst1q_f64(&ret[index], in_ret);
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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vmulq_f64(in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
            ret[index] = vgetq_lane_f64(in_ret, 0);
        }
    }
//    printf("\n");
}

// AVX 256bits: ret := a / b
static void _bncneon_ddiv(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b, in_c;
    //double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vdivq_f64(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        //_mm256_store_pd(&ret[index], in_ret);
        vst1q_f64(&ret[index], in_ret);
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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vdivq_f64(in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
            ret[index] = vgetq_lane_f64(in_ret, 0);
        }
    }
//    printf("\n");
}

// AVX 256bits: ret := a + b
static void _bncneon_dadd(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b, in_c;
    //double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vaddq_f64(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        //_mm256_store_pd(&ret[index], in_ret);
        vst1q_f64(&ret[index], in_ret);
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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vaddq_f64(in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
            ret[index] = vgetq_lane_f64(in_ret, 0);
        }
    }
//    printf("\n");
}

// AVX 256bits: ret := a - b
static void _bncneon_dsub(double ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b, in_c;
    //double in_ret_rem[4];

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vsubq_f64(in_a, in_b);

        // store in_ret to ret
//        _mm256_storeu_pd(&ret[index], in_ret);
        //_mm256_store_pd(&ret[index], in_ret);
        vst1q_f64(&ret[index], in_ret);
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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vsubq_f64(in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
            ret[index] = vgetq_lane_f64(in_ret, 0);
        }
    }
//    printf("\n");
}

// AVX2 256bits: ret := dotp(a, b)
static double _bncneon_ddotp(double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 2; //4;
    float64x2_t in_ret, in_a, in_b;
    double ret = 0.0;

    //printf("dim = %d\n", dim);
    if(dim <= 0) return ret;

    div = dim / unit;
    rem = dim % unit;

//    printf("dim, div, rem = %d, %d, %d, %d\n", dim, div, rem, unit);

    // main
    //for(i = 0; i < div; i++)
    in_ret = vdupq_n_f64(0.0); // 初期化
    for(index = 0; index < dim; index += unit)
    {
        in_a = vld1q_f64(&a[index]); //_mm256_load_pd(&a[index]);
        in_b = vld1q_f64(&b[index]); //_mm256_load_pd(&b[index]);
 
        // in_ret := in_a * in_b + in_c
        //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
        in_ret = vmlaq_f64(in_ret, in_a, in_b);

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
            //in_a = _mm256_set_pd(0.0, 0.0, 0.0, a[index]);
            //in_b = _mm256_set_pd(0.0, 0.0, 0.0, b[index]);
            //in_c = _mm256_set_pd(0.0, 0.0, 0.0, c[index]);
            in_a = vsetq_lane_f64(a[index], vdupq_n_f64(0.0), 1);
            in_b = vsetq_lane_f64(b[index], vdupq_n_f64(0.0), 1);

            // in_ret := in_a * in_b + in_c
            //in_ret = _mm256_fmadd_pd(in_a, in_b, in_c);
            in_ret = vmlaq_f64(in_ret, in_a, in_b);

            // store in_ret to ret
            //_mm256_storeu_pd(in_ret_rem, in_ret);
            //ret[index] = in_ret_rem[0];
        }
    }
    ret = _bncneon_dsum128(in_ret);
    //    printf("\n");

    return ret; 
}
#endif //if defined(__ARM_NEON)

#endif //ifndef __BNCNEON_D_H
