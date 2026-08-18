#ifndef __BNC_SORT_H__
#define __BNC_SORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>

static void _bncavx2_print_m256d_array(__m256d data[], int num)
{
    int i, j;

    printf("      ");
    for(i = 0; i < num; i++)
        printf(" d[%3d] ", i);
    printf("\n");

    for(j = 0; j < 4; j++)
    {
        printf("[%d] = ", j);
        for(i = 0; i < num; i++)
            printf("%5.1e ", data[i][j]);
        printf("\n");
    }
}

// http://www.cs.cmu.edu/afs/cs/academic/class/15213-s19/www/lectures613/04-simd.pdf
// data0[i] >= data1[i], i = 0, 1, 2, 3
static inline void _bncavx2_swap_max(__m256d *data0, __m256d *data1)
{
    __m256d checked_flag;
    __m256d new_data[2], tmp[2];

    // maximum is stored in new_data[0]
    checked_flag = _mm256_cmp_pd(*data0, *data1, _CMP_GE_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[0] = _mm256_add_pd(tmp[0], tmp[1]);

    // minimum is stored in new_data[1]
    checked_flag = _mm256_cmp_pd(*data0, *data1, _CMP_LT_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[1] = _mm256_add_pd(tmp[0], tmp[1]);

//    printf("->\n");
//    _bncavx2_print_m256d_array(new_data, 2);

    // swap data0 and data1
    *data0 = new_data[0];
    *data1 = new_data[1];    
}

// data0[i] <= data1[i], i = 0, 1, 2, 3
static inline void _bncavx2_swap_min(__m256d *data0, __m256d *data1)
{
    __m256d checked_flag;
    __m256d new_data[2], tmp[2];

    // maximum is stored in new_data[0]
    checked_flag = _mm256_cmp_pd(*data0, *data1, _CMP_LE_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[0] = _mm256_add_pd(tmp[0], tmp[1]);

    // minimum is stored in new_data[1]
    checked_flag = _mm256_cmp_pd(*data0, *data1, _CMP_GT_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[1] = _mm256_add_pd(tmp[0], tmp[1]);

//    printf("->\n");
//    _bncavx2_print_m256d_array(new_data, 2);

    // swap data0 and data1
    *data0 = new_data[0];
    *data1 = new_data[1];    
}

// https://fukushimalab.github.io/hpc_exercise/#%E6%B5%AE%E5%8B%95%E5%B0%8F%E6%95%B0%E7%82%B9%E3%81%AE%E7%B5%B6%E5%AF%BE%E5%80%A4%E3%81%AE%E8%A8%88%E7%AE%97
// return |org|
/*
static inline __m256d _bncavx2_fabs(__m256d org)
{
    __m256d rev_sign_mask = {0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
    return _mm256_and_pd(rev_sign_mask, org);
}
*/

// |data0[i]| >= |data1[i]|, i = 0, 1, 2, 3
static inline void _bncavx2_swap_absmax(__m256d *data0, __m256d *data1)
{
    __m256d checked_flag;
    __m256d new_data[2], abs_data[2], tmp[2];

    abs_data[0] = _bncavx2_fabs(*data0);
    abs_data[1] = _bncavx2_fabs(*data1);

    // maximum is stored in new_data[0]
    checked_flag = _mm256_cmp_pd(abs_data[0], abs_data[1], _CMP_GE_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[0] = _mm256_add_pd(tmp[0], tmp[1]);

    // minimum is stored in new_data[1]
    checked_flag = _mm256_cmp_pd(abs_data[0], abs_data[1], _CMP_LT_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[1] = _mm256_add_pd(tmp[0], tmp[1]);

//    printf("->\n");
//    _bncavx2_print_m256d_array(new_data, 2);

    // swap data0 and data1
    *data0 = new_data[0];
    *data1 = new_data[1];    
}

// |data0[i]| <= |data1[i]|, i = 0, 1, 2, 3
static inline void _bncavx2_swap_absmin(__m256d *data0, __m256d *data1)
{
    __m256d checked_flag;
    __m256d new_data[2], abs_data[2], tmp[2];

    abs_data[0] = _bncavx2_fabs(*data0);
    abs_data[1] = _bncavx2_fabs(*data1);

    // maximum is stored in new_data[0]
    checked_flag = _mm256_cmp_pd(abs_data[0], abs_data[1], _CMP_LE_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[0] = _mm256_add_pd(tmp[0], tmp[1]);

    // minimum is stored in new_data[1]
    checked_flag = _mm256_cmp_pd(abs_data[0], abs_data[1], _CMP_GT_OS);
    tmp[0] = _mm256_and_pd(checked_flag, *data0);
    tmp[1] = _mm256_andnot_pd(checked_flag, *data1);
    new_data[1] = _mm256_add_pd(tmp[0], tmp[1]);

//    printf("->\n");
//    _bncavx2_print_m256d_array(new_data, 2);

    // swap data0 and data1
    *data0 = new_data[0];
    *data1 = new_data[1];    
}


// data[i] >= data[i + 1]
// flag == 0 -> max order
// flag == 1 -> min order
// flag == 2 -> absmax order
// flag == 3 -> absmin order
#define _BNC_SORT_ORDER_MAX 0
#define _BNC_SORT_ORDER_MIN 1
#define _BNC_SORT_ORDER_ABSMAX 2
#define _BNC_SORT_ORDER_ABSMIN 3

static void _bncavx2_bubble_sort(__m256d data[], int num, int flag)
{
    int i, j;

    switch(flag)
    {
        // max order
        case _BNC_SORT_ORDER_MAX:
            for(i = num - 1; i >= 0; i--)
            {
                for(j = 0; j < i; j++)
                    _bncavx2_swap_max(&data[j], &data[j + 1]);
            }
            break;
        // max order
        case _BNC_SORT_ORDER_MIN:
            for(i = num - 1; i >= 0; i--)
            {
                for(j = 0; j < i; j++)
                    _bncavx2_swap_min(&data[j], &data[j + 1]);
            }
            break;
        // max order
        case _BNC_SORT_ORDER_ABSMAX:
            for(i = num - 1; i >= 0; i--)
            {
                for(j = 0; j < i; j++)
                    _bncavx2_swap_absmax(&data[j], &data[j + 1]);
            }
            break;
        // max order
        case _BNC_SORT_ORDER_ABSMIN:
            for(i = num - 1; i >= 0; i--)
            {
                for(j = 0; j < i; j++)
                    _bncavx2_swap_absmin(&data[j], &data[j + 1]);
            }
            break;
        default: 
            // max order
            for(i = num - 1; i >= 0; i--)
            {
                for(j = 0; j < i; j++)
                    _bncavx2_swap_max(&data[j], &data[j + 1]);
            }
            break;
    }
        
}
#endif //defined(__AVX2__)

#if 0 //def DEBUG
#if defined(__AVX2__) // __AVX2__
#define DIM 8

int main()
{
    __m256d a, b, c, zero, d[DIM], org_d[DIM];
    int i, j;

    zero = _mm256_setzero_pd();
 //   a = _mm256_set_pd(1.0, 2.0, 3.0, 4.0);
    a = _mm256_set_pd((double)rand(), (double)rand(), (double)rand(), (double)rand());
    b = _mm256_set_pd((double)rand(), (double)rand(), (double)rand(), (double)rand());
    c = _mm256_set_pd((double)rand(), (double)rand(), (double)rand(), (double)rand());

    for(i = 0; i < DIM; i++)
        org_d[i] = _mm256_set_pd(
            pow(-1.0, (double)rand()) * (double)rand(), 
            pow(-1.0, (double)rand()) * (double)rand(),
            pow(-1.0, (double)rand()) * (double)rand(),
            pow(-1.0, (double)rand()) * (double)rand()
        );

    printf("a = %g %g %g %g\n", a[0], a[1], a[2], a[3]);
    printf("b = %g %g %g %g\n", b[0], b[1], b[2], b[3]);

    c = _mm256_cmp_pd(b, a, _CMP_GE_OQ);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, b, _CMP_GE_OS);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, b, _CMP_NGE_UQ);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, b, _CMP_NGE_US);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    printf("comp_zero, zero = %g, %g, %g, %g\n", zero[0], zero[1], zero[2], zero[3]);
    c = _mm256_cmp_pd(zero, zero, _CMP_EQ_OQ);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, zero, _CMP_NEQ_OS);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, zero, _CMP_NEQ_UQ);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    c = _mm256_cmp_pd(a, zero, _CMP_NEQ_US);
    printf("c = %g %g %g %g\n", c[0], c[1], c[2], c[3]);

    // sort
    for(i = 0; i < DIM; i++) d[i] = org_d[i];
    _bncavx2_print_m256d_array(d, DIM);

    printf("MAX\n");
    _bncavx2_bubble_sort(d, DIM, _BNC_SORT_ORDER_MAX);
    _bncavx2_print_m256d_array(d, DIM);

    for(i = 0; i < DIM; i++) d[i] = org_d[i];
    printf("MIN\n");
    _bncavx2_bubble_sort(d, DIM, _BNC_SORT_ORDER_MIN);
    _bncavx2_print_m256d_array(d, DIM);

    for(i = 0; i < DIM; i++) d[i] = org_d[i];
    printf("ABSMAX\n");
    _bncavx2_bubble_sort(d, DIM, _BNC_SORT_ORDER_ABSMAX);
    _bncavx2_print_m256d_array(d, DIM);

    for(i = 0; i < DIM; i++) d[i] = org_d[i];
    printf("ABSMIN\n");
    _bncavx2_bubble_sort(d, DIM, _BNC_SORT_ORDER_ABSMIN);
    _bncavx2_print_m256d_array(d, DIM);

    return 0;
}
#endif // __AVX2__
#endif //0 // DEBUG

#endif // ifndef __BNC_SORT_H__
