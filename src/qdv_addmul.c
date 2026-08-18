/* add */
void _bncavx2_qdadd(qdfloat ret[], qdfloat a[], qdfloat b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[4], in_a_val[4], in_b_val[4], s1, s2, t1, t2;
    double in_s1[4], in_s2[4], in_t1[4], in_t2[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_pd(
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_pd(
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_a_val[2] = _mm256_set_pd(
            a[index + 3].val[2],
            a[index + 2].val[2],
            a[index + 1].val[2],
            a[index    ].val[2]
        );
        in_a_val[3] = _mm256_set_pd(
            a[index + 3].val[3],
            a[index + 2].val[3],
            a[index + 1].val[3],
            a[index    ].val[3]
        );
        in_b_val[0] = _mm256_set_pd(
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_pd(
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );
        in_b_val[2] = _mm256_set_pd(
            b[index + 3].val[2],
            b[index + 2].val[2],
            b[index + 1].val[2],
            b[index    ].val[2]
        );
        in_b_val[3] = _mm256_set_pd(
            b[index + 3].val[3],
            b[index + 2].val[3],
            b[index + 1].val[3],
            b[index    ].val[3]
        );

        _bncavx2_rqd_add(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0];
        ret[index + 1].val[0] = in_ret[0][1];
        ret[index + 2].val[0] = in_ret[0][2];
        ret[index + 3].val[0] = in_ret[0][3];

        ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[1] = in_ret[1][3];

        ret[index    ].val[2] = in_ret[2][0];
        ret[index + 1].val[2] = in_ret[2][1];
        ret[index + 2].val[2] = in_ret[2][2];
        ret[index + 3].val[2] = in_ret[2][3];

        ret[index    ].val[3] = in_ret[3][0];
        ret[index + 1].val[3] = in_ret[3][1];
        ret[index + 2].val[3] = in_ret[3][2];
        ret[index + 3].val[3] = in_ret[3][3];
   }
}

void _bncavx2_qdvadd(QDVector ret, QDVector a, QDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[4], in_a_val[4], in_b_val[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_pd(&(a->element[3][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_pd(&(b->element[3][index]));

        _bncavx2_rqd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
        _mm256_store_pd(&ret->element[2][index], in_ret[2]);
        _mm256_store_pd(&ret->element[3][index], in_ret[3]);
   }

}

/* mul */
void _bncavx2_qdmul(qdfloat ret[], qdfloat a[], qdfloat b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[4], in_a_val[4], in_b_val[4], s1, s2, t1, t2;
    double in_s1[4], in_s2[4], in_t1[4], in_t2[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_pd(
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_pd(
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_a_val[2] = _mm256_set_pd(
            a[index + 3].val[2],
            a[index + 2].val[2],
            a[index + 1].val[2],
            a[index    ].val[2]
        );
        in_a_val[3] = _mm256_set_pd(
            a[index + 3].val[3],
            a[index + 2].val[3],
            a[index + 1].val[3],
            a[index    ].val[3]
        );
        in_b_val[0] = _mm256_set_pd(
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_pd(
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );
        in_b_val[2] = _mm256_set_pd(
            b[index + 3].val[2],
            b[index + 2].val[2],
            b[index + 1].val[2],
            b[index    ].val[2]
        );
        in_b_val[3] = _mm256_set_pd(
            b[index + 3].val[3],
            b[index + 2].val[3],
            b[index + 1].val[3],
            b[index    ].val[3]
        );

        _bncavx2_rqd_mul(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0];
        ret[index + 1].val[0] = in_ret[0][1];
        ret[index + 2].val[0] = in_ret[0][2];
        ret[index + 3].val[0] = in_ret[0][3];

        ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[1] = in_ret[1][3];

        ret[index    ].val[2] = in_ret[2][0];
        ret[index + 1].val[2] = in_ret[2][1];
        ret[index + 2].val[2] = in_ret[2][2];
        ret[index + 3].val[2] = in_ret[2][3];

        ret[index    ].val[3] = in_ret[3][0];
        ret[index + 1].val[3] = in_ret[3][1];
        ret[index + 2].val[3] = in_ret[3][2];
        ret[index + 3].val[3] = in_ret[3][3];

   }
}

void _bncavx2_qdvmul(QDVector ret, QDVector a, QDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[4], in_a_val[4], in_b_val[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_pd(&(a->element[3][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_pd(&(b->element[3][index]));

        _bncavx2_rqd_mul(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
        _mm256_store_pd(&ret->element[2][index], in_ret[2]);
        _mm256_store_pd(&ret->element[3][index], in_ret[3]);
   }

}