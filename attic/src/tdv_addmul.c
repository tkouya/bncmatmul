/* add */
/*
static inline void _bncavx2_rtd_add(__m256d ret[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
    __m256d z[6], e[6];
    _bncavx2_merge(z, a, 3, b, 3);
    _bncavx2_vec_sum(e, z, 6);
    _bncavx2_vseb(ret, 3, e, 6);
}
*/
void _bncavx2_tdadd(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[3], in_a_val[3], in_b_val[3], in_z[6], in_e[6];

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

//        _bncavx2_rtd_add(in_ret, in_a_val, in_b_val);
       _bncavx2_rtd_addq(in_ret, in_a_val, in_b_val);

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
    }
}

void _bncavx2_tdvadd(TDVector ret, TDVector a, TDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[3], in_a_val[3], in_b_val[3], in_z[6], in_e[6];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));

//        _bncavx2_rtd_add(in_ret, in_a_val, in_b_val);
        _bncavx2_rtd_addq(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
        _mm256_store_pd(&ret->element[2][index], in_ret[2]);
   }
}

/* mul */
/*
// mul
static inline void _bncavx2_rtd_mul(__m256d ret[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
	__m256d z00[2], z01[2], z10[2];
	__m256d in_b[3], in_c, z[3], e[4], temp[4];

	z00[0] = _bncavx2_dtwo_prod(a[0], b[0], &z00[1]);
	z01[0] = _bncavx2_dtwo_prod(a[0], b[1], &z01[1]);
	z10[0] = _bncavx2_dtwo_prod(a[1], b[0], &z10[1]);

	z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];

	_bncavx2_vec_sum(in_b, z, 3);
	in_c = _mm256_fmadd_pd(a[1], b[1], in_b[2]);

	z[0] = _mm256_fmadd_pd(a[0], b[2], z10[1]);
	z[1] = _mm256_fmadd_pd(a[2], b[0], z01[1]);
    z[2] = _mm256_add_pd(z[0], z[1]);
	temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
    temp[3] = _mm256_add_pd(in_c, z[2]);
	_bncavx2_vec_sum(e, temp, 4);
	ret[0] = e[0];
	_bncavx2_vseb(&ret[1], 2, &e[1], 3);
}
*/
void _bncavx2_tdmul(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_a_val[3], in_b_val[3], in_ret[3];
	__m256d z00[2], z01[2], z10[2];
	__m256d in_b[3], in_c, z[3], e[4], temp[4];

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

/*
        z00[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[0], &z00[1]);
        z01[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[1], &z01[1]);
        z10[0] = _bncavx2_dtwo_prod(in_a_val[1], in_b_val[0], &z10[1]);

        z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];

        _bncavx2_vec_sum(in_b, z, 3);
        in_c = _mm256_fmadd_pd(in_a_val[1], in_b_val[1], in_b[2]);

        z[0] = _mm256_fmadd_pd(in_a_val[0], in_b_val[2], z10[1]);
        z[1] = _mm256_fmadd_pd(in_a_val[2], in_b_val[0], z01[1]);
        z[2] = _mm256_add_pd(z[0], z[1]);
        temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
        temp[3] = _mm256_add_pd(in_c, z[2]);
        _bncavx2_vec_sum(e, temp, 4);
        in_ret[0] = e[0];
        _bncavx2_vseb(&in_ret[1], 2, &e[1], 3);
*/
        _bncavx2_rtd_mul(in_ret, in_a_val, in_b_val);
//        _bncavx2_rtd_mulq(in_ret, in_a_val, in_b_val);

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

    }
}

void _bncavx2_tdvmul(TDVector ret, TDVector a, TDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_a_val[3], in_b_val[3], in_ret[3];
	__m256d z00[2], z01[2], z10[2];
	__m256d in_b[3], in_c, z[3], e[4], temp[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));

        _bncavx2_rtd_mul(in_ret, in_a_val, in_b_val);
        //_bncavx2_rtd_mulq(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
        _mm256_store_pd(&ret->element[2][index], in_ret[2]);
   }
 }