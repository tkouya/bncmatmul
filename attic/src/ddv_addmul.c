/* double-double = double + double */
void _bncavx2_ddadd_d_d(ddfloat ret[], double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_dtwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        //_mm256_store_pd(&ret[index].val[0], s);
        //_mm256_store_pd(&ret[index].val[1], e);

        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
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

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index].val[0] = in_s[0]; ret[index].val[1] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        }
    }
//    printf("\n");
}

/* double-double = double + double */
//void _bncavx2_ddvadd_d_d(ddvector *ret, double a[], double b[], int dim)
void _bncavx2_ddvadd_d_d(DDVector ret, double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_dtwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_pd(&ret->element[0][index], s);
        _mm256_store_pd(&ret->element[1][index], e);
        /*
        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
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

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index] = in_s[0]; ret->element[1][index] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_sum(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
            ret->element[0][index + 2] = in_s[2]; ret->element[2][index + 1] = in_e[2];
        }
    }
//    printf("\n");
}

/* add */
void _bncavx2_ddadd(ddfloat ret[], ddfloat a[], ddfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

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

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
   }

}

void _bncavx2_ddvadd(DDVector ret, DDVector a, DDVector b, int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2], s1, s2, t1, t2;
    double in_s1[4], in_s2[4], in_t1[4], in_t2[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
   }
}

/* double-double = double * double */
void _bncavx2_ddvmul_d_d(DDVector ret, double a[], double b[], int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret, in_a, in_b, s, e;
    double in_s[4], in_e[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_pd(&a[index]);
        in_b = _mm256_load_pd(&b[index]);

       	//s = two_prod(a, b, &e);
        s = _bncavx2_dtwo_prod(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_pd(&ret->element[0][index], s);
        _mm256_store_pd(&ret->element[1][index], e);
        /*
        _mm256_storeu_pd(in_s, s);
        _mm256_storeu_pd(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
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

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index] = in_s[0]; ret->element[1][index] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_pd(0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_pd(0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_pd(0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = _bncavx2_dtwo_prod(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_pd(in_s, s);
            _mm256_storeu_pd(in_e, e);
            ret->element[0][index    ] = in_s[0]; ret->element[1][index    ] = in_e[0];
            ret->element[0][index + 1] = in_s[1]; ret->element[1][index + 1] = in_e[1];
            ret->element[0][index + 2] = in_s[2]; ret->element[2][index + 1] = in_e[2];
        }
    }
//    printf("\n");
}

/* ddmul */
void _bncavx2_ddmul(ddfloat ret[], ddfloat a[], ddfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

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

        _bncavx2_rdd_mul(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
   }

}

void _bncavx2_ddvmul(DDVector ret, DDVector a, DDVector b, int dim)
{
    int i, index, div, rem, unit = 4;
    __m256d in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_mul(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&ret->element[0][index], in_ret[0]);
        _mm256_store_pd(&ret->element[1][index], in_ret[1]);
   }
}