static inline void _bncavx2_rtd_add(__m256d ret[TDSIZE], __m256d a[TDSIZE], __m256d b[TDSIZE])
{
    __m256d z[6], e[6];
    _bncavx2_merge(z, a, 3, b, 3);
    _bncavx2_vec_sum(e, z, 6);
    _bncavx2_vseb(ret, 3, e, 6);
}

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