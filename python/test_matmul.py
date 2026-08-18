# test_matmul.py: Test script for MV and MM

from bncampy import *

# start
librdd.rdd_start()

mpfr_sqrt2 = gmpy2.sqrt(gmpy2.mpfr('2'))
mpfr_sqrt3 = gmpy2.sqrt(gmpy2.mpfr(3))
mpfr_sqrt5 = gmpy2.sqrt(gmpy2.mpfr(5))
mp_sqrt2 = mpmath.sqrt(mpmath.mp.mpf('2'))
mp_sqrt3 = mpmath.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.sqrt(mpmath.mp.mpf(5))

print(type(mpfr_sqrt2))

#sq_dim = 3
str_sq_dim = input('Dimension = ')
sq_dim = int(str_sq_dim)

# mpfr
row_dim = sq_dim
mid_dim = sq_dim
col_dim = sq_dim
dd_zero = rdd.dd_float(0.0, 0.0)

# A := sqrt(2) * [(i + j + 1)]
mpfr_mat_a = [mpfr_zero] * (row_dim * mid_dim)
print('size_mat_a = ', len(mpfr_mat_a))
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j
		mpfr_mat_a[ij_index] = mpfr_sqrt2 * (i + j + 1)

#print(mpfr_mat_a)

# B := sqrt(3) * [(i + j + 1)]
mpfr_mat_b = [mpfr_zero] * (mid_dim * col_dim)
print('size_mat_b = ', len(mpfr_mat_b))	
for i in range(mid_dim):
	for j in range(col_dim):
		ij_index = i * mid_dim + j
		mpfr_mat_b[ij_index] = mpfr_sqrt3 * (i + j + 1)

#print(mpfr_mat_b)

# C := 0
mpfr_mat_c = [mpfr_zero] * (row_dim * col_dim)

# MM
start_time = time.time()
mpfr_mat_c = xd_mymatmul(mpfr_mat_a, row_dim, mid_dim, mpfr_mat_b, mid_dim, col_dim, mpfr_zero)
end_time = time.time()
mpfr_matmul_time = end_time - start_time
#print(mpfr_mat_c)
print(xd_normf(mpfr_mat_c, row_dim, mid_dim, mpfr_zero))


# Convert A to MPFMatrix
mpfmat_a = init_mpfmatrix(row_dim, mid_dim)
mpfmat_b = init_mpfmatrix(mid_dim, col_dim)
mpfmat_c = init_mpfmatrix(row_dim, col_dim)

set_mpfmatrix_mpfr(mpfmat_a, mpfr_mat_a)
#print_mpfmatrix(mpfmat_a)
set_mpfmatrix_mpfr(mpfmat_b, mpfr_mat_b)
#print_mpfmatrix(mpfmat_b)

# MM
start_time = time.time()
mul_mpfmatrix(mpfmat_c, mpfmat_a, mpfmat_b)
end_time = time.time()
mpfmatmul_time = end_time - start_time

#mpfr_init(normf)
#normf_mpfmatrix(ct.byref(normf), mpfmat_c)

#print_mpfmatrix(mpfmat_c)
#libmpfr.mpfr_out_str(normf, 10, 0, 0)

print(f'mpfr_matmul vs. mul_mpfmatrix: , {mpfr_matmul_time:10.3g}, {mpfmatmul_time:10.3g}')

# time

librdd.rdd_end()
